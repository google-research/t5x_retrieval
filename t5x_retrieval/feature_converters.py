# Copyright 2022 The T5X Retrieval Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Feature converter for dual encoders."""

import dataclasses
from typing import Any, Iterable, Mapping, Sequence, Tuple
import ml_collections
import seqio
import tensorflow as tf

FeatureSpecConfig = Tuple[str, str, int, int]

utils = seqio.utils
FeatureConverter = seqio.FeatureConverter


_MODEL_FEATURES_MAPPING = {
    "inputs": "left_encoder_input_tokens",
    "targets": "right_encoder_input_tokens",
    "negative_targets": "right_negative_encoder_input_tokens",
    "soft_label_pos": "soft_label_pos",
    "soft_label_neg": "soft_label_neg",
    "labels": "labels",
    "loss_weights": "loss_weights",
}


@dataclasses.dataclass
class FeatureSpec:
  """Container class for a feature's name, dtype, and rank."""
  name: str
  dtype: tf.DType
  rank: int
  sequence_dim: int

  @staticmethod
  def to_map(
      feature_specs: Iterable[FeatureSpecConfig]) -> Mapping[str, Any]:
    feature_spec_map = {}
    for name, dtype_str, rank, sequence_dim in feature_specs:
      feature_spec_map[name] = FeatureSpec(
          name, getattr(tf, dtype_str), rank, sequence_dim)
    return feature_spec_map


class DualEncoderFeatureConverterFactory(object):
  """Factory for dual encoder feature converters."""

  def __init__(self, feature_specs: Iterable[FeatureSpecConfig],
               is_multimodal: bool = False):
    self.feature_specs = feature_specs
    self.is_multimodal = is_multimodal

  def __call__(self,
               pack: bool = False,
               use_custom_packing_ops: bool = False):
    feature_spec_map = FeatureSpec.to_map(self.feature_specs)
    return DualEncoderFeatureConverter(
        input_features=feature_spec_map.values(),
        is_multimodal=self.is_multimodal,
        pack=pack,
        use_custom_packing_ops=use_custom_packing_ops)


class DualEncoderFeatureConverter(FeatureConverter):
  """Feature converter for dual-encoder achitecture.

  The inputs and targets to the dual-encoder are sent to the left and right
  encoders separately.

  Attributes:
    input_features: a list of feature specs that are used to define the
    feature's name, dtype and rank.
    is_multimodal: a boolean variable to indicate whether it is a feature
    converter to multimodal inputs. For multimodal inputs, we don't use the
    default MODEL_FEATURES_MAPPING.
  """
  input_features: Sequence[FeatureSpec] = ()
  is_multimodal: bool = False

  def __init__(self,
               input_features: Sequence[FeatureSpec],
               is_multimodal: bool = False,
               pack: bool = False,
               use_custom_packing_ops: bool = True):
    self.input_features = input_features
    self.is_multimodal = is_multimodal
    # NOTE: for multimodal inputs, make sure the inputs either (1) include both
    # "left_" and "right_" features for training or (2) don't include any of
    # them for inference time.
    if self.is_multimodal:
      has_left, has_right = False, False
      for f in self.input_features:
        if "left_" in f.name:
          has_left = True
        elif "right_" in f.name:
          has_right = True
      if (has_left and not has_right) or (not has_left and has_right):
        raise ValueError(
            "Multimodal inputs features should have both left and right tower"
            "features for training."
        )
    super().__init__(pack=pack, use_custom_packing_ops=use_custom_packing_ops)

  @property
  def TASK_FEATURES(self):
    feature_specs_map = {
        f.name: seqio.FeatureConverter.FeatureSpec(
            dtype=f.dtype, rank=f.rank, sequence_dim=f.sequence_dim)
        for f in self.input_features
    }
    return feature_specs_map

  # NOTE: only use the _MODEL_FEATURES_MAPPING for non-multimodal inputs.
  @property
  def MODEL_FEATURES(self):
    feature_specs_map = {}
    for f in self.input_features:
      name = f.name if self.is_multimodal else _MODEL_FEATURES_MAPPING[f.name]
      feature_specs_map[name] = seqio.FeatureConverter.FeatureSpec(
          dtype=f.dtype, rank=f.rank, sequence_dim=f.sequence_dim)
    return feature_specs_map

  @property
  def PACKING_FEATURE_DTYPES(self):
    return None

  def _convert_features(self, ds: tf.data.Dataset,
                        input_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Convert the input dataset to an output dataset to be fed to the model.

    The conversion process involves three steps

    1. Each feature in the `input_lengths` is padded.
    2. "inputs" fields are mapped to the left encoder input and "targets" are
       mapped to right encoder input.

    Assume the input dataset has two examples each with "inputs" and "targets".

    ds = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
          {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]

    task_feature_lengths = {"inputs": 8, "targets": 4}

    First, the `inputs` are padded to length 8 and assigned to
    "left_encoder_input_tokens" field. The `targets` are processed similarly.

    converted_ds = [
    {
         "left_encoder_input_tokens": [7, 8, 5, 1, 0, 0, 0, 0],
         "right_encoder_input_tokens": [3, 9, 1, 0],
    },
    {
         "left_encoder_input_tokens": [8, 4, 9, 3, 1, 0, 0, 0],
         "right_encoder_input_tokens": [4, 1, 0, 0],
    },
         ]

    Args:
      ds: an input tf.data.Dataset to be converted.
      input_lengths: a mapping from a feature to its length

    Returns:
      ds: the converted dataset.
    """

    def convert_example(
        features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
      d = {}
      for f in self.input_features:
        name = f.name if self.is_multimodal else _MODEL_FEATURES_MAPPING[f.name]
        d[name] = features[f.name]
      return d

    if self.pack:
      raise ValueError(
          "Dual encoder only takes non packed examples at this moment."
      )

    # Stop padding features with rank > 1, since _pack_or_pad adds padding to
    # the first dimension instead of the last dimension.
    for f_name in input_lengths:
      if f_name in self.TASK_FEATURES and self.TASK_FEATURES[f_name].rank > 1:
        # input should already be padded and dense.
        input_lengths = dict(input_lengths)
        if isinstance(input_lengths, ml_collections.ConfigDict):
          input_lengths.unlock()
        del input_lengths[f_name]

    ds = self._pack_or_pad(ds, input_lengths)
    return ds.map(
        convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""
    model_feature_lengths = {}
    for k in self.TASK_FEATURES:
      model_feature = k if self.is_multimodal else _MODEL_FEATURES_MAPPING[k]
      model_feature_lengths[model_feature] = task_feature_lengths[k]
    if self.pack:
      raise ValueError("Packing not supported")

    return model_feature_lengths


