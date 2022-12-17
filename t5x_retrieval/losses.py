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
"""Loss layer implementations for dual encoders."""
import abc
from typing import Mapping, Optional, Tuple

import clu.metrics as clu_metrics
from flax import linen as nn
from flax.core import scope as flax_scope
import gin
from jax import numpy as jnp
import seqio
from t5x import metrics as t5x_metrics
from t5x import models as t5x_models
from t5x_retrieval import feature_converters
from t5x_retrieval import utils

FeatureSpec = feature_converters.FeatureSpec
SeqIOFeatureSpec = seqio.feature_converters.FeatureConverter.FeatureSpec


@gin.configurable
class DualEncoderLoss(nn.Module, abc.ABC):
  """Base class for loss layers accepted by dual encoders.

  """

  @property
  @abc.abstractmethod
  def LOSS_MODEL_FEATURES(self) -> Mapping[str, SeqIOFeatureSpec]:
    """Model features required by loss layer for computing loss/metrcs.

    This property specifies the features that are expected to be present in
    `batch` argument passed to the __call__ method by a dual encoder model.
    These must be described as a map of feature keys to SeqIO `FeatureSpec`s,
    following the format of the `MODEL_FEATURES` attribute of SeqIO's
    `FeatureConverter`s. The features specified here in this manner should be
    validated by `validate_model_features` method (the default base class
    implementation does this).
    """
    pass

  @nn.compact
  @abc.abstractmethod
  def __call__(self, batch: Mapping[str, jnp.ndarray], logits: jnp.ndarray,
               **kwargs) -> Tuple[jnp.float64, t5x_metrics.MetricsMap]:
    """Computes loss and loss-related metrics on the inputs.

    Args:
      batch: Features passed by dual-encoder model. Any external supervision
        signals/labels should be passed through this argument.
      logits: Output of dual-encoder model's similarity layer.
      **kwargs: Additional arguments that may be needed by the call method

    Returns:
      Tuple of (loss, metrics)
    """
    pass

  @abc.abstractmethod
  def get_initial_variables(self, rng, input_shapes,
                            input_types) -> flax_scope.FrozenVariableDict:
    """Gets the initial variables for a loss layer."""
    pass

  def validate_model_features(self, feature_converter: seqio.FeatureConverter):
    """Ensures loss-layer's required features are provided by feature converter.

    This method checks that the feature-spec for loss model features match. Any
    additional validation should be performed by child classes.

    Args:
      feature_converter: Feature converter used by the dual encoder model that
        invokes this loss layer.

    Raises:
      ValueError, if feature is missing or has a spec mismatch
    """
    model_features = feature_converter.MODEL_FEATURES
    for name, feature in self.LOSS_MODEL_FEATURES.items():
      model_feature = model_features.get(name, None)
      if not model_feature:
        raise ValueError(f"Missing required loss-layer feature {name} "
                         f"in model features {model_features}")
      if not isinstance(feature, type(model_feature)):
        raise ValueError(
            "Found incorrect type for feature spec of loss layer feature ",
            f"{name}, expected {type(model_feature)}, found {type(feature)}.")
      if feature != model_feature:
        raise ValueError("Found incorrect feature spec for loss layer feature "
                         f"{name}, expected {feature}, found {model_feature}")


class InBatchCrossEntropyLoss(DualEncoderLoss):
  """Dual encoder in-batch cross-entropy loss implementation.

  Attributes:
    bidirectional: Whether to use bi-directional in-batch softmax loss. If set
      to True, consider both left-to-right and right-to-left losses.
    label_smoothing: Label smoothing constant, used to determine the on and off
      values.
  """

  bidirectional: bool = True
  label_smoothing: float = 0.0

  @property
  def LOSS_MODEL_FEATURES(self):
    """Model features required by loss layer for computing loss/metrcs."""
    # This loss relies only on in-batch logits, and therefore doesn't
    # need any additional model features
    return {}

  def get_initial_variables(self, rng, input_shapes,
                            input_types) -> flax_scope.FrozenVariableDict:
    """Gets the initial variables for a loss layer."""
    # `logits` is of shape [B, B*(1+num_negatives)] that considers the
    # negatives while `right_logits` is in shape [B, B] that doesn't considers
    # negatives. `num_negatives` could be greater than 1 in the future.
    left_encoder_shape = input_shapes["left_encoder_input_tokens"]
    batch_size = left_encoder_shape[0]

    num_negatives = 0
    if "right_negative_encoder_input_tokens" in input_shapes:
      # right_negative_encoder_input_tokens: batch_size x num_negatives
      num_negatives = input_shapes["right_negative_encoder_input_tokens"][1]

    return self.init(
        rng,
        params={},
        batch={},
        logits=jnp.ones([batch_size, (batch_size * (1 + num_negatives))]),
        right_logits=jnp.ones([batch_size, batch_size]))

  @nn.compact
  def __call__(self,
               batch: Mapping[str, jnp.ndarray],
               logits: jnp.ndarray,
               right_logits: Optional[jnp.ndarray] = None,
               **kwargs) -> Tuple[jnp.float64, t5x_metrics.MetricsMap]:
    """Computes loss and loss-related metrics on inputs.

    `logits` is of shape [B, B*(1+num_negatives)] that considers the
    negatives while `right_logits` is in shape [B, B] that doesn't considers
    negatives. `num_negatives` could be greater than 1 in the future.

    Args:
      batch: Features passed by dual-encoder model. Unused.
      logits: Output of similarity layer that considers negatives. Has shape [B,
        B*(1+num_negatives)].
      right_logits: Output of similarity layer that doesn't consider negatives.
        Has shape [B, B]. If None, the right loss is skipped.
      **kwargs: Unused.

    Returns:
      loss: a float scalar for contrastive loss.
      metrics: metrics defined in `t5x_models.compute_base_metrics` and `MRR`.
    """
    del kwargs
    del batch  # we don't require any external inputs for this loss

    # z_loss is already added to loss, which is a workaround for the numerical
    # instability issue.
    z_loss = 0.
    loss = utils.in_batch_cross_entropy(
        logits, label_smoothing=self.label_smoothing)
    if right_logits is not None and self.bidirectional:
      right_loss = utils.in_batch_cross_entropy(
          right_logits, label_smoothing=self.label_smoothing)
      loss = jnp.mean(loss + right_loss)

    metrics = t5x_models.compute_base_metrics(
        logits=logits,
        targets=utils.sparse_labels_for_in_batch_cross_entropy(logits),
        mask=None,
        loss=loss,
        z_loss=z_loss)

    return loss, metrics


