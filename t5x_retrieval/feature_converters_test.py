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

"""Tests for seqio.feature_converters."""

import seqio
from t5x_retrieval import feature_converters
import tensorflow.compat.v2 as tf
tf.compat.v1.enable_eager_execution()

test_utils = seqio.test_utils
assert_dataset = test_utils.assert_dataset
create_default_dataset = test_utils.create_default_dataset


class DualEncoderFeatureConverterTest(tf.test.TestCase):

  def test_dual_encoder_unpacked(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 7, "targets": 5}
    feature_specs = [("inputs", "int32", 1, 0), ("targets", "int32", 1, 0)]
    input_features = feature_converters.FeatureSpec.to_map(
        feature_specs).values()

    converter = feature_converters.DualEncoderFeatureConverter(
        input_features, pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "left_encoder_input_tokens": [9, 4, 3, 8, 1, 0, 0],
        "right_encoder_input_tokens": [3, 9, 4, 1, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_dual_encoder_with_negative(self):
    x = [{"inputs": [9, 4, 3, 8, 1],
          "targets": [3, 9, 4, 1],
          "negative_targets": [[1, 3, 1], [2, 4, 1]]}]
    ds = create_default_dataset(
        x,
        feature_names={"inputs", "targets", "negative_targets"},
        output_shapes={"inputs": [None],
                       "targets": [None],
                       "negative_targets": [2, 3]})
    task_feature_lengths = {"inputs": 7, "targets": 5, "negative_targets": 3}
    feature_specs = [("inputs", "int32", 1, 0),
                     ("targets", "int32", 1, 0),
                     ("negative_targets", "int32", 2, 1)]
    input_features = feature_converters.FeatureSpec.to_map(
        feature_specs).values()

    converter = feature_converters.DualEncoderFeatureConverter(
        input_features, pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "left_encoder_input_tokens": [9, 4, 3, 8, 1, 0, 0],
        "right_encoder_input_tokens": [3, 9, 4, 1, 0],
        "right_negative_encoder_input_tokens": [[1, 3, 1],
                                                [2, 4, 1]]
    }
    assert_dataset(converted_ds, expected)

  def test_dual_encoder_with_labels_unpacked(self):
    x = [{
        "inputs": [9, 4, 3, 8, 1],
        "targets": [3, 9, 4, 1],
        "labels": [1.0],
        "loss_weights": [1.0, 0.0]
    }]
    output_types = {
        "inputs": tf.int32,
        "targets": tf.int32,
        "labels": tf.float32,
        "loss_weights": tf.float32
    }
    ds = create_default_dataset(
        x,
        feature_names=("inputs", "targets", "labels", "loss_weights"),
        output_types=output_types)
    task_feature_lengths = {
        "inputs": 7,
        "targets": 5,
        "labels": 1,
        "loss_weights": 2
    }
    feature_specs = [("inputs", "int32", 1, 0), ("targets", "int32", 1, 0),
                     ("labels", "float32", 1, 0),
                     ("loss_weights", "float32", 1, 0)]
    input_features = feature_converters.FeatureSpec.to_map(
        feature_specs).values()

    converter = feature_converters.DualEncoderFeatureConverter(
        input_features, pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "left_encoder_input_tokens": [9, 4, 3, 8, 1, 0, 0],
        "right_encoder_input_tokens": [3, 9, 4, 1, 0],
        "labels": [1.0],
        "loss_weights": [1.0, 0.0],
    }
    assert_dataset(converted_ds, expected)

  def test_dual_encoder_extra_long_inputs(self):
    x = [{"inputs": [9, 4, 3, 8, 4, 5, 1], "targets": [3, 9, 4, 7, 8, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 5, "targets": 8}
    feature_specs = [("inputs", "int32", 1, 0), ("targets", "int32", 1, 0)]
    input_features = feature_converters.FeatureSpec.to_map(
        feature_specs).values()

    expected_msg = (
        r".*Feature \\'inputs\\' has length not less than or equal to the "
        r"expected length of 5 during input_validation.*")
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, expected_msg):
      converter = feature_converters.DualEncoderFeatureConverter(
          input_features, pack=False)
      converted_ds = converter(ds, task_feature_lengths)
      list(converted_ds.as_numpy_iterator())




if __name__ == "__main__":
  tf.test.main()
