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
"""Preprocessors for T5X Retrieval."""

import tensorflow as tf


def to_stsb_label(dataset: tf.data.Dataset, label_field_name: str,
                  label_type: str) -> tf.data.Dataset:
  """Converts the labels to scores within [0, 1] or multi class labels.

  Args:
    dataset: A TensorFlow dataset.
    label_field_name: A string of the label field name.
    label_type: A string indicating the label type.

  Returns:
    A TensorFlow dataset after the transformation.
  """

  def map_fn(example):
    if label_type == "score":
      label = example[label_field_name] / 5
    elif label_type == "multi_class":
      label = example[label_field_name]
      label = tf.round(label * 2)
    else:
      raise ValueError(f"Unsupported label type: {label_type}")
    label = tf.expand_dims(label, axis=-1)
    example[label_field_name] = label
    return example

  return dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
