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
"""Postprocessors for T5X Retrieval."""

from typing import Mapping, Optional, Union

import tensorflow as tf


def extract_label_postprocessor(
    output: Mapping[str, tf.Tensor],
    example: Optional[Mapping[str, tf.Tensor]] = None,
    is_target: Optional[bool] = False
) -> Union[tf.Tensor, Mapping[str, tf.Tensor]]:
  """Extracts the label to feed into the SeqIO evaluator.

  Args:
    output: A mapping of strings and tensors.
    example: An optional mapping of strings and tensors.
    is_target: An optional variable to indicate whether the postprocessor is
      applied on the output or the target (i.e. the "labels" field.).

  Returns:
    The target tensor or the output mapping.
  """
  if is_target:
    return example["labels"]
  return output
