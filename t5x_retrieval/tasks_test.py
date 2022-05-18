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

"""Tests for tasks."""

# pylint: disable=unused-import

from absl.testing import absltest
from absl.testing import parameterized

import seqio
from t5x_retrieval import tasks  # pylint:disable=unused-import
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.enable_eager_execution()

_TASKS = ['beir_msmarco_retrieval']


class TasksTest(parameterized.TestCase):
  """This test ensures that the task.py file can be loaded without errors.

  It does not ensure that every single task/mixture works.
  """

  @parameterized.parameters(((name,) for name in _TASKS))
  def test_beir_msmarco_tasks(self, name):
    task = seqio.TaskRegistry.get(name)
    self.assertIn('train', task.splits)


if __name__ == '__main__':
  absltest.main()
