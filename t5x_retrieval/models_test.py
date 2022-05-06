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

"""Tests for T5X retrieval models."""
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import t5.data.tasks  # pylint:disable=unused-import
from t5x import partitioning
from t5x import trainer as trainer_lib
from t5x import utils
from t5x_retrieval import models
from t5x_retrieval import partitioning as t5xr_partitioning
from t5x_retrieval import test_utils
import tensorflow as tf

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

PartitionSpec = partitioning.PartitionSpec

class DualEncoderModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='dual_encoder', with_negatives=False),
      dict(testcase_name='dual_encoder_with_negatives', with_negatives=True))
  def test_loss_fn(self, with_negatives):
    left_encoder_input_tokens = jnp.ones((2, 3))
    right_encoder_input_tokens = jnp.ones((2, 3))
    left_logits = jnp.ones((2, 2))
    right_logits = jnp.ones((2, 2))
    variables = {}
    logits = jnp.ones((2, 2))
    params = {'foo': jnp.zeros(3)}

    mock_transformer = mock.Mock()
    mock_transformer.use_negatives = with_negatives
    mock_transformer.apply.return_value = (left_logits, right_logits,
                                           logits), variables
    mock_transformer.dtype = jnp.float32

    batch = {
        'left_encoder_input_tokens': left_encoder_input_tokens,
        'right_encoder_input_tokens': right_encoder_input_tokens,
    }

    def mock_init(self):
      self.module = mock_transformer
      # Follow the default argument value for DualEncoderModel __init__().
      self._inference_mode = 'encode'
      self._use_negatives = False
      self._use_align_uniform = False
      self._logit_scale = 100
      self._logit_margin = 0.0

    with mock.patch.object(models.DualEncoderModel, '__init__', new=mock_init):
      model = models.DualEncoderModel()
      res, _ = model.loss_fn(params, batch, dropout_rng=None)

    mock_transformer.apply.assert_called_with({'params': params},
                                              left_encoder_input_tokens,
                                              right_encoder_input_tokens,
                                              enable_dropout=False,
                                              rngs=None,
                                              mutable='dropout')
    np.testing.assert_allclose(res, 1.386294, atol=1e-5)

  def test_score_batch(self):
    left_encoder_input_tokens = jnp.ones((2, 3))
    right_encoder_input_tokens = jnp.ones((2, 3))
    encoding = jnp.ones((2, 2))
    params = {'foo': jnp.zeros(3)}

    mock_transformer = mock.Mock()
    mock_transformer.use_negatives = False
    mock_transformer.apply.return_value = encoding
    mock_transformer.dtype = jnp.float32

    batch = {
        'left_encoder_input_tokens': left_encoder_input_tokens,
        'right_encoder_input_tokens': right_encoder_input_tokens,
    }

    def mock_init(self):
      self.module = mock_transformer
      # Follow the default argument value for DualEncoderModel __init__().
      self._inference_mode = 'encode'
      self._use_negatives = False
      self._use_align_uniform = False
      self._logit_scale = 100
      self._logit_margin = 0.0

    with mock.patch.object(
        models.DualEncoderModel, '__init__', new=mock_init):
      model = models.DualEncoderModel()
      res = model.score_batch(params, batch)

    mock_transformer.apply.assert_called_with(
        {'params': params},
        left_encoder_input_tokens,
        enable_dropout=False,
        method=mock_transformer.encode)
    np.testing.assert_allclose(res, encoding, atol=1e-6)

  def test_train_dual_encoder(self):
    # Dummy input data
    input_shape = (16, 8)
    left_encoder_input_tokens = np.ones(shape=input_shape, dtype=np.float32)
    right_encoder_input_tokens = np.ones(shape=input_shape, dtype=np.float32)

    input_data = {
        'left_encoder_input_tokens': left_encoder_input_tokens,
        'right_encoder_input_tokens': right_encoder_input_tokens,
    }

    partitioner = partitioning.PjitPartitioner(
        num_partitions=1,
        model_parallel_submesh=None,
        logical_axis_rules=partitioning.standard_logical_axis_rules(
            additional_rules=t5xr_partitioning.standard_logical_axis_rules()))

    task_feature_lengths = "{'inputs': 5, 'targets': 10}"
    model = test_utils.get_tiny_de_t5_model(
        gin_str=f"""TASK_FEATURE_LENGTHS = {task_feature_lengths}""")

    ds_iter = tf.data.Dataset.from_tensors(input_data).as_numpy_iterator()
    input_shapes = {k: input_shape for k in input_data}
    train_state_initializer = utils.TrainStateInitializer(
        optimizer_def=model.optimizer_def,
        init_fn=model.get_initial_variables,
        input_shapes=input_shapes,
        partitioner=partitioner)
    train_state_axes = train_state_initializer.train_state_axes
    train_state = train_state_initializer.from_scratch(jax.random.PRNGKey(0))

    trainer = trainer_lib.Trainer(
        model,
        train_state=train_state,
        partitioner=partitioner,
        eval_names=[],
        summary_dir=None,
        train_state_axes=train_state_axes,
        rng=jax.random.PRNGKey(0),
        learning_rate_fn=lambda x: 0.001,
        num_microbatches=1)

    trainer.train(ds_iter, 1)
    self.assertEqual(trainer.train_state.step, 1)


if __name__ == '__main__':
  absltest.main()
