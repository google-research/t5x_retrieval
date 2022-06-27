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
"""Tests for utils."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
import seqio
from t5x_retrieval import utils
import tensorflow as tf
import tensorflow_datasets as tfds


class BatchUnmixedDatasetFnsTest(seqio.test_utils.FakeMixtureTest):

  def check_ds_shape(self, ds, batch_size, sequence_length):
    for k, v in tf.compat.v1.data.get_output_shapes(ds).items():
      feat = k.split("_")[0]
      if len(v) == 0:  # pylint:disable=g-explicit-length-test
        expected_shape = []
      elif feat in sequence_length:
        expected_shape = [batch_size, sequence_length[feat]]
      else:
        expected_shape = [None]
      self.assertEqual(expected_shape, v.as_list())

  def verify_batch_unmixed_dataset_fn(self, cfg, shard_id, num_shards,
                                      batch_size, sequence_length,
                                      feature_converter_cls):

    ds = utils.get_batch_unmixed_dataset(cfg, shard_id, num_shards,
                                         feature_converter_cls)

    # Verify the batch shapes.
    self.check_ds_shape(
        ds,
        batch_size,
        sequence_length={
            "encoder": sequence_length["inputs"],
            "decoder": sequence_length["targets"]
        })
    # Materialize a few batches to test for errors.
    list(zip(range(10), tfds.as_numpy(ds)))

  def test_get_batch_unmixed_dataset_fn(self):

    batch_size = 10
    shard_id = 0
    num_shards = 2
    sequence_length = {"inputs": 7, "targets": 9}
    cfg = utils.DatasetConfig(
        mixture_or_task_name="cached_mixture",
        task_feature_lengths=sequence_length,
        split="train",
        batch_size=batch_size,
        shuffle=False,
        seed=None)

    self.verify_batch_unmixed_dataset_fn(cfg, shard_id, num_shards,
                                         batch_size // 2, sequence_length,
                                         seqio.EncDecFeatureConverter)


class UtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="logits_of_2x2",
          logits=np.ones((2, 2)),
          expected_labels=[0, 1]),
      dict(
          testcase_name="logits_of_5x5",
          logits=np.ones((5, 5)),
          expected_labels=[0, 1, 2, 3, 4]),
      dict(
          testcase_name="logits_of_2x5",
          logits=np.ones((2, 5)),
          expected_labels=[0, 1]))
  def test_sparse_labels_for_in_batch_cross_entropy(self, logits: jnp.array,
                                                    expected_labels: jnp.array):
    np.testing.assert_allclose(
        utils.sparse_labels_for_in_batch_cross_entropy(logits), expected_labels)

  def test_in_batch_cross_entropy(self):
    # Test row loss without label and weight.
    row_loss = utils.in_batch_cross_entropy(
        logits=np.array([[1, 0], [0, 1]]), reduce_fn=None)
    row_loss_worse1 = utils.in_batch_cross_entropy(
        logits=np.array([
            [0, 1],  # logits[0:] has lower positive than negative
            [0, 1]
        ]),
        reduce_fn=None)
    row_loss_worse2 = utils.in_batch_cross_entropy(
        logits=np.array([
            [1, 0],  # logits[0:] has lower positive than negative
            [1, 0]
        ]),
        reduce_fn=None)
    self.assertEqual(row_loss.shape, (2,))
    self.assertEqual(row_loss_worse1.shape, (2,))
    self.assertEqual(row_loss_worse2.shape, (2,))
    self.assertLess(row_loss[0], row_loss_worse1[0])
    self.assertEqual(row_loss[1], row_loss_worse1[1])
    self.assertEqual(row_loss[0], row_loss_worse2[0])
    self.assertEqual(row_loss_worse1[0], row_loss_worse2[1])

    # Specify labels instead of using the default.
    row_loss_equal = utils.in_batch_cross_entropy(
        logits=np.array([[0, 1], [0, 1]]),
        labels=np.array([[0, 1], [0, 1]]),  # note both example have label 1.
        reduce_fn=None)
    np.testing.assert_allclose(row_loss_equal, row_loss)

    # Test scalar loss.
    scalar_loss_mean = utils.in_batch_cross_entropy(
        logits=np.array([[1, 0], [0, 1]]), reduce_fn=jnp.mean)
    scalar_loss_sum = utils.in_batch_cross_entropy(
        logits=np.array([[1, 0], [0, 1]]), reduce_fn=jnp.sum)
    scalar_loss_worse = utils.in_batch_cross_entropy(
        logits=np.array([
            [0, 1],  # logits[0:] has lower positive than negative
            [0, 1]
        ]),
        reduce_fn=jnp.mean)

    self.assertLess(scalar_loss_mean, scalar_loss_worse)
    self.assertEqual(2.0 * scalar_loss_mean, scalar_loss_sum)

  @parameterized.named_parameters(
      dict(
          testcase_name="logits_of_2x2",
          logits=np.arange(4).reshape(2, 2),
          expected_rr=[0.5, 1.]),
      dict(
          testcase_name="logits_of_5x5",
          logits=np.arange(25).reshape(5, 5),
          expected_rr=[0.2, 0.25, 0.333, 0.5, 1.]),
      dict(
          testcase_name="logits_of_2x5",
          logits=np.arange(10).reshape(2, 5),
          expected_rr=[0.2, 0.25]))
  def test_compute_rr(self, logits, expected_rr):
    np.testing.assert_allclose(
        utils.compute_rr(
            logits, utils.sparse_labels_for_in_batch_cross_entropy(logits)),
        expected_rr,
        rtol=0.01)


class CheckpointUtilsTest(parameterized.TestCase):

  def test_partially_load_checkpoint(self):
    test_checkpoint = {
        "foo": 123,
        "bar": {
            "include": 123,
            "exclude": 456
        },
        "exclude_key": 123
    }
    state_transformation_fn = utils.partially_load_checkpoint([r".*exclude.*"])
    actual = state_transformation_fn(test_checkpoint, {})
    self.assertEqual(actual, {"foo": 123, "bar": {"include": 123}})
    self.assertNotEqual(actual, test_checkpoint)

  @parameterized.named_parameters(
      dict(testcase_name="load_left_tower", side="left"),
      dict(testcase_name="load_right_tower", side="right"))
  def test_load_tower(self, side):
    test_checkpoint = {
        "optimizer": {
            "targets": {
                "left_encoder": {
                    "layer_1": 1,
                    "layer_2": 2,
                    "layer_3": 3
                },
                "right_encoder": {
                    "layer_1": 1,
                    "layer_2": 2,
                    "layer_3": 3
                },
                "left_projection_layer": 100,
                "right_projection_layer": 100,
            },
            "states": {
                "params": {
                    "left_encoder": {
                        "m": 1,
                        "v": 2,
                    },
                    "right_encoder": {
                        "m": 1,
                        "v": 2,
                    },
                },
                "step": 100
            },
        }
    }
    state_transform_fn = utils.load_tower(side)
    actual = state_transform_fn(test_checkpoint, {})
    # Test checkpoint should not be modified
    self.assertNotEqual(actual, test_checkpoint)
    self.assertEqual(
        actual, {
            "optimizer": {
                "targets": {
                    f"{side}_encoder": {
                        "layer_1": 1,
                        "layer_2": 2,
                        "layer_3": 3
                    },
                    f"{side}_projection_layer": 100,
                },
                "states": {
                    "params": {
                        f"{side}_encoder": {
                            "m": 1,
                            "v": 2,
                        },
                    },
                    "step": 100
                },
            }
        })


if __name__ == "__main__":
  absltest.main()
