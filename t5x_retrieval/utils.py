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

"""Utility functions for training and evaluation.

"""

import time
from typing import Callable, Optional, Type

import clu.metrics
import flax
import jax
from jax import lax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import seqio
import sklearn.metrics
from t5x import losses as t5x_losses
from t5x import utils as t5x_utils
import tensorflow as tf

DatasetConfig = t5x_utils.DatasetConfig


# ===== Datasets ===== #
def get_batch_unmixed_dataset(
    cfg: DatasetConfig,
    shard_id: int,
    num_shards: int,
    feature_converter_cls: Type[seqio.FeatureConverter],
    num_epochs: Optional[int] = None,
    continue_from_last_checkpoint: bool = False) -> tf.data.Dataset:
  """Returns a dataset by sampling each batch from each single task."""
  if continue_from_last_checkpoint:
    raise ValueError(
        '`continue_from_last_checkpoint` must be set to False as this is not '
        'supported by this dataset fn.')
  del continue_from_last_checkpoint

  if cfg.batch_size % num_shards:
    raise ValueError(
        f'Batch size ({cfg.batch_size}) must be divisible by number of '
        f'shards ({num_shards}).')

  shard_info = seqio.ShardInfo(index=shard_id, num_shards=num_shards)

  if cfg.seed is None:
    # Use a shared timestamp across devices as the seed.
    seed = multihost_utils.broadcast_one_to_all(np.int32(time.time()))
  else:
    seed = cfg.seed

  num_epochs = None  # repeat indefinitely.

  mixture_or_task = seqio.get_mixture_or_task(cfg.mixture_or_task_name)
  if not isinstance(mixture_or_task, seqio.Mixture):
    raise ValueError('Only SeqIO Mixture supports batch unmixed data accesss')

  datasets = []
  rates = []
  for task in mixture_or_task.tasks:
    cfg.mixture_or_task_name = task.name
    # Returns a batched dataset.
    datasets.append(
        t5x_utils.get_dataset_inner(cfg, shard_info, feature_converter_cls,
                                    seed, num_epochs))
    rates.append(mixture_or_task.get_rate(task))
  return tf.data.experimental.sample_from_datasets(datasets, rates)


# ===== Losses ===== #
# More details about alignment and uniformity loss can be found at
# https://arxiv.org/pdf/2005.10242.pdf. They measure the quality of embeddings.
def compute_align_loss(x, y, alpha=2):
  loss = jnp.linalg.norm(x - y, ord=2, axis=1)
  return lax.pow(loss, 1.0 * alpha).mean()


def compute_uniform_loss(xs, t=2):
  """Computes the euclidean distance between every pair of row vectors in the input."""
  distance_kernel = lambda x, y: jnp.sqrt(jnp.sum((x - y)**2))
  loss = jax.vmap(lambda x: jax.vmap(lambda y: distance_kernel(x, y))(xs))(xs)
  return jnp.log(jnp.exp(-t * lax.pow(loss, 2.0)).mean())


def sigmoid_cross_entropy_with_logits(logits, labels):
  """Compute binary cross entropy loss with logits input.

  Args:
    logits: similarity scores
    labels: ground-truth labels
  Returns:
    cross-entropy loss
  """

  x, z = logits, labels
  # Follow the implementation in Tensorflow:
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_impl.py#L186
  loss = jnp.maximum(x, 0) - x * z + jnp.log(1 + jnp.exp(-jnp.absolute(x)))
  return loss


def apply_temperature(probs, temperature):
  """Apply a temperature to probabilities."""
  if temperature <= 0:
    raise ValueError('Temperature must be positive.')
  inv_temp = 1.0 / temperature
  x = lax.pow(probs, inv_temp)
  y = lax.pow(1 - probs, inv_temp)
  return x / (x + y)


def binary_cross_entropy_with_logits(logits,
                                     labels,
                                     weights=None,
                                     temperature=1.0):
  """Binary cross-entropy loss function.

  This loss is used for point-wise distillation where the goal is to match the
  point-wise logits for each example to match the gold labels. The temperature
  is commonly added on both the logits and the labels as described in previous
  distillation methods: https://arxiv.org/abs/1503.02531.

  Args:
    logits: similarity scores corresponding to all pairs in the batch, with
      shape [batch_size].
    labels: labels for each pair.
    weights: weights for each pair.
    temperature: temperature for the distillation.

  Returns:
    binary cross-entropy loss for distillation.
  """
  logits = logits / temperature
  labels = apply_temperature(labels, temperature)
  loss = sigmoid_cross_entropy_with_logits(logits, labels)

  if weights:
    loss = loss * weights
  # Multiply the loss by temperature^2, to keep it on the same scale as the
  # batch softmax loss.
  return (temperature ** 2) * loss


def sparse_labels_for_in_batch_cross_entropy(logits: jnp.array) -> jnp.array:
  """Generates labels assuming the diagnoal |logits| are ground truth."""
  return jnp.arange(logits.shape[0])


def in_batch_cross_entropy(
    logits: jnp.array,
    labels: Optional[jnp.array] = None,
    weights: Optional[jnp.array] = None,
    reduce_fn: Optional[Callable[[jnp.array], jnp.array]] = jnp.mean
    ) -> jnp.array:
  """In batch cross-entropy loss function.

  This corresponds to computing a softmax for each row (and optionally each
  column) of the similarities matrix, where the diagonal element corresponds
  to the only positive pair, and off-diagonal elements are random negatives.

  Args:
    logits: [batch_size, batch_size + sample_size].  Tensor of similarities
      between all pairs of a left and a right element in a batch, with shape
      [batch_size, batch_size + sample_size] where sample_size is the number of
      extra negative right examples, if any.
    labels: [batch_size, batch_size + sample_size].  If None, then this function
      generates one hot labels which assumes the diagnoal elements correspond to
      the ground truth.
    weights: [batch_size].  Weights for each pair (or row).
    reduce_fn: Callable on how to reduce losses to a scalar.  If none, returns
      row-wise loss.

  Returns:
    [batch_size] array of cross entropy losses if reduce_fn is None.  Otherwise,
    return a scalar of reduced row loss.
  """
  if labels is None:
    num_classes = logits.shape[1]
    sparse_labels = sparse_labels_for_in_batch_cross_entropy(logits)
    labels = jax.nn.one_hot(sparse_labels, num_classes, dtype=logits.dtype)

  row_loss, _ = t5x_losses.cross_entropy_with_logits(logits, labels, z_loss=0.0)
  if weights:
    row_loss = row_loss * weights

  return reduce_fn(row_loss) if reduce_fn is not None else row_loss


# ===== Metrics ===== #
@flax.struct.dataclass
class AUC(clu.metrics.CollectingMetric.from_outputs(('labels', 'logits'))):

  def compute(self):
    labels_sum = jnp.sum(self.values['labels'])
    # Do not compute AUC if positives only have one class.
    if labels_sum == 0 or labels_sum == len(self.values['labels']):
      return 0.0
    return sklearn.metrics.roc_auc_score(self.values['labels'],
                                         self.values['logits'])


def compute_auc(targets, predictions, targets_threshold=None):
  """Compute Area Under the ROC and PR curves.

  ROC - Receiver Operating Characteristic
  PR  - Precision and Recall

  Args:
    targets: np.ndarray of targets, either 0 or 1, or continuous values.
    predictions: np.ndarray of predictions, any value.
    targets_threshold: float, if target values are continuous values, this
      threshold binarizes them.

  Returns:
    A dictionary with AUC-ROC and AUC-PR scores.
  """

  if targets_threshold is not None:
    targets = jnp.array(targets)
    targets = jnp.where(targets < targets_threshold,
                        jnp.zeros_like(targets, dtype=jnp.int32),
                        jnp.ones_like(targets, dtype=jnp.int32))

  a = jnp.min(predictions)
  b = jnp.max(predictions)
  scale = 3.0 / (b - a + 1e-6)
  scaled_predictions = scale * (2 * predictions - b - a)
  transformed_predictions = jax.nn.sigmoid(scaled_predictions)
  binarized_targets = jnp.round(targets)
  return {
      'auc-roc':
          AUC.from_model_output(
              logits=transformed_predictions, labels=binarized_targets),
  }
