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
"""T5X Retrieval Models.

This module uses Flaxformer modules to build a higher-level model structure and
define methods for the loss computation as well as a train, prediction, and
evaluation steps.
"""
# pylint: disable=attribute-defined-outside-init,g-bare-generic,g-multiple-import
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

from flax import linen as nn
from flax import optim
from flax.core import scope as flax_scope
from flax.training import common_utils
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections
import numpy as np
import seqio
from t5x import losses as t5x_losses
from t5x import metrics as metrics_lib
from t5x import models as t5x_models
from t5x import utils as t5x_utils
from t5x_retrieval import utils
import tensorflow as tf

Array = Union[np.ndarray, jnp.ndarray, jax.pxla.ShardedDeviceArray, tf.Tensor]
DType = jnp.dtype
ConfigDict = ml_collections.ConfigDict
PyTreeDef = type(jax.tree_structure(None))
Optimizer = optim.Optimizer


class DualEncoderBase(t5x_models.BaseTransformerModel):
  """Base class for dual encoder models.

  Subclasses must implement `score_batch` and `_compute_logits`.
  """

  FEATURE_CONVERTER_CLS: Callable[..., seqio.FeatureConverter]

  ALLOWED_INFERENCE_MODE = frozenset({'encode', 'similarity'})

  def __init__(
      self,
      module: nn.Module,
      feature_converter_cls: Callable[[bool], seqio.FeatureConverter],
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      optimizer_def: optim.OptimizerDef,
      inference_mode: str = 'encode',
  ):
    self.FEATURE_CONVERTER_CLS = feature_converter_cls  # pylint: disable=invalid-name
    self._inference_mode = inference_mode
    super(DualEncoderBase, self).__init__(
        module=module,
        input_vocabulary=input_vocabulary,
        output_vocabulary=output_vocabulary,
        optimizer_def=optimizer_def)

  def get_initial_variables(
      self,
      rng: jnp.ndarray,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, DType]] = None
  ) -> flax_scope.FrozenVariableDict:
    """Get the initial variables for an dual-encoder model."""
    input_types = {} if input_types is None else input_types
    encoder_type = input_types.get('left_encoder_input_tokens', jnp.float32)
    left_encoder_shape = input_shapes['left_encoder_input_tokens']
    right_encoder_shape = input_shapes['right_encoder_input_tokens']
    initial_variables = self.module.init(
        rng,
        jnp.ones(left_encoder_shape, encoder_type),
        jnp.ones(right_encoder_shape, encoder_type),
        enable_dropout=False)
    return initial_variables

  def loss_weights(self, batch: Mapping[str,
                                        jnp.ndarray]) -> Optional[jnp.ndarray]:
    raise NotImplementedError('Not implemented for dual encoder.')

  def predict_batch_with_aux(
      self,
      params: Mapping[str, Array],
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.random.KeyArray] = None,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    raise NotImplementedError(
        'Autoregressive prediction is not implemented for dual encoder.')

  def _encode_batch(self, params: Mapping[str, Array],
                    batch: Mapping[str, jnp.ndarray]) -> Array:
    """Encode the embeddings for the inputs."""
    return self.module.apply(
        {'params': params},
        batch['left_encoder_input_tokens'],
        # Disable the dropout during inference.
        enable_dropout=False,
        method=self.module.encode)

  def _similarity_batch(self,
                        params: Mapping[str, Array],
                        batch: Mapping[str, jnp.ndarray],
                        return_intermediates: bool = False) -> Array:
    """Score the similarity of the left and right inputs."""
    _, _, logits = self.module.apply({'params': params},
                                     batch['left_encoder_input_tokens'],
                                     batch['right_encoder_input_tokens'],
                                     enable_dropout=False)
    return logits

  def score_batch(self,
                  params: Mapping[str, Array],
                  batch: Mapping[str, jnp.ndarray],
                  return_intermediates: bool = False) -> jnp.ndarray:
    """Model prediction for batch.

    Args:
      params: Model parameters.
      batch: A batch of inputs.
      return_intermediates: Whether to return intermediates.

    Returns:
      an array of encodings or similarity scores (with optional intermediates).
    """
    if self._inference_mode not in self.ALLOWED_INFERENCE_MODE:
      raise ValueError(
          'Invalid `inference_mode`: %s. Supported inference mode: %s.' %
          (self._inference_mode, self.ALLOWED_INFERENCE_MODE))
    if self._inference_mode == 'encode':
      return self._encode_batch(params, batch)
    elif self._inference_mode == 'similarity':
      return self._similarity_batch(params, batch, return_intermediates)


class DualEncoderModel(DualEncoderBase):
  """Model class for Dual Encoder."""

  ALLOWED_INFERENCE_MODE = frozenset(
      {'encode', 'similarity', 'pointwise_similarity'})

  def __init__(
      self,
      module: nn.Module,
      feature_converter_cls: Callable[[bool], seqio.FeatureConverter],
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      optimizer_def: optim.OptimizerDef,
      inference_mode: str = 'encode',
      use_negatives: bool = False,
      use_align_uniform: bool = False,
      logit_scale: float = 100,
      logit_margin: float = 0.0,
  ):
    """Initialization function.

    Args:
      module: Flax module.
      feature_converter_cls: SeqIO feature converters to apply to the dataset.
      input_vocabulary: Vocabulary for the input features.
      output_vocabulary: Vocabulary for the output features.
      optimizer_def: Optimizer.
      inference_mode: Inference mode (e.g. encode or similarity).
      use_negatives: Whether to use hard negatives.  If True, the model encodes
        the additional feature for hard negatives on the right tower.
      use_align_uniform: Whether to compute alignment and uniformity metrics. If
        True, compute alignment and uniformity metrics.
      logit_scale: A factor for logits scaling.
      logit_margin: A constant for logits margin.
    """
    self._use_negatives = use_negatives
    self._use_align_uniform = use_align_uniform
    self._logit_scale = logit_scale
    self._logit_margin = logit_margin
    super(DualEncoderModel, self).__init__(
        module=module,
        feature_converter_cls=feature_converter_cls,
        input_vocabulary=input_vocabulary,
        output_vocabulary=output_vocabulary,
        optimizer_def=optimizer_def,
        inference_mode=inference_mode)

  def _compute_logits(
      self,
      params: Mapping[str, Any],
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray] = None
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes logits via a forward pass of `self.module_cls`."""
    # Dropout is provided only for the training mode.
    rngs = {'dropout': dropout_rng} if dropout_rng is not None else None

    if not self._use_negatives and 'right_negative_encoder_input_tokens' in batch:
      ValueError(
          'Invalid module. Please select `DualEncoderWithNegativesModel` for negative inputs.'
      )

    if self._use_negatives and 'right_negative_encoder_input_tokens' not in batch:
      ValueError(
          'Invalid inputs. Please prepare negative inputs for DualEncoderWithNegativesModel.'
      )

    if self._use_negatives:
      left_tokens = batch['left_encoder_input_tokens']
      right_positive_tokens = batch['right_encoder_input_tokens']
      right_negative_tokens = batch['right_negative_encoder_input_tokens']

      # left/right_encoder_input_tokens should be 2d tensor.
      assert left_tokens.ndim == 2
      assert right_positive_tokens.ndim == 2
      # right_negative_encoder_input_tokens can be a 2d tensor (when feature
      # spec set up for single negative) or a 3d tensor (when feature spec
      # set up for multiple negatives).
      assert right_negative_tokens.ndim == 2 or right_negative_tokens.ndim == 3

      # All tensors should have the same batch size.
      batch_size = right_positive_tokens.shape[0]
      assert left_tokens.shape[0] == batch_size
      assert right_negative_tokens.shape[0] == batch_size

      if right_negative_tokens.ndim == 3:
        # We have multiple negatives, so need to reshape the
        # right_negative_encoder_input_tokens.

        # Right positive and negative should have the same sequence length.
        right_seq_length = right_positive_tokens.shape[1]
        assert right_seq_length == right_negative_tokens.shape[2]

        num_negatives = right_negative_tokens.shape[1]
        right_negative_tokens = jnp.reshape(
            right_negative_tokens,
            (batch_size * num_negatives, right_seq_length))

      (left_encodings, right_encodings,
       logits), _ = self.module.apply({'params': params},
                                      left_tokens,
                                      right_positive_tokens,
                                      right_negative_tokens,
                                      enable_dropout=rngs is not None,
                                      rngs=rngs,
                                      mutable='dropout')

      # `left_logits` is of shape [B, B*(1+num_negatives)] that considers the
      # negatives while `right_logits` is in shape [B, B] that doesn't considers
      # negatives. `num_negatives` could be greater than 1 in the future.
      left_logits, right_logits = logits, jnp.dot(right_encodings,
                                                  left_encodings.transpose())
    else:
      (left_encodings, right_encodings, logits), _ = self.module.apply(
          {'params': params},
          batch['left_encoder_input_tokens'],
          batch['right_encoder_input_tokens'],
          # TODO(jianmon): switch to call parameter
          enable_dropout=rngs is not None,
          rngs=rngs,
          mutable='dropout')
      # TODO(jianmon): Support uni-directional logits in the future.
      left_logits, right_logits = logits, logits.transpose()

    left_logits *= self._logit_scale
    right_logits *= self._logit_scale

    # Only additive margin to the logits for training mode.
    # For details please check https://arxiv.org/abs/1902.08564. The tensor
    # shapes are not changed after scaling.
    if dropout_rng is not None:
      left_logits = (
          left_logits - self._logit_margin *
          jnp.eye(N=left_logits.shape[0], M=left_logits.shape[1]))
      right_logits = (
          right_logits - self._logit_margin * jnp.eye(right_logits.shape[0]))

    return left_encodings, right_encodings, left_logits, right_logits

  def _compute_loss(
      self,
      batch: Mapping[str, jnp.ndarray],
      left_logits: jnp.ndarray,
      right_logits: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the cross entropy loss given the left and right logits.

    Args:
      batch: a batch of inputs.
      left_logits: array of shape (global_batch_size, global_batch_size) that
        represents the scaled dot product of the left tower.
      right_logits: array of shape (global_batch_size, global_batch_size) that
        represents the scaled dot product of the right tower.

    Returns:
      loss: (bi-directional) batch cross-entropy loss.
      weight_sum: global batch size.
    """
    # targets: [batch, 1] -> [batch, batch]
    left_loss = utils.in_batch_cross_entropy(left_logits)
    right_loss = utils.in_batch_cross_entropy(right_logits)
    loss = jnp.mean(left_loss + right_loss)
    return loss, 0.0, left_logits.shape[0]

  def _compute_metrics(
      self,
      params: Mapping[str, Any],
      batch: Mapping[str, jnp.ndarray],
      left_logits: jnp.ndarray,
      loss: jnp.ndarray,
      total_z_loss: jnp.ndarray,
      weight_sum: jnp.ndarray,
      align_loss: Optional[jnp.ndarray] = None,
      uniform_loss: Optional[jnp.ndarray] = None,
  ) -> metrics_lib.MetricsMap:
    """Compute metrics given the logits, targets and loss."""
    metrics = t5x_models.compute_base_metrics(
        logits=left_logits,
        targets=utils.sparse_labels_for_in_batch_cross_entropy(left_logits),
        mask=None,
        loss=loss,
        z_loss=total_z_loss)
    metrics.update({
        'mrr':
            metrics_lib.AveragePerStep.from_model_output(
                utils.compute_rr(
                    left_logits,
                    utils.sparse_labels_for_in_batch_cross_entropy(
                        (left_logits))))
    })
    if self._use_align_uniform:
      metrics.update({
          'align_loss':
              metrics_lib.AveragePerStep.from_model_output(align_loss),
          'uniform_loss':
              metrics_lib.AveragePerStep.from_model_output(uniform_loss),
      })
    return metrics

  def loss_fn(
      self,
      params: Mapping[str, Any],
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray],
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Loss function used for training with a cross-entropy loss."""

    left_encodings, right_encodings, left_logits, right_logits = self._compute_logits(
        params, batch, dropout_rng)
    # z_loss is already added to loss, which is a workaround for the numerical
    # instability issue.
    loss, z_loss, weight_sum = self._compute_loss(batch, left_logits,
                                                  right_logits)
    if self._use_align_uniform:
      align_loss = utils.compute_align_loss(left_encodings, right_encodings)
      uniform_loss = utils.compute_uniform_loss(
          left_encodings) + utils.compute_uniform_loss(right_encodings)
      metrics = self._compute_metrics(params, batch, left_logits, loss, z_loss,
                                      weight_sum, align_loss, uniform_loss)
    else:
      metrics = self._compute_metrics(
          params,
          batch,
          left_logits,
          loss,
          z_loss,
          weight_sum,
      )

    return loss, metrics

  def score_batch(self,
                  params: Mapping[str, Array],
                  batch: Mapping[str, jnp.ndarray],
                  return_intermediates: bool = False) -> jnp.ndarray:
    """Model prediction for batch.

    Args:
      params: Model parameters.
      batch: A batch of inputs.
      return_intermediates: Whether to return intermediates.

    Returns:
      an array of encodings or similarity scores (with optional intermediates).
    """
    if self._inference_mode not in self.ALLOWED_INFERENCE_MODE:
      raise ValueError(
          'Invalid `inference_mode`: %s. Supported inference mode: %s.' %
          (self._inference_mode, self.ALLOWED_INFERENCE_MODE))
    if self._inference_mode == 'encode':
      return self._encode_batch(params, batch)
    elif self._inference_mode == 'similarity':
      return self._similarity_batch(params, batch, return_intermediates)
    elif self._inference_mode == 'pointwise_similarity':
      logits = self._similarity_batch(params, batch, return_intermediates)
      return jnp.diagonal(logits)


