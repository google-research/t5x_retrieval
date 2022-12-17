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
from t5x_retrieval import losses
from t5x_retrieval import utils
import tensorflow as tf

Array = Union[np.ndarray, jnp.ndarray, jax.pxla.ShardedDeviceArray, tf.Tensor]
DType = jnp.dtype
ConfigDict = ml_collections.ConfigDict
PyTreeDef = type(jax.tree_util.tree_structure(None))
Optimizer = optim.Optimizer


LEFT_ENCODINGS = 'left_encodings'
RIGHT_ENCODINGS = 'right_encodings'


class DualEncoderBase(t5x_models.BaseTransformerModel):
  """Base class for dual encoder models.

  Subclasses must implement `score_batch` and `_compute_logits`.
  """

  FEATURE_CONVERTER_CLS: Callable[..., seqio.FeatureConverter]

  ALLOWED_INFERENCE_MODE = frozenset({'encode', 'similarity'})

  # TODO(b/262639556): Change loss_module from Optional to required once
  # loss-layers have been implemented
  def __init__(
      self,
      module: nn.Module,
      feature_converter_cls: Callable[[bool], seqio.FeatureConverter],
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      optimizer_def: optim.OptimizerDef,
      inference_mode: str = 'encode',
      loss_module_factory: Optional[nn.Module] = None,
  ):
    self.FEATURE_CONVERTER_CLS = feature_converter_cls  # pylint: disable=invalid-name
    self._inference_mode = inference_mode

    self.loss_module = None
    # TODO(b/262639556): Remove check once loss-layer is not Optional
    if loss_module_factory:
      self.loss_module = loss_module_factory()
      self.loss_module.validate_model_features(feature_converter_cls(False))

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

    # TODO(b/262639556): Remove check once loss-layer is not Optional
    if self.loss_module:
      loss_variables = self.loss_module.get_initial_variables(
          rng, input_shapes, input_types)
      initial_variables = initial_variables.copy(loss_variables)

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
    left_encodings, right_encodings, logits = self.module.apply(
        {'params': params},
        batch['left_encoder_input_tokens'],
        batch['right_encoder_input_tokens'],
        enable_dropout=False)
    if return_intermediates:
      return logits, {
          LEFT_ENCODINGS: (left_encodings,),
          RIGHT_ENCODINGS: (right_encodings,),
      }
    else:
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

  ALLOWED_INFERENCE_MODE = frozenset({
      'encode', 'encode_left', 'encode_right', 'similarity',
      'pointwise_similarity'
  })

  def __init__(
      self,
      module: nn.Module,
      feature_converter_cls: Callable[[bool], seqio.FeatureConverter],
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      optimizer_def: optim.OptimizerDef,
      loss_module_factory: Optional[losses.DualEncoderLoss] = None,
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
      loss_module_factory: Factory to produce loss module.
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
        inference_mode=inference_mode,
        loss_module_factory=loss_module_factory)

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
          enable_dropout=rngs is not None,
          rngs=rngs,
          mutable='dropout')
      left_logits, right_logits = logits, logits.transpose()

    left_logits *= self._logit_scale
    right_logits *= self._logit_scale

    # Only additive margin to the logits for training mode.
    # For details please check https://arxiv.org/abs/1902.08564. The tensor
    # shapes are not changed after scaling.
    if dropout_rng is not None and self._logit_margin != 0:
      left_logits = (
          left_logits - self._logit_margin *
          jnp.eye(N=left_logits.shape[0], M=left_logits.shape[1]))
      right_logits = (
          right_logits - self._logit_margin * jnp.eye(right_logits.shape[0]))

    return left_encodings, right_encodings, left_logits, right_logits

  def loss_fn(
      self,
      params: Mapping[str, Any],
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray],
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Loss function used for training with a cross-entropy loss."""

    left_encodings, right_encodings, left_logits, right_logits = self._compute_logits(
        params, batch, dropout_rng)
    loss, metrics = self.loss_module.apply({'params': params},
                                           batch=batch,
                                           logits=left_logits,
                                           right_logits=right_logits)
    if self._use_align_uniform:
      align_loss = utils.compute_align_loss(left_encodings, right_encodings)
      uniform_loss = utils.compute_uniform_loss(
          left_encodings) + utils.compute_uniform_loss(right_encodings)
      metrics.update({
          'align_loss':
              metrics_lib.AveragePerStep.from_model_output(align_loss),
          'uniform_loss':
              metrics_lib.AveragePerStep.from_model_output(uniform_loss),
      })

    return loss, metrics

  def _encode_batch(self, params: Mapping[str, Array],
                    batch: Mapping[str, jnp.ndarray]) -> Array:
    """Encode the embeddings for the inputs."""
    if self._inference_mode == 'encode_right':
      encoder_input_tokens = batch['right_encoder_input_tokens']
    else:
      encoder_input_tokens = batch['left_encoder_input_tokens']
    return self.module.apply(
        {'params': params},
        encoder_input_tokens,
        # Disable the dropout during inference.
        enable_dropout=False,
        method=self.module.encode)

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
    if self._inference_mode.startswith('encode'):
      return self._encode_batch(params, batch)
    elif self._inference_mode == 'similarity':
      return self._similarity_batch(params, batch, return_intermediates)
    elif self._inference_mode == 'pointwise_similarity':
      if return_intermediates:
        logits, intermediates = (
            self._similarity_batch(params, batch, return_intermediates))
        return jnp.diagonal(logits), intermediates
      else:
        logits = self._similarity_batch(params, batch, return_intermediates)
        return jnp.diagonal(logits)


