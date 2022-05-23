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

"""Testing helpers for fake device meshes and data chunking."""

import dataclasses
import itertools
import operator
from typing import Sequence, Tuple

import gin
import jax
import numpy as np
import tensorflow.compat.v2 as tf


# Mock JAX devices
@dataclasses.dataclass
class CpuDevice:
  id: int
  process_index: int
  device_kind: str = 'cpu'
  platform: str = 'cpu'


@dataclasses.dataclass
class GpuDevice:
  id: int
  process_index: int
  device_kind: str = 'gpu'
  platform: str = 'Tesla V100-SXM2-16GB'


@dataclasses.dataclass
class TpuDevice:
  id: int
  process_index: int
  coords: Sequence[int]
  core_on_chip: int
  device_kind: str = 'TPU v3'
  platform: str = 'tpu'


# Mock TPU device meshes.
def coords_to_idx(coords: Tuple[int, ...], bounds: Tuple[int, ...]) -> int:
  """Convert grid coordinates to linear index given a dimension ordering.

  Args:
    coords: coordinates in minor to major ordering.
    bounds: coordinate grid bonuds in SAME minor to major ordering as above.

  Returns:
    Linear index for grid point.
  """
  # Calculate stride multipliers.
  strides = tuple(itertools.accumulate((1,) + bounds[:-1], operator.mul))
  # Sum linear index from strides and coords
  return sum(jax.tree_multimap(lambda x, y: x * y, coords, strides))


def make_devices(nx: int,
                 ny: int,
                 nz: int,
                 nc: int = 2,
                 host_layout: Tuple[int, ...] = (2, 2, 1, 2),
                 kind='TPU v3'):
  """Create mock TPU devices."""
  devices = []
  device_bounds = (nx, ny, nz, nc)
  hnx, hny, hnz, hnc = jax.tree_multimap(lambda a, b: a // b, device_bounds,
                                         host_layout)
  for x, y, z, c in itertools.product(*map(range, device_bounds)):
    hx, hy, hz, hc = jax.tree_multimap(lambda a, b: a // b, (x, y, z, c),
                                       host_layout)
    device_id = coords_to_idx((c, x, y, z), (nc, nx, ny, nz))  # pytype: disable=wrong-arg-types
    process_index = coords_to_idx((hc, hx, hy, hz), (hnc, hnx, hny, hnz))  # pytype: disable=wrong-arg-types
    devices.append(
        TpuDevice(
            id=device_id,
            process_index=process_index,
            coords=(x, y, z),
            core_on_chip=c,
            platform='tpu',
            device_kind=kind))
  return devices


def get_test_model(base_model_config,
                   *,
                   emb_dim=4,
                   head_dim=2,
                   num_heads=2,
                   mlp_dim=8,
                   relative_attention_num_buckets=32,
                   relative_attention_max_distance=128,
                   gin_str=None):
  """Creates and returns an instance of `models.BaseModel`."""
  gin.clear_config()
  gin.parse_config_file(
      f't5x_retrieval/configs/models/{base_model_config}')
  gin.parse_config(f"""
      NUM_HEADS = {num_heads}
      HEAD_DIM = {head_dim}
      EMBED_DIM = {emb_dim}
      MLP_DIM = {mlp_dim}
      NUM_LAYERS = 2

      relative_position_biases.RelativePositionBiases:
        num_buckets = {relative_attention_num_buckets}
        max_distance = {relative_attention_max_distance}
  """)

  if gin_str:
    gin.parse_config(gin_str)
  gin.finalize()
  return gin.query_parameter('%MODEL').scoped_configurable_fn()


def get_test_model_for_asymmetric_dual_encoder(model_config, gin_str=None):
  """Creates and returns an instance of Asymmetric Dual Encoder."""
  gin.clear_config()
  gin.parse_config_file(
      f't5x_retrieval/configs/models/{model_config}')
  if gin_str:
    gin.parse_config(gin_str)
  gin.finalize()
  return gin.query_parameter('%MODEL').scoped_configurable_fn()


def get_tiny_de_t5_model(gin_str=None):
  return get_test_model(
      'de_t5_base.gin',
      emb_dim=4,
      head_dim=2,
      num_heads=2,
      mlp_dim=8,
      relative_attention_num_buckets=4,
      relative_attention_max_distance=8,
      gin_str=gin_str)


def get_tiny_de_t5_1_1_model(gin_str=None):
  return get_test_model(
      't5_1_1_base.gin',
      emb_dim=32,
      head_dim=64,
      num_heads=2,
      mlp_dim=64,
      relative_attention_num_buckets=32,
      relative_attention_max_distance=128,
      gin_str=gin_str)


def get_ade_mt5_tiny_tiny(gin_str=None):
  return get_test_model_for_asymmetric_dual_encoder(
      'ade_mt5_tiny_tiny.gin', gin_str=gin_str)


def get_tiny_p1_de_t5_1_1_model(gin_str=None):
  return get_test_model(
      'p1_de_t5_1_1_tiny.gin', gin_str=gin_str)


def get_fake_vocab():
  """Creates fake vocabulary compatible with `get_fake_tokenized_dataset`."""

  @dataclasses.dataclass
  class DummyVocab:
    vocab_size: int = 128
    eos_id: int = 1

  vocab = DummyVocab()
  return (vocab, vocab)


# Text preprocessed and tokenized.
_FAKE_TOKENIZED_DATASET = {
    'train': [
        {
            'inputs': (3, 13, 7, 14, 15, 9, 4, 16),
            'inputs_pretokenized': 'complete: this',
            'targets': (3, 8, 6, 3, 5, 10),
            'targets_pretokenized': 'is a test'
        },
        {
            'inputs': (3, 13, 7, 14, 15, 9, 4, 16),
            'inputs_pretokenized': 'complete: that',
            'targets': (17, 5, 6, 3, 5, 10),
            'targets_pretokenized': 'was a test'
        },
        {
            'inputs': (3, 13, 7, 14, 15, 9, 4, 16),
            'inputs_pretokenized': 'complete: those',
            'targets': (17, 4, 23, 4, 10, 6),
            'targets_pretokenized': 'were tests'
        },
    ],
    # Notice that we repeat consecutively each examples 4 times,
    # this needed for tests like infer_tests to validate determinism.
    'validation': [{
        'inputs': (3, 13, 7, 14, 15, 9, 4, 16),
        'inputs_pretokenized': 'complete: this',
        'targets': (3, 8, 6, 3, 5, 3, 25, 5),
        'targets_pretokenized': 'is a validation',
    }] * 4 + [{
        'inputs': (3, 13, 7, 14, 15, 9, 4, 17),
        'inputs_pretokenized': 'complete: that',
        'targets': (17, 5, 6, 3, 5, 22, 7, 24),
        'targets_pretokenized': 'was another validation',
    }] * 4
}


def get_fake_tokenized_dataset(*_, split='validation', **__):
  """Creates fake dataset compatible with models inputs."""

  if split == 'test':
    split = 'validation'
  output_types = {
      'inputs': tf.int32,
      'targets': tf.int32,
      'inputs_pretokenized': tf.string,
      'targets_pretokenized': tf.string
  }
  output_shapes = {
      'inputs': [None],
      'targets': [None],
      'inputs_pretokenized': [],
      'targets_pretokenized': []
  }
  ds = tf.data.Dataset.from_generator(lambda: _FAKE_TOKENIZED_DATASET[split],
                                      output_types, output_shapes)
  if split == 'train':
    ds = ds.repeat(None)
  return ds


def assert_same(tree_a, tree_b):
  """Asserts that both trees are the same."""
  tree_a, tree_b = jax.device_get((tree_a, tree_b))
  jax.tree_multimap(np.testing.assert_array_equal, tree_a, tree_b)
