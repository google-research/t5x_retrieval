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

"""Adafactor logical rules for T5X Retrieval."""

from flax.core import freeze
from flax.core import unfreeze
from t5x import adafactor


def logical_factor_rules():
  """Logical factor rules for T5X Retrieval (i.e. two tower models)."""
  rules = unfreeze(adafactor.standard_logical_factor_rules())
  rules.update({
      'affinity': adafactor.FactorDim.COLUMN,
      'idf_buckets': adafactor.FactorDim.NONE,
  })
  return freeze(rules)
