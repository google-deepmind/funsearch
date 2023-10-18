# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from absl.testing import absltest
from absl.testing import parameterized

from funsearch.implementation import funsearch

_PY_PROMPT = '''\
import itertools
import jax


@funsearch.run
@jax.jit
def run(n: int):
  return capset(n)


@funsearch.evolve
def capset(n: int):
  """Trivial implementation of capset.

  Args: ...
  """
  return [[1,] * n]
'''

_PY_PROMPT_EVOLVE_RUN = '''\
import itertools


@funsearch.run
@funsearch.evolve
def capset(n: int):
  return [[1,] * n]
'''

_PY_PROMPT_NO_RUN = '''\
import itertools


def run(n: int):
  return capset(n)

@funsearch.evolve
def capset(n: int):
  """Trivial implementation of capset.

  Args: ...
  """
  return [[1,] * n]
'''

_PY_PROMPT_NO_EVOLVE = '''\
import itertools


@funsearch.run
def run(n: int):
  return capset(n)


def capset(n: int):
  """Trivial implementation of capset.

  Args: ...
  """
  return [[1,] * n]
'''

_PY_PROMPT_DOUBLE_RUN = '''\
import itertools

@funsearch.run
def run(n: int):
  return capset(n)

@funsearch.run
def capset(n: int):
  """Trivial implementation of capset.

  Args: ...
  """
  return [[1,] * n]
'''


class FunsearchTest(parameterized.TestCase):

  def test_extract_function_names(self):
    to_evolve, to_run = funsearch._extract_function_names(_PY_PROMPT)
    self.assertEqual(to_run, 'run')
    self.assertEqual(to_evolve, 'capset')

  def test_extract_function_names_evolve_and_run(self):
    to_evolve, to_run = funsearch._extract_function_names(_PY_PROMPT_EVOLVE_RUN)
    self.assertEqual(to_run, 'capset')
    self.assertEqual(to_evolve, 'capset')

  def test_extract_function_names_no_run(self):
    with self.assertRaisesRegex(
        ValueError, r'Expected 1 function decorated with `@funsearch.run`.'):
      funsearch._extract_function_names(_PY_PROMPT_NO_RUN)

  def test_extract_function_names_no_evolve(self):
    with self.assertRaisesRegex(
        ValueError, r'Expected 1 function decorated with `@funsearch.evolve`.'):
      funsearch._extract_function_names(_PY_PROMPT_NO_EVOLVE)

  def test_extract_function_names_double_run(self):
    with self.assertRaisesRegex(
        ValueError, r'Expected 1 function decorated with `@funsearch.run`.'):
      funsearch._extract_function_names(_PY_PROMPT_DOUBLE_RUN)


if __name__ == '__main__':
  absltest.main()
