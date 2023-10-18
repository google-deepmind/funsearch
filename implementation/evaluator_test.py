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

import textwrap

from absl.testing import absltest
from absl.testing import parameterized

from funsearch.implementation import evaluator


class EvaluatorTest(parameterized.TestCase):

  def test_trim_function_body_docstring(self):
    code = '''\
  x = 1

  return 0
"""Docstring"""'''
    desired = '''\
  x = 1

  return 0

'''
    actual = evaluator._trim_function_body(code)
    self.assertEqual(desired, actual)

  def test_trim_function_body_function(self):
    code = '''\
  return 0
def new_f():'''
    desired = '''\
  return 0

'''
    actual = evaluator._trim_function_body(code)
    self.assertEqual(desired, actual)

  def test_trim_function_body_empty(self):
    code = '''  return 0\n'''
    desired = '''  return 0\n\n'''
    actual = evaluator._trim_function_body(code)
    self.assertEqual(desired, actual)

  def test_trim_function_indentation_corner_case(self):
    code = textwrap.dedent(
        '''\
          return (1 +
        2)
        def unfinished_code('''
    )
    desired = textwrap.dedent(
        '''\
          return (1 +
        2)

        '''
    )
    actual = evaluator._trim_function_body(code)
    self.assertEqual(desired, actual)

  def test_trim_function_backlash_corner_case(self):
    code = textwrap.dedent(
        '''\
            return score + ((el[0] + 1) * (el[0] + 2) * el[1] / 6 == el[2])\\
         + ((el[0] + 1) * (el[0] + 2) * (el[0] + 3) * el[1] / 24 == el[2])\\
         + ((el[0] + 1) * (el[0] + 2) * el[1] * el[2] / 6 == n)\\
         + ((el[0] + 1) * (el[0] + 2) * el[1] * el[2] / 3 == n + el[0])\\

        '''
    )
    actual = evaluator._trim_function_body(code)
    self.assertEqual(actual, code)


if __name__ == '__main__':
  absltest.main()
