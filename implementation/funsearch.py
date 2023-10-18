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

"""A single-threaded implementation of the FunSearch pipeline."""
from typing import Any

from funsearch.implementation import code_manipulation
from funsearch.implementation import config as config_lib
from funsearch.implementation import evaluator
from funsearch.implementation import programs_database
from funsearch.implementation import sampler


def _extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  run_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
  if len(run_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
  evolve_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
  if len(evolve_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
  return evolve_functions[0], run_functions[0]


def main(specification: str, test_inputs: list[Any], config: config_lib.Config):
  """Launches a FunSearch experiment."""
  function_to_evolve, function_to_run = _extract_function_names(specification)

  template = code_manipulation.text_to_program(specification)
  database = programs_database.ProgramsDatabase(
      config.programs_database, template, function_to_evolve)

  evaluators = []
  for _ in range(config.num_evaluators):
    evaluators.append(evaluator.Evaluator(
        database,
        template,
        function_to_evolve,
        function_to_run,
        test_inputs,
    ))
  # We send the initial implementation to be analysed by one of the evaluators.
  initial = template.get_function(function_to_evolve).body
  evaluators[0].analyse(initial, island_id=None, version_generated=None)

  samplers = [sampler.Sampler(database, evaluators, config.samples_per_prompt)
              for _ in range(config.num_samplers)]

  # This loop can be executed in parallel on remote sampler machines. As each
  # sampler enters an infinite loop, without parallelization only the first
  # sampler will do any work.
  for s in samplers:
    s.sample()
