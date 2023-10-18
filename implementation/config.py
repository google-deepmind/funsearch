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

"""Configuration of a FunSearch experiment."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
  """Configuration of a ProgramsDatabase.

  Attributes:
    functions_per_prompt: Number of previous programs to include in prompts.
    num_islands: Number of islands to maintain as a diversity mechanism.
    reset_period: How often (in seconds) the weakest islands should be reset.
    cluster_sampling_temperature_init: Initial temperature for softmax sampling
        of clusters within an island.
    cluster_sampling_temperature_period: Period of linear decay of the cluster
        sampling temperature.
  """
  functions_per_prompt: int = 2
  num_islands: int = 10
  reset_period: int = 4 * 60 * 60
  cluster_sampling_temperature_init: float = 0.1
  cluster_sampling_temperature_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class Config:
  """Configuration of a FunSearch experiment.

  Attributes:
    programs_database: Configuration of the evolutionary algorithm.
    num_samplers: Number of independent Samplers in the experiment. A value
        larger than 1 only has an effect when the samplers are able to execute
        in parallel, e.g. on different matchines of a distributed system.
    num_evaluators: Number of independent program Evaluators in the experiment.
        A value larger than 1 is only expected to be useful when the Evaluators
        can execute in parallel as part of a distributed system.
    samples_per_prompt: How many independently sampled program continuations to
        obtain for each prompt.
  """
  programs_database: ProgramsDatabaseConfig = dataclasses.field(
      default_factory=ProgramsDatabaseConfig)
  num_samplers: int = 15
  num_evaluators: int = 140
  samples_per_prompt: int = 4
