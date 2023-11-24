# FunSearch

This repository accompanies the publication

> Romera-Paredes, B. et al. New discoveries in mathematical sciences via Large Language Models with FunSearch. *Nature* (2023)

There are 6 independent directories:

- `cap_set` contains functions discovered by FunSearch that construct large cap
sets, and we also provide those cap sets in a numerical format for convenience.

- `admissible_set` contains functions discovered by FunSearch that construct
large admissible sets, and we also provide those admissible sets in a numerical
format for convenience.

- `bin_packing` contains heuristics discovered by FunSearch for online 1D bin
packing problems, and an evaluation suite to reproduce the results reported in
the paper.

- `cyclic_graphs` contains functions discovered by FunSearch that construct
large independent sets in strong products of cyclic graphs, and we also provide
those sets in a numerical format for convenience.

- `corner_free_sets` contains the discovered sets of indices, in numerical
format, satisfying the combinatorial degeneration constraints described for the
corners-free problem in the Supplementary Information.

- `implementation` contains an implementation of the evolutionary algorithm,
code manipulation routines, and a single-threaded implementation of the
FunSearch pipeline. It does not contain language models for generating new
programs, the sandbox for executing untrusted code, nor the infrastructure for
running FunSearch on our distributed system. This directory is intended to be
useful for understanding the details of our method, and for adapting it for use
with any available language models, sandboxes, and distributed systems.

## Installation

No installation is required. All notebooks can be opened and run in Google
Colab.

## Usage

- `cap_set`: The notebook `cap_set.ipynb` can be opened via
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/funsearch/blob/master/cap_set/cap_set.ipynb).

- `admissible_set`: The notebook `admissible_set.ipynb` can be opened
via
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/funsearch/blob/master/admissible_set/admissible_set.ipynb).

- `bin_packing`: The notebook `bin_packing.ipynb` can be opened via
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/funsearch/blob/master/bin_packing/bin_packing.ipynb).

- `cyclic_graphs`: The notebook `cyclic_graphs.ipynb` can be opened via
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/funsearch/blob/master/cyclic_graphs/cyclic_graphs.ipynb).

## Citing this work

If you use the code or data in this package, please cite:

```bibtex
@Article{FunSearch2023,
  author  = {Romera-Paredes, Bernardino and Barekatain, Mohammadamin and Novikov, Alexander and Balog, Matej and Kumar, M. Pawan and Dupont, Emilien and Ruiz, Francisco J. R. and Ellenberg, Jordan and Wang, Pengming and Fawzi, Omar and Kohli, Pushmeet and Fawzi, Alhussein},
  journal = {Nature},
  title   = {New discoveries in mathematical sciences via Large Language Models with {F}un{S}earch},
  year    = {2023}
}
```

## License and disclaimer

Copyright 2023 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
