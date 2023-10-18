"""Generating maximal admissible sets of different dimensionalities."""
import itertools
import numpy as np


def block_children(scores, admissible_set, new_element):
  """Modifies `scores` to -inf for elements blocked by `new_element`."""
  n = admissible_set.shape[-1]
  powers = np.array([3 ** i for i in range(n - 1, -1, -1)], dtype=np.int32)

  invalid_vals_raw = {
      (0, 0): (0,),
      (0, 1): (1,),
      (0, 2): (2,),
      (1, 0): (1,),
      (1, 1): (0, 1, 2),
      (1, 2): (1, 2),
      (2, 0): (2,),
      (2, 1): (1, 2),
      (2, 2): (0, 1, 2),
  }
  invalid_vals = [[np.array(invalid_vals_raw[(i, j)], dtype=np.int32)
                   for j in range(3)] for i in range(3)]

  # Block 2^w elements with the same support as `new_element`.
  w = np.count_nonzero(new_element)
  all_12s = np.array(list(itertools.product((1, 2), repeat=w)), dtype=np.int32)
  blocking = np.einsum('aw,w->a', all_12s, powers[new_element != 0])
  scores[blocking] = -np.inf

  # Block elements disallowed by a pair of an extant point and `new_element`.
  for extant_element in admissible_set:
    blocking = np.zeros(shape=(1,), dtype=np.int32)
    for e1, e2, power in zip(extant_element, new_element, powers):
      blocking = (blocking[:, None] + (invalid_vals[e1][e2] * power)[None, :]
                  ).ravel()
    scores[blocking] = -np.inf


def solve(n: int, w: int) -> np.ndarray:
  """Builds a large constant-weight-w n-dimensional admissible set.

  A constant weight admissible set is a subset of {0, 1, 2}^n. The weight of
  each element is the number of non-zero entries. The admissible set consists of
  a maximum of n choose w elements where w is the desired weight. Any three
  distinct elements of the set satisfy the triplet constraints, that is, there
  exists an index k such that the k-th components of the elements are either
  {0, 0, 1}, {0, 0, 2} or {0, 1, 2}.

  Args:
    n: dimension
    w: weight

  Returns:
    An array of shape [admissible_set_size, n], with entries in {0, 1, 2}.
  """
  children = np.array(list(itertools.product((0, 1, 2), repeat=n)),
                      dtype=np.int32)

  scores = -np.inf * np.ones((3 ** n,), dtype=np.float32)
  for child_index, child in enumerate(children):
    if sum(child == 0) == n-w:
      scores[child_index] = priority(np.array(child), n, w)

  max_admissible_set = np.empty((0, n), dtype=np.int32)
  while np.any(scores != -np.inf):
    # Find element with largest score
    max_index = np.argmax(scores)
    child = children[max_index]
    block_children(scores, max_admissible_set, child)
    max_admissible_set = np.concatenate([max_admissible_set, child[None]],
                                        axis=0)
  return max_admissible_set


@funsearch.run
def evaluate(set_description: tuple[int, int]) -> int:
  """Returns the size of the constructed admissible set."""
  n, w = set_description
  admissible_set = solve(n, w)
  return len(admissible_set)


@funsearch.evolve
def priority(el: tuple[int, ...], n: int, w: int) -> float:
  """Trivial scoring function.

  Args:
    el: an element to be potentially added to the current admissible set.
    n: dimensionality of admissible set.
    w: weight of admissible set.

  Returns:
    A number reflecting the priority with which we want to add `el` to the set.
  """
  return 0.0
