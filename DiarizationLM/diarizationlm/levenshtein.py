"""Function for the Levenshtein algorithm.

Note: This Python implementation is very inefficient. Please use this C++
implementation instead: https://github.com/wq2012/word_levenshtein
"""
import numba
import numpy as np


# Edit operations.
_CORRECT = 0
_SUBSTITUTION = 1
_INSERTION = 2
_DELETION = 3


# Computes the Levenshtein alignment between strings ref and hyp, where the
# tokens in each string are separated by delimiter.
# Outputs a tuple : (edit_distance, alignment) where
# alignment is a list of pairs (ref_pos, hyp_pos) where ref_pos is a position
# in ref and hyp_pos is a position in hyp.
# As an example, for strings 'a b' and 'a c', the output would look like:
# (1, [(0,0), (1,1)]
# Note that insertions are represented as (-1, j) and deletions as (i, -1).
@numba.njit
def levenshtein_with_edits(
    ref: str,
    hyp: str,
    print_debug_info: bool = False) -> tuple[int, list[tuple[int, int]]]:
  align = []
  s1 = ref.split()
  s2 = hyp.split()
  n1 = len(s1)
  n2 = len(s2)
  costs = np.zeros((n1+1, n2+1), dtype=np.int32)
  backptr = np.zeros((n1+1, n2+1), dtype=np.int32)

  for i in range(n1+1):  # ref
    costs[i][0] = i  # deletions

  for j in range(n2):  # hyp
    costs[0][j+1] = j+1  # insertions
    for i in range(n1):  # ref
      # (i,j) <- (i,j-1)
      ins = costs[i+1][j] + 1
      # (i,j) <- (i-1,j)
      del_ = costs[i][j+1] + 1
      # (i,j) <- (i-1,j-1)
      sub = costs[i][j] + (s1[i] != s2[j])
      costs[i + 1][j + 1] = min(ins, del_, sub)
      if (costs[i+1][j+1] == ins):
        backptr[i+1][j+1] = _INSERTION
      elif (costs[i+1][j+1] == del_):
        backptr[i+1][j+1] = _DELETION
      elif (s1[i] == s2[j]):
        backptr[i+1][j+1] = _CORRECT
      else:
        backptr[i+1][j+1] = _SUBSTITUTION

  if print_debug_info:
    print("Mincost: ", costs[n1][n2])
  i = n1
  j = n2
  # Emits pairs (n1_pos, n2_pos) where n1_pos is a position in n1 and n2_pos
  # is a position in n2.
  while (i > 0 or j > 0):
    if print_debug_info:
      print("i: ", i, " j: ", j)
    ed_op = _CORRECT
    if (i >= 0 and j >= 0):
      ed_op = backptr[i][j]
    if (i >= 0 and j < 0):
      ed_op = _DELETION
    if (i < 0 and j >= 0):
      ed_op = _INSERTION
    if (i < 0 and j < 0):
      raise RuntimeError("Invalid alignment")
    if (ed_op == _INSERTION):
      align.append((-1, j-1))
      j -= 1
    elif (ed_op == _DELETION):
      align.append((i-1, -1))
      i -= 1
    else:
      align.append((i-1, j-1))
      i -= 1
      j -= 1

  align.reverse()
  return costs[n1][n2], align
