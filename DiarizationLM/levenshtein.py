"""Function for the Levenshtein algorithm."""
import numpy as np
from enum import Enum


class EditOp(Enum):
  Correct = 0
  Substitution = 1
  Insertion = 2
  Deletion = 3


# Computes the Levenshtein alignment between strings ref and hyp, where the
# tokens in each string are separated by delimiter.
# Outputs a tuple : (edit_distance, alignment) where
# alignment is a list of pairs (ref_pos, hyp_pos) where ref_pos is a position
# in ref and hyp_pos is a position in hyp.
# As an example, for strings 'a b' and 'a c', the output would look like:
# (1, [(0,0), (1,1)]
# Note that insertions are represented as (-1, j) and deletions as (i, -1).
def levenshtein_with_edits(
    ref: str,
    hyp: str,
    delimiter: str = " ",
    print_debug_info: bool = False) -> tuple[int, list[tuple[int, int]]]:
  align = []
  s1 = ref.split(delimiter)
  s2 = hyp.split(delimiter)
  n1 = len(s1)
  n2 = len(s2)
  costs = np.zeros((n1+1, n2+1), dtype=np.int32)
  backptr = np.zeros((n1+1, n2+1), dtype=EditOp)

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
        backptr[i+1][j+1] = EditOp.Insertion
      elif (costs[i+1][j+1] == del_):
        backptr[i+1][j+1] = EditOp.Deletion
      elif (s1[i] == s2[j]):
        backptr[i+1][j+1] = EditOp.Correct
      else:
        backptr[i+1][j+1] = EditOp.Substitution

  if print_debug_info:
    print("Mincost: ", costs[n1][n2])
  i = n1
  j = n2
  # Emits pairs (n1_pos, n2_pos) where n1_pos is a position in n1 and n2_pos
  # is a position in n2.
  while (i > 0 or j > 0):
    if print_debug_info:
      print("i: ", i, " j: ", j)
    ed_op = EditOp.Correct
    if (i >= 0 and j >= 0):
      ed_op = backptr[i][j]
    if (i >= 0 and j < 0):
      ed_op = EditOp.Deletion
    if (i < 0 and j >= 0):
      ed_op = EditOp.Insertion
    if (i < 0 and j < 0):
      raise RuntimeError("Invalid alignment")
    if (ed_op == EditOp.Insertion):
      align.append((-1, j-1))
      j -= 1
    elif (ed_op == EditOp.Deletion):
      align.append((i-1, -1))
      i -= 1
    else:
      align.append((i-1, j-1))
      i -= 1
      j -= 1

  align.reverse()
  return costs[n1][n2], align
