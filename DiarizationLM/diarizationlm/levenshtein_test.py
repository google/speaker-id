"""Test levenshterin."""
import unittest
from diarizationlm import levenshtein


class LevenshteinTest(unittest.TestCase):

  def test_levenshtein_with_edits_1(self):
    s1 = "a b"
    s2 = "a c"
    align = levenshtein.levenshtein_with_edits(s1, s2)
    self.assertEqual(1, align[0])
    self.assertListEqual([(0, 0), (1, 1)], align[1])

  def test_levenshtein_with_edits_2(self):
    s1 = "hello good morning how are you"
    s2 = "hello morning hi how are you"
    align = levenshtein.levenshtein_with_edits(s1, s2)
    self.assertEqual(2, align[0])
    self.assertListEqual(
      [(0, 0), (1, -1), (2, 1), (-1, 2), (3, 3), (4, 4), (5, 5)],
      align[1])


if __name__ == "__main__":
  unittest.main()
