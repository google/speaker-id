"""Tests for utils."""

from lingvo import compat as tf
from lingvo.core import test_utils

import numpy as np

import utils

_RANDOM_SEED = 1321


class SVLUtilsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()

    self.seq_len = 5
    self.feature_dim = 2
    self.num_spks_per_batch = 2
    self.num_utts_per_spk = 2

  def testGetLastSeqOutput(self):
    with self.session() as sess:
      seq = tf.constant(
          list(
              range(self.seq_len * self.num_spks_per_batch *
                    self.num_utts_per_spk * self.feature_dim)))
      seq = tf.reshape(seq, [
          self.num_spks_per_batch * self.num_utts_per_spk, self.seq_len,
          self.feature_dim
      ])
      seq = tf.transpose(seq, [1, 0, 2])
      paddings = []
      lengths = [2, 1, 5, 3]
      for i in lengths:
        assert i <= self.seq_len
        example = tf.zeros([1, i], dtype=tf.float32)
        padded_example = tf.pad(
            example, [[0, 0], [0, self.seq_len - i]], constant_values=1.0)
        print(padded_example)
        paddings.append(padded_example)
      padding = tf.concat(paddings, axis=0)
      padding = tf.transpose(padding, [1, 0])
      last_seq_output = sess.run(utils.GetLastSeqOutput(seq, padding))
      expected_output = np.asarray([[2, 3], [10, 11], [28, 29], [34, 35]])
      np.testing.assert_array_almost_equal(
          last_seq_output, expected_output, decimal=4)

  def testComputeSimilaritySimple(self):
    with self.session() as sess:
      logits = tf.constant([[1.0, 2.0], [1.0, 5.0], [-2.0, -1.0], [-5.0, -1.0]],
                           tf.float32)
      scores = sess.run(
          utils.ComputeSimilaritySimple(logits, self.num_spks_per_batch,
                                        self.num_utts_per_spk))
      # Expect the same speaker utt scores to the centroid being the same,
      # because it is symmetric when there are only 2 utts per speaker.
      expected_scores = np.array([[0.991152, -0.713282], [0.991152, -0.503735],
                                  [-0.713282, 0.991152], [-0.503735, 0.991152]],
                                 np.float32)
      np.testing.assert_array_almost_equal(scores, expected_scores, decimal=4)

  def testComputeSimilaritySplit(self):
    with self.session() as sess:
      logits = tf.constant([[1.0, 2.0], [1.0, 5.0], [-2.0, -1.0], [-5.0, -1.0],
                            [2.0, 2.0], [3.0, 5.0], [-4.0, -1.0], [-6.0, -1.0]],
                           tf.float32)
      self.num_utts_per_spk = 4
      tf.random.set_seed(_RANDOM_SEED)

      scores = sess.run(
          utils.ComputeSimilaritySplit(logits, self.num_spks_per_batch,
                                       self.num_utts_per_spk))

      # Expect positive trial speaker scores to be lower than the scores from
      # ComputeSimilaritySimple, because for each utt, it is not taken for its
      # own speaker centroid computation, AKA there is no cheating.
      expected_scores = np.array(
          [[0.3162278, 0.4876413], [0.55470014, 0.7566749],
           [0.3162278, 0.13371336], [0.55470014, 0.31247067], [0., 0.1865364],
           [0.24253565, 0.4926988], [0.5144958, 0.34551173],
           [0.58123815, 0.34293512]], np.float32)

      np.testing.assert_array_almost_equal(scores, expected_scores, decimal=4)

  def testComputeSimilaritySplitWithVaryingEnroll(self):
    with self.session() as sess:
      logits = tf.constant([[1.0, 2.0], [1.0, 5.0], [-2.0, -1.0], [-5.0, -1.0],
                            [2.0, 2.0], [3.0, 5.0], [-4.0, -1.0], [-6.0, -1.0]],
                           tf.float32)
      self.num_utts_per_spk = 4
      tf.random.set_seed(_RANDOM_SEED)

      # Vary the number of enrollments (last parameter set to True)
      scores = sess.run(
          utils.ComputeSimilaritySplit(logits, self.num_spks_per_batch,
                                       self.num_utts_per_spk, True))

      expected_scores = np.array(
          [[0.3162278, 0.9970546], [-0.6139406, 0.7566749],
           [0.3162278, -0.84366155], [0.9647638, 0.31247067], [0., 0.97014254],
           [-0.84366167, 0.4926988], [0.5144958, -0.7071068],
           [0.95577914, 0.34293512]], np.float32)

      np.testing.assert_array_almost_equal(scores, expected_scores, decimal=4)


if __name__ == '__main__':
  tf.test.main()
