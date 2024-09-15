"""Tests for loss_layers."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils

import numpy as np
import loss_layers

FLAGS = tf.flags.FLAGS
_RANDOM_SEED = 1314


class GEnd2EndSoftmaxLayerTest(test_utils.TestCase):

  BATCH_SIZE = 6
  NUM_SPKS_PER_BATCH = 3
  NUM_UTTS_PER_SPK = 2

  def setUp(self):
    super().setUp()
    tf.reset_default_graph()

    # test data
    logits = tf.convert_to_tensor(
        np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [1.0, 2.0], [2.0, 2.0],
                  [3.0, 2.0]]),
        dtype=tf.float32)
    labels = tf.convert_to_tensor(np.array([0, 0, 1, 1, 2, 2]), dtype=tf.int32)
    self.predictions = py_utils.NestedMap(logits=logits)
    self.input_batch = py_utils.NestedMap(label=labels)
    # scores for testing the duplicate speaker masking
    self.scores = tf.convert_to_tensor(
        np.array([[1.0, 0.1, 0.8], [2.0, 1.0, 1.8], [0.1, 1.0, 0.2],
                  [0.2, 1.1, 0.1], [2.0, 0.2, 2.0], [3.0, 0.3, 2.0]]),
        dtype=tf.float32)

  def _PrepareLossLayerParam(self):
    p = loss_layers.GEnd2EndSoftmaxLayer.Params()
    p.name = 'test'
    p.random_seed = 123

    p.batch_size = self.BATCH_SIZE
    p.num_spks_per_batch = self.NUM_SPKS_PER_BATCH
    p.num_utts_per_spk = self.NUM_UTTS_PER_SPK
    p.softmax.input_dim = self.NUM_SPKS_PER_BATCH
    p.softmax.num_classes = self.NUM_SPKS_PER_BATCH

    return p

  def testMaskDuplicateSpeakerScoresOnDuplicates(self):
    with self.session() as sess:
      tf.random.set_seed(_RANDOM_SEED)
      p = self._PrepareLossLayerParam()
      p.split_batch = True
      test_layer = p.Instantiate()
      tf.global_variables_initializer().run()
      # Overwrite the speaker labels to have duplicate speakers
      labels = tf.convert_to_tensor(
          np.array([0, 0, 1, 1, 0, 0]), dtype=tf.int32)
      self.input_batch = py_utils.NestedMap(label=labels)
      masked_scores, masked_input_batch = sess.run(
          test_layer.MaskDuplicateSpeakerScores(self.scores, self.input_batch))
      expected_scores = np.array(
          [[1.0, 0.1, -99.2], [2.0, 1.0, -98.2], [0.1, 1.0, 0.2],
           [0.2, 1.1, 0.1], [-98.0, 0.2, 2.0], [-97.0, 0.3, 2.0]], np.float32)
      expected_labels = np.array([0, 0, 1, 1, 2, 2], np.int32)
      self.assertAllEqual(masked_scores, expected_scores)
      self.assertAllEqual(masked_input_batch.label, expected_labels)

  def testMaskDuplicateSpeakerScoresOnUniques(self):
    with self.session() as sess:
      tf.random.set_seed(_RANDOM_SEED)
      p = self._PrepareLossLayerParam()
      p.split_batch = True
      test_layer = p.Instantiate()
      tf.global_variables_initializer().run()
      (masked_scores,
       masked_input_batch), expected_scores, expected_labels = sess.run([
           test_layer.MaskDuplicateSpeakerScores(self.scores, self.input_batch),
           self.scores, self.input_batch.label
       ])
      # Expect no-op
      self.assertAllEqual(masked_scores, expected_scores)
      self.assertAllEqual(masked_input_batch.label, expected_labels)

  def testFPropWithNoSplit(self):
    with self.session() as sess:
      tf.random.set_seed(_RANDOM_SEED)
      p = self._PrepareLossLayerParam()
      p.split_batch = False
      test_layer = p.Instantiate()
      tf.global_variables_initializer().run()
      metrics, _ = sess.run(
          test_layer.FPropDefaultTheta(self.predictions, self.input_batch))

    self.assertAlmostEqual(1.0893, metrics.loss[0], places=4)

  def testFPropWithSplit(self):
    with self.session() as sess:
      tf.random.set_seed(_RANDOM_SEED)
      p = self._PrepareLossLayerParam()
      p.split_batch = True
      test_layer = p.Instantiate()
      tf.global_variables_initializer().run()
      metrics, _ = sess.run(
          test_layer.FPropDefaultTheta(self.predictions, self.input_batch))

    # Expect higher average loss than the no split scenario.
    self.assertAlmostEqual(1.8346, metrics.loss[0], places=4)

  def testFPropWithDuplicateSpeakers(self):
    with self.session() as sess:
      tf.random.set_seed(_RANDOM_SEED)
      p = self._PrepareLossLayerParam()
      p.split_batch = True
      test_layer = p.Instantiate()
      tf.global_variables_initializer().run()
      # Overwrite the speaker labels to have duplicate speakers
      labels = tf.convert_to_tensor(
          np.array([0, 0, 1, 1, 0, 0]), dtype=tf.int32)
      self.input_batch = py_utils.NestedMap(label=labels)
      metrics, _ = sess.run(
          test_layer.FPropDefaultTheta(self.predictions, self.input_batch))

    # Expect the loss weight to be 0, no matter what the loss is.
    self.assertAllEqual(0, metrics.loss[1])

  def testFPropWithDuplicateSpeakersMask(self):
    with self.session() as sess:
      tf.random.set_seed(_RANDOM_SEED)
      p = self._PrepareLossLayerParam()
      p.split_batch = True
      p.mask_dup_spk_scores = True
      test_layer = p.Instantiate()
      tf.global_variables_initializer().run()
      # Overwrite the speaker labels to have duplicate speakers
      labels = tf.convert_to_tensor(
          np.array([0, 0, 1, 1, 0, 0]), dtype=tf.int32)
      self.input_batch = py_utils.NestedMap(label=labels)
      metrics, _ = sess.run(
          test_layer.FPropDefaultTheta(self.predictions, self.input_batch))

    # Expect lower average loss than the no masking scenario.
    self.assertAlmostEqual(1.5610, metrics.loss[0], places=4)

    # Expect the loss weight to be non-zero under masking.
    self.assertAllEqual(6, metrics.loss[1])


class GEnd2EndExtendedSetSoftmaxLayerTest(GEnd2EndSoftmaxLayerTest):

  def _PrepareLossLayerParam(self):
    p = loss_layers.GEnd2EndExtendedSetSoftmaxLayer.Params()
    p.name = 'test'
    p.random_seed = 123

    p.batch_size = self.BATCH_SIZE
    p.num_spks_per_batch = self.NUM_SPKS_PER_BATCH
    p.num_utts_per_spk = self.NUM_UTTS_PER_SPK

    num_proxy_classes = self.NUM_SPKS_PER_BATCH * (self.NUM_SPKS_PER_BATCH -
                                                   1) + 1
    p.softmax.input_dim = num_proxy_classes
    p.softmax.num_classes = num_proxy_classes

    return p

  def testFPropWithNoSplit(self):
    with self.session() as sess:
      tf.random.set_seed(_RANDOM_SEED)
      p = self._PrepareLossLayerParam()
      p.split_batch = False
      test_layer = p.Instantiate()
      tf.global_variables_initializer().run()
      metrics, _ = sess.run(
          test_layer.FPropDefaultTheta(self.predictions, self.input_batch))

    self.assertAlmostEqual(1.9803, metrics.loss[0], places=4)

  def testFPropWithAttentionScoring(self):
    with self.session() as sess:
      tf.random.set_seed(_RANDOM_SEED)
      p = self._PrepareLossLayerParam()
      p.split_batch = True
      p.select_embedding_comparison_type = (
          loss_layers.EmbeddingComparisonType.ATTENTIVE_SCORING)
      p.attentive_scoring.num_keys = 1
      p.attentive_scoring.key_dim = 1
      p.attentive_scoring.value_dim = 1
      print(self.input_batch)

      test_layer = p.Instantiate()
      tf.global_variables_initializer().run()
      metrics, _ = sess.run(
          test_layer.FPropDefaultTheta(self.predictions, self.input_batch))

    self.assertAlmostEqual(1.94591, metrics.loss[0], places=4)

  def testFPropWithSplit(self):
    with self.session() as sess:
      tf.random.set_seed(_RANDOM_SEED)
      p = self._PrepareLossLayerParam()
      p.split_batch = True
      test_layer = p.Instantiate()
      tf.global_variables_initializer().run()
      metrics, _ = sess.run(
          test_layer.FPropDefaultTheta(self.predictions, self.input_batch))

    # Expect higher average loss than the no split scenario.
    self.assertAlmostEqual(2.7612, metrics.loss[0], places=4)

  def testFPropWithDuplicateSpeakersMask(self):
    with self.session() as sess:
      tf.random.set_seed(_RANDOM_SEED)
      p = self._PrepareLossLayerParam()
      p.split_batch = True
      p.mask_dup_spk_scores = True
      test_layer = p.Instantiate()
      tf.global_variables_initializer().run()
      # Overwrite the speaker labels to have duplicate speakers
      labels = tf.convert_to_tensor(
          np.array([0, 0, 1, 1, 0, 0]), dtype=tf.int32)
      self.input_batch = py_utils.NestedMap(label=labels)
      metrics, _ = sess.run(
          test_layer.FPropDefaultTheta(self.predictions, self.input_batch))

    # Expect lower average loss than the no masking scenario.
    self.assertAlmostEqual(2.4000, metrics.loss[0], places=4)

    # Expect the loss weight to be non-zero under masking.
    self.assertAllEqual(6, metrics.loss[1])


if __name__ == '__main__':
  tf.test.main()
