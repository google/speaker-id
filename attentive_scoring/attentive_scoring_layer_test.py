"""Tests for attentive scoring."""

import attentive_scoring_layer
from lingvo import compat as tf
from lingvo.core import test_utils
import numpy as np

_TF_RANDOM_SEED = 1314
_ABS_TOLERANCE = 1e-05
_REL_TOLERANCE = 1e-04


class AttentiveScoringLayerTest(test_utils.TestCase):
  """The attentive scoring layer tests."""

  def setUp(self):
    """Generates test examples and the attentive scoring configuration."""

    super().setUp()

    tf.reset_default_graph()

    # Generate the example to test the attentive scoring
    (self.test_data, self.enroll_data, self.data_info) = self._GenerateExample()

    # Set up the basic attentive scoring layer config
    p = attentive_scoring_layer.AttentiveScoringLayer.Params()
    p.num_keys = self.data_info['num_keys']
    p.key_dim = self.data_info['key_dim']
    p.value_dim = self.data_info['value_dim']
    p.scale_factor = 1.0
    p.use_trainable_scale_factor = False
    p.apply_l2_norm_to_keys = False
    p.apply_l2_norm_to_values = True
    p.apply_global_l2_norm_to_concat_form = False

    self.params = p

  def _GenerateExample(self):
    """Generates an example to use in the tests.

    This example has 2 enrollment speakers (2 utterances per speaker), and 2
    test utterances. There are 2 keys used. Keys are 2 dimensional and values
    are 3 dimensional.

    Given these constraints, utterance representations are given as:
    Feature indexes | Description
             0 to 1 | First key vector (length 2)
             2 to 4 | First value vector (length 3)
             5 to 6 | Second key vector (length 2)
             7 to 9 | Second value vector (length 3)

    Returns:
      A tuple of (test_data, enroll_data, data_info)
      test_data: Test data related to 2 test utterances with 2 key vectors of 2
        dimensions and 2 value vectors of 3 dimensions. Each utterance
        representation contains a packed form of the key and value vectors. The
        result is a 2d tensor (of tf.float32 type elements) of dimension
        [num_test_utts, representation_dim]. In this example, num_test_utts
        is 2 and representation_dim is 10 (or 2 keys * (2 key_dim +
        3 value_dim)).
      enroll_data: Enrollment data related to 2 speakers each with 2 enrollment
        utterances. Each utterance is composed of 2 key vectors of 2 dimensions
        and 2 value vectors of 3 dimensions. Each utterance is a packed form of
        the key and value vectors. The result is a 3d tensor (of tf.float32
        type elements) of dimension [num_enroll_spks, num_enroll_utts_per_spk,
        representation_dim].
      data_info: A dictionary with keys containing the relevant information for
        the returned data. The dictionary keys include num_keys, key_dim,
        value_dim, num_enroll_spks, num_enroll_utts_per_spk, representation_dim
        and num_test_utts. All values are Python integers.
    """

    # Generate test utterance representations in packed key/value form.
    test_utt1 = np.array([1, 2, -1, 2, 6, 3, -4, 5, 10, -2], dtype=np.float32)
    test_utt2 = np.array([3, -5, -10, 4, 7, 2, 7, 3, -12, -3], dtype=np.float32)

    # Form the test data. They are 2d tensors of tf.float32 and shape:
    # ([num_test_utts, representation_dim])
    test_data = tf.convert_to_tensor(np.stack((test_utt1, test_utt2), axis=0))

    # Generate enrollment utterance representations in packed key/value form.
    # Each utterance is packed in the same way the test utterances are.
    enroll_spk1_utt1 = np.array([1, 2, -1, 2, 6, 3, -4, 5, 10, -2],
                                dtype=np.float32)
    enroll_spk1_utt2 = np.array([3, 5, -2, 3, -6, -1, -5, -5, 9, -7],
                                dtype=np.float32)
    enroll_spk2_utt1 = np.array([5, -2, -3, 7, 5, -6, 8, 1, 3, -7],
                                dtype=np.float32)
    enroll_spk2_utt2 = np.array([4, 5, -10, 8, 7, 3, 2, 7, 2, -5],
                                dtype=np.float32)

    # Form the enrollment data. They are 3d tensors of tf.float32 and shape:
    # [num_enroll_spks, num_enroll_utts_per_spk, representation_dim]
    enroll_data = tf.convert_to_tensor(
        np.stack((np.stack((enroll_spk1_utt1, enroll_spk1_utt2), axis=0),
                  np.stack((enroll_spk2_utt1, enroll_spk2_utt2), axis=0)),
                 axis=0))

    # Specify the data configuration
    data_info = {}
    data_info['num_keys'] = 2
    data_info['key_dim'] = 2
    data_info['value_dim'] = 3
    data_info['num_enroll_spks'] = 2
    data_info['num_enroll_utts_per_spk'] = 2
    data_info['representation_dim'] = 10
    data_info['num_test_utts'] = 2

    return (test_data, enroll_data, data_info)

  def _TestHelper(self, params, test_data, enroll_data):
    """Returns the attentive scores for the given test and enrollment data.

    Args:
      params: Babelfish configuration parameters for setting up the
        attentive_scoring_layer.
      test_data: Test data related to 2 test utterances each with 2 key vectors
        of 2 dimensions and 2 value vectors of 3 dimensions. Each utterance
        representation contains a packed form of the key and value vectors. The
        result is a 2d tensor (of tf.float32 elements) of dimension
        [num_test_utts, representation_dim]. In this example, num_test_utts is 2
        and representation_dim is 10 (or 2 keys * (2 key_dim + 3 value_dim)).
      enroll_data: Enrollment data related to 2 speakers each with 2 enrollment
        utterances. Each utterance is composed of 2 key vectors of 2 dimensions
        and 2 value vectors of 3 dimensions. Each utterance is a packed form of
        the key and value vectors. The result is a 3d tensor (of tf.float32
        elements) of dimension [num_enroll_spks, num_enroll_utts_per_spk,
        representation_dim].

    Returns:
      The output of the attentive scoring. The result is a numpy np.float32
      tensor of shape [num_test_utts, num_enroll_spks].
    """

    with self.session() as sess:
      tf.random.set_seed(_TF_RANDOM_SEED)
      attention_network = params.Instantiate()

      output = attention_network.FProp((test_data, enroll_data))

      sess.run(
          tf.group(tf.global_variables_initializer(), tf.tables_initializer()))

      return sess.run(output)

  def testAttentionNetworkFProp(self):
    """Checks that the forward propagation is correct for attentive modeling."""

    # Run the test
    output = self._TestHelper(self.params, self.test_data, self.enroll_data)

    expected_output = np.array([[0.99984203, 0.43492252],
                                [-0.26937336, -0.30256001]])
    self.assertAllClose(
        expected_output, output, rtol=_REL_TOLERANCE, atol=_ABS_TOLERANCE)

  def testAttentionNetworkFPropWithPerTestKeySoftmax(self):
    """Checks forward propagation result for softmax calculated per test key."""

    # Calculate the softmax for each test utterance key rather than a softmax
    # across all enrollment and test keys for a trial.
    self.params.apply_softmax_per_test_key = True

    # Run the test
    output = self._TestHelper(self.params, self.test_data, self.enroll_data)

    expected_output = np.array([[0.18785089, 0.5673896],
                                [-0.21675828, 0.24606878]])
    self.assertAllClose(
        expected_output, output, rtol=_REL_TOLERANCE, atol=_ABS_TOLERANCE)

  def testAttentionNetworkFPropTrainableScaleFactor(self):
    """Checks that the forward propagation is correct for trainable scaling."""

    # Enable trainable scaling
    self.params.use_trainable_scale_factor = True
    self.params.scale_factor = 2.0

    with self.session() as sess:
      tf.random.set_seed(_TF_RANDOM_SEED)
      attention_network = self.params.Instantiate()

      output = attention_network.FProp((self.test_data, self.enroll_data),
                                       attention_network.theta)
      sess.run(
          tf.group(tf.global_variables_initializer(), tf.tables_initializer()))
      output_result = sess.run(output)
      log_scale_factor_result = sess.run(
          attention_network.theta.trainable_log_scale_factor)

    expected_output = np.array([[1.0, 0.434889], [-0.269374, -0.202443]])
    self.assertAllClose(
        expected_output,
        output_result,
        rtol=_REL_TOLERANCE,
        atol=_ABS_TOLERANCE)

    # Check that the log of the scale_factor=2 is as expected
    self.assertAllClose(
        0.6931471824645996,
        log_scale_factor_result,
        rtol=_REL_TOLERANCE,
        atol=_ABS_TOLERANCE)

  def testAttentionNetworkFPropForKeysAndQueries(self):
    """Checks forward propagation result for key and query calculations."""

    # Calculate the scores using separate keys and queries.
    self.params.use_keys_and_queries = True
    self.params.key_dim = 1

    # Run the test
    output = self._TestHelper(self.params, self.test_data, self.enroll_data)

    expected_output = np.array([[-0.227293, 0.561662], [-0.477102, -0.92821]])
    self.assertAllClose(
        expected_output, output, rtol=_REL_TOLERANCE, atol=_ABS_TOLERANCE)

  def testAttentionNetworkFPropL2NormConcatForm(self):
    """Check forward prop. for L2-norm concatenated representation form."""

    # Calculate the scores given L2-norm on the concatenated representation.
    self.params.apply_l2_norm_to_keys = False
    self.params.apply_l2_norm_to_values = False
    self.params.apply_global_l2_norm_to_concat_form = True

    # Run the test
    output = self._TestHelper(self.params, self.test_data, self.enroll_data)

    expected_output = np.array([[0.9998327494, 0.4348915219],
                                [-0.2693726420, -0.3787475228]])
    self.assertAllClose(
        expected_output, output, rtol=_REL_TOLERANCE, atol=_ABS_TOLERANCE)


if __name__ == '__main__':
  tf.test.main()
