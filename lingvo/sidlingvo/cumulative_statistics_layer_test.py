"""Tests for cumulative_statistics_layer."""

from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
import numpy as np

import cumulative_statistics_layer

_TF_RANDOM_SEED = 1314


class CumulativeStatisticsLayerTest(test_utils.TestCase,
                                    parameterized.TestCase):
  """The cumulative statistics layer tests."""

  def setUp(self):
    """Generates test minibatches, the expected output and the layer config."""

    super().setUp()

    tf.reset_default_graph()

    # Generate the common list of minibatches
    self.input_batch_list = self._GenerateListOfMinibatches()

    # Get the expected test results to compare with
    self.expected_outputs = self._GetExpectedTestOutputs()

    # Set up the basic cumulative statistics layer config
    input_dim = self.input_batch_list[0].features.shape[2]
    p = cumulative_statistics_layer.CumulativeStatisticsLayer.Params()
    p.stats_type = 'PASS_THRU'
    p.use_weighted_frames = False
    p.input_dim = input_dim
    p.features_name = 'features'
    p.paddings_name = 'paddings'
    p.frame_weight_ffn.activation = ['SIGMOID']
    p.frame_weight_ffn.has_bias = [True]
    p.frame_weight_ffn.hidden_layer_dims = [1]
    p.frame_weight_ffn.input_dim = input_dim
    self.params = p

  def _GenerateListOfMinibatches(self):
    """Generates a list of 2 minibatches to use in the tests.

    Returns:
      A list of 2 padded batches of examples.
      The structure is a list of the following:
      {
        'features': tf.tensor(float32) of shape(len, batch, dim)
        'paddings': tf.tensor(float32) of shape(len, batch)
      }
      where len=8, batch=2 and dim=3
    """

    # Specify the minibatch size
    max_seq_length = 8
    batch_size = 2
    feature_dims = 3

    # Specify the features and paddings for the first minibatch
    input_tensor1 = tf.reshape(
        tf.range(max_seq_length * batch_size * feature_dims, dtype=tf.float32),
        [max_seq_length, batch_size, feature_dims])
    paddings1 = tf.zeros(shape=[max_seq_length, batch_size], dtype=tf.float32)
    input_batch1 = py_utils.NestedMap(
        features=input_tensor1, paddings=paddings1)

    # Specify the features and paddings for the second minibatch
    # Also modify paddings2 to have a 1 at position [2, 0] in the tensor
    input_tensor2 = input_tensor1 + 1.0
    paddings2 = tf.zeros(shape=[max_seq_length, batch_size], dtype=tf.float32)
    paddings2 = tf.tensor_scatter_nd_update(paddings2, [[2, 0]], [1])
    input_batch2 = py_utils.NestedMap(
        features=input_tensor2, paddings=paddings2)

    # Create the list of minibatches
    input_batch_list = [input_batch1, input_batch2]

    return input_batch_list

  def _GetExpectedTestOutputs(self):
    """Returns the expected outputs for the tests.

    Returns:
      A 2D dictionary containing numpy (float32) arrays of the expected test
      outputs for PASS_THRU, MEAN and MEAN_STD while setting the
      use_weighted_frames setting to either True or False.
    """

    expected_outputs = {}
    expected_outputs['PASS_THRU'] = {
        False: np.array([[43.0, 44.0, 45.0], [46.0, 47.0, 48.0]])
    }
    expected_outputs['MEAN'] = {
        False: np.array(
          [[22.066668, 23.066668, 24.066668], [24.5, 25.5, 26.5]]),
        True: np.array(
          [[25.424778, 26.424774, 27.424776], [26.75111, 27.751108, 28.75111]])
    }
    expected_outputs['MEAN_STD'] = {
        False: np.array(
          [[22.066668, 23.066668, 24.066668, 14.026007, 14.026006, 14.026007],
           [24.5, 25.5, 26.5, 13.756816, 13.756816, 13.756816]]),
        True: np.array(
          [[25.424778, 26.424774, 27.424776, 12.595981, 12.595986, 12.595986],
           [26.75111, 27.751108, 28.75111, 12.927436, 12.927438, 12.927436]])
    }

    return expected_outputs

  def _TestHelperWithState(self, params, list_of_batches):
    """Returns the expected outputs for the tests.

    Args:
      params: Babelfish configuration parameters for setting up the
        cumulative_statistics_layer.
      list_of_batches: A list of padded batches of examples.
        The structure is a list of the following: {
        'features': tf.tensor(float32) of shape(len, batch, dim)
        'paddings': tf.tensor(float32) of shape(len, batch) }

    Returns:
      A dictionary containing numpy arrays of the expected test outputs.
      The structure is as follows:
      {
        'features': np.array(float32) of shape(len, batch, dim)
        'paddings': np.array(float32) of shape(len, batch)
      }
    """

    with self.session() as sess:
      tf.random.set_seed(_TF_RANDOM_SEED)
      network = params.Instantiate()

      batch_size = list_of_batches[0].features.shape[1]
      state = network.zero_state(network.theta, batch_size)

      for batch_t in list_of_batches:
        output = network.FProp(network.theta, batch_t, state)
        # Pass the output state over to the next batch as input state.
        state = output.state

      sess.run(
          tf.group(tf.global_variables_initializer(), tf.tables_initializer()))

      return sess.run(output)

  def _TestExpectedOutputs(self, stats_type, use_weighted_frames,
                           expected_last_frame_output):
    """Ochestrates the tests and checks that the expected outputs are correct.

    Args:
      stats_type: Specifies how the statistics will be accumulated. Options are
        'PASS_THRU', 'MEAN', and 'MEAN_STD'.
      use_weighted_frames: Whether or not to use frame-based weighting using a
        linear transform with sigmoid to determine the weight. Type is Boolean,
        True/False
      expected_last_frame_output: The last frame of output features once
        passed through the cumulative statistics layer. It is of the form:
          np.array(float32) with shape(batch, dim)
    """

    # Change basic params config to do the mean based statistics
    self.params.stats_type = stats_type
    self.params.use_weighted_frames = use_weighted_frames

    # Run the test
    output = self._TestHelperWithState(self.params, self.input_batch_list)

    # Check that for the 2nd minibatch that the input paddings are the same as
    # the output paddings.
    self.assertAllEqual(self.input_batch_list[1].paddings, output.paddings)

    # Check the size of the returned features. For the mean and standard
    # deviation case, the output should be double the dimensions.
    (max_seq_length, batch_size,
     feature_dims) = self.input_batch_list[0].features.shape.as_list()
    if stats_type == 'MEAN_STD':
      self.assertAllEqual((max_seq_length, batch_size, 2 * feature_dims),
                          output.features.shape)
    else:
      self.assertAllEqual((max_seq_length, batch_size, feature_dims),
                          output.features.shape)

    # Check that the last accumulated frame is correct
    last_frame = output.features[-1, :, :]
    self.assertAllClose(
        last_frame, expected_last_frame_output, rtol=1e-03, atol=1e-03)

  @parameterized.parameters(
      {
          'stats_type': 'PASS_THRU',
          'use_weighted_frames': False
      },
      {
          'stats_type': 'MEAN',
          'use_weighted_frames': False
      },
      {
          'stats_type': 'MEAN',
          'use_weighted_frames': True
      },
      {
          'stats_type': 'MEAN_STD',
          'use_weighted_frames': False
      },
      {
          'stats_type': 'MEAN_STD',
          'use_weighted_frames': True
      },
  )
  def testCumulativeStatisticsFProp(self, stats_type, use_weighted_frames):
    """Runs a test for a particular configuration.

    Args:
      stats_type: Specifies how the statistics will be accumulated. Options are
        'PASS_THRU', 'MEAN', and 'MEAN_STD'.
      use_weighted_frames: Whether or not to use frame-based weighting using a
        linear transform with sigmoid to determine the weight. Type is Boolean,
        True/False
    """

    # Run the particular test
    expected_last_frame_output = self.expected_outputs[stats_type][
        use_weighted_frames]
    self._TestExpectedOutputs(stats_type, use_weighted_frames,
                              expected_last_frame_output)


if __name__ == '__main__':
  tf.test.main()
