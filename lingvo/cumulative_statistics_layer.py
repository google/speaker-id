"""Layer to perform online frame-based averaging.
"""

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import layers as lingvo_layers
from lingvo.core import py_utils


class CumulativeStatisticsLayer(base_layer.BaseLayer):
  """Online layer to calculate cumulative statistics of the input over time."""

  @classmethod
  def Params(cls):
    p = super(CumulativeStatisticsLayer, cls).Params()
    p.name = 'cumulative_statistics_layer'

    p.Define(
        'stats_type', 'PASS_THRU',
        'If <PASS_THRU> then overrides all other options and copies the input '
        'directly to the output. If <MEAN> then calculates the mean statistics.'
        ' If <MEAN_STD> then calculates the mean and standard deviation '
        'statistics.')
    p.Define(
        'use_weighted_frames', False,
        'If True, use a feedforward network to weight the frames for statistics'
        ' accumulation purposes. It is recommended that a sigmoid output is '
        'used so that counts are handled correctly. If False, a simple average '
        'is used. This is only applicable for non-pass-thru configurations.')
    p.Define(
        'features_name', 'encoded',
        'The key in input_batch pointing to the features. The output features '
        'after FProp will be stored with the same name.')
    p.Define(
        'paddings_name', 'padding',
        'The key in input_batch pointing to the paddings. (Paddings are given '
        'as 0 for speech, 1 for non-speech.) The output paddings after FProp '
        'will be stored with the same name.')
    p.Define(
        'input_dim', None, 'Size of the input features to the network. Must be '
        'set to a value greater than 0. It is used for defining/retrieving the '
        'inference states shape.')
    p.Define(
        'epsilon', 1e-4,
        'This is used for 2 reasons. (1) This small value is added to the count'
        ' for each frame as part of the overall frame count statistics. It is '
        'to ensure no division by zero issues. (2) This same value is added to '
        'each variance element in the standard deviation calculation. It is to '
        'ensure that we do not compute the square root of a negative number.')
    p.Define(
        'frame_weight_ffn', lingvo_layers.FeedForwardNet.Params(),
        'The FeedForwardNet params for the frame-weighting layer. This network '
        'is activated when the use_weighted_frames flag is set to True.')

    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    # Check that the statistics type is valid:
    # We are checking that the statistics type is one of the three types
    # because we do not explicitly check for the 'MEAN' statistics type later
    # in the code.
    if p.stats_type not in ['PASS_THRU', 'MEAN', 'MEAN_STD']:
      raise ValueError('The stats_type must be <PASS_THRU>, '
                       '<MEAN>, or <MEAN_STD>.')

    # Check that an input dimension is specified
    if p.input_dim is None and p.stats_type != 'PASS_THRU':
      raise ValueError('The input dimension (input_dim) must be explicitly set '
                       'and specifies the dimensionality of a frame of input '
                       'features. The value must also be greater than 0 (and '
                       'not None).')

    # Create the feedforward layers for the frame-weighting result
    if p.use_weighted_frames:
      self.CreateChild('frame_weight_ffn', p.frame_weight_ffn)

  def zero_state(self, theta, batch_size):
    """Returns the initial state of the current layer.

    Args:
      theta: A NestedMap containing layer weights (unused for this function).
      batch_size: The number of examples (utterances) in the minibatch

    Returns:
      A NestedMap containing the initial state of the sufficient statistics. If
      the statistics type (stats_type) is set to 'PASS_THRU', then an empty
      NestedMap is returned. If the statistics type is set to 'MEAN', then
      the count and sum of X information is returned. If the statistics type is
      set to 'MEAN_STD', then the sum of X squared statistics are included in
      addition to the count and sum of X information. Depending on configuration
      settings, the NestedMap can contain the following keys within the
      NestedMap accumulated_stats:
      - count: tf.zeros([batch_size]), fprop_dtype (default tf.float32)
      - sum_x: tf.zeros([batch_size, input_dim]), fprop_dtype (default
      tf.float32)
      - sum_xx: tf.zeros([batch_size, input_dim]), fprop_dtype (default
      tf.float32)
    """

    p = self.params

    state0 = py_utils.NestedMap()

    # Return the initial state for PASS_THRU mode
    if p.stats_type == 'PASS_THRU':
      return state0

    # Specify the initial state
    state0.accumulated_stats = py_utils.NestedMap()
    state0.accumulated_stats.count = tf.zeros([batch_size], dtype=p.dtype)
    state0.accumulated_stats.sum_x = tf.zeros([batch_size, p.input_dim],
                                              dtype=p.dtype)

    # Add zero state for sum x-squared statistics if requested
    if p.stats_type == 'MEAN_STD':
      state0.accumulated_stats.sum_xx = tf.zeros([batch_size, p.input_dim],
                                                 dtype=p.dtype)

    return state0

  def IsNullState(self, state):
    """Checks if the input state is either None or empty NestedMap."""
    return all(x is None for x in tf.nest.flatten(state))

  def NullState(self):
    """Returns empty NestedMap as null state."""
    return py_utils.NestedMap()

  def FProp(self, theta, in_nmap, state0=None):
    """Generates frame-weighted mean/std-dev statistics from the inputs.

    Args:
      theta: A NestedMap containing layer weights containing the key
        frame_weight_ffn describing the feed-forward network. This key is needed
        only when use_weighted_frames is set to True and stats_type is not
        'PASS_THRU'.
      in_nmap: A NestedMap. Members include:
        - in_map[p.features_name]: Features tensor of shape [len, batch,
          input_dim], tf.float32.
        - in_map[p.paddings_name]: Paddings tensor of shape [len, batch],
          tf.float32.
      state0: A NestedMap containing sufficient statistics for the previous
        state. When not in inference mode state0 should be NullState. When in
        inference mode, state0, in addition to containing state0 information
        from other child layers, should also include the following keys within
        the NestedMap state0.accumulated_stats:
        - count: [batch_size], tf.float32
        - sum_x: [batch_size, input_dim], tf.float32
        - sum_xx: [batch_size, input_dim], tf.float32 The above keys point to
          the sufficient statistics accumulated across all data packets
          excluding the current data packet of features.

    Returns:
      A NestedMap (out_nmap). For the 'PASS_THRU' case, in_nmap is returned. For
      the 'MEAN' and 'MEAN_STD' cases, a NestedMap with the same information and
        structure as in_nmap with the following additional updates:
        out_nmap[p.features_name]: Features tensor of shape [len, batch,
          output_dim], tf.float32. The output_dim is either input_dim or
          2*input_dim depending on whether standard deviation statistics are
          also included.
        out_nmap[p.paddings_name]: Paddings tensor of shape [len, batch],
          tf.float32.
        out_nmap.state: If state0 is NullState, NullState is returned as the
          output state. If state0 is not NullState, then a NestedMap containing
          sufficient statistics of this cumulative statistics layer gets
          returned. When the mode is 'PASS_THRU', the state is an empty
          NestedMap. In other cases, the state includes the frame-based counts,
          sum of X and sum of X-squared (if 'MEAN_STD' information is requested)
          The additional NestedMap keys are included within
          out_nmap.state.accumulated_stats as follows:
          - count: [batch_size], tf.float32
          - sum_x: [batch_size, input_dim], tf.float32
          - sum_xx: [batch_size, input_dim], tf.float32
          For example, to reference the counts, given an output variable,
          out_nmap, the following would be used:
          out_nmap.state.accumulated_stats.count
    """

    p = self.params

    # Return the input if we are using PASS_THRU mode.
    # Note: the state, even though it is empty, is always populated so as to be
    # consistent with the non-PASS_THRU cases.
    if p.stats_type == 'PASS_THRU':
      # Do not mutate the input NestedMap but the state, because of PASS_THRU.
      out_nmap = in_nmap.copy()
      out_nmap.state = self.NullState()
      return out_nmap

    # Get the input data and padding information
    input_features = in_nmap[p.features_name]
    padding = in_nmap[p.paddings_name]

    # Convert padding to frame based flag indicating if it is speech
    effective_frame_weight = tf.cast(1.0 - padding, dtype=p.dtype)

    # If using frame weighted analysis, calculate the sigmoid output and
    # multiply it with the speech/non-speech effective_frame_weight.
    if p.use_weighted_frames:
      # For each speech frame, a single weight value is generated between 0 and
      # 1 (if a sigmoid activation is used). The expected output (after the
      # squeeze function) is a tensor of shape [len, batch] with a tf.float32
      # data type.
      ffn_frame_weight = tf.squeeze(
          self.frame_weight_ffn.FProp(theta.frame_weight_ffn, input_features,
                                      None),
          axis=2)

      effective_frame_weight = effective_frame_weight * ffn_frame_weight

    # Add a small floor for the effective_frame_weight
    effective_frame_weight = effective_frame_weight + p.epsilon

    # Calculate the cumulative count and cumulative sum_x for the current packet
    # of frames
    cumulative_count = tf.math.cumsum(effective_frame_weight, axis=0)
    cumulative_sum_x = tf.math.cumsum(
        input_features * effective_frame_weight[:, :, tf.newaxis], axis=0)

    # If standard deviation statistics are needed, calculate the cumulative
    # sum_xx (sum x-squared) for the current packet of frames
    if p.stats_type == 'MEAN_STD':
      cumulative_sum_xx = tf.math.cumsum(
          input_features * input_features *
          effective_frame_weight[:, :, tf.newaxis],
          axis=0)

    state1 = self.NullState()
    # If we are running in online mode, be sure to add in the total sums from
    # the past packets of features. If in offline mode, there is no state to
    # update.
    if not self.IsNullState(state0):
      # Calculate cumulative sums up to the current point in time. This includes
      # past packets and the current packet of data.
      cumulative_count = cumulative_count + state0.accumulated_stats.count[
          tf.newaxis, :]
      cumulative_sum_x = cumulative_sum_x + state0.accumulated_stats.sum_x[
          tf.newaxis, :, :]

      # Update the internal state by copying across the statistics of the very
      # last frame
      state1.accumulated_stats = py_utils.NestedMap()
      state1.accumulated_stats.count = cumulative_count[-1, :]
      state1.accumulated_stats.sum_x = cumulative_sum_x[-1, :, :]

      # If standard deviation statistics are needed, calculate the cumulative
      # sum_xx (sum x-squared) for the past and the current packet of frames
      if p.stats_type == 'MEAN_STD':
        cumulative_sum_xx = cumulative_sum_xx + state0.accumulated_stats.sum_xx[
            tf.newaxis, :, :]
        state1.accumulated_stats.sum_xx = cumulative_sum_xx[-1, :, :]

    # Calculate the running mean for the current packet of features
    output_features = tf.math.divide(cumulative_sum_x,
                                     cumulative_count[:, :, tf.newaxis])

    # If requested, calculate and append the standard deviation statistics.
    if p.stats_type == 'MEAN_STD':
      cumulative_mean_xx = tf.math.divide(cumulative_sum_xx,
                                          cumulative_count[:, :, tf.newaxis])
      cumulative_std_dev = tf.math.sqrt(cumulative_mean_xx -
                                        output_features * output_features +
                                        p.epsilon)
      output_features = tf.concat((output_features, cumulative_std_dev), axis=2)

    # Return the output
    out_nmap = py_utils.NestedMap()
    out_nmap[p.features_name] = output_features
    out_nmap[p.paddings_name] = padding
    out_nmap.state = state1
    return out_nmap
