"""Layer to perform attentive scoring."""

import math

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils


class AttentiveScoringLayer(base_layer.BaseLayer):
  """Calculates attentive scores for enroll & test set representations."""

  @classmethod
  def Params(cls):
    p = super(AttentiveScoringLayer, cls).Params()
    p.name = 'attentive_scoring_layer'
    p.Define(
        'num_keys', 32, 'How many keys to use for attentive scoring. Must '
        'select one or more. (Default: 32)')
    p.Define(
        'key_dim', 16,
        'The number of elements to use for each key vector. Must use 1 or more.'
        ' (Default: 16)')
    p.Define(
        'value_dim', 48,
        'The number of elements to use for each value vector. Must use 1 or '
        'more. (Default: 48)')
    p.Define(
        'use_keys_and_queries', False, 'If set to True, uses separate keys'
        ' and queries in the standard way that attention is applied to '
        'perform the comparison (i.e. keys and queries are estimated '
        'independently). If False, keys are treated as both keys and queries '
        '(i.e. they are the same). (Default: False)')
    p.Define(
        'scale_factor', 16.0, 'The value to multiply the key scores by to '
        'determine how aggressive the softmax function should be. '
        '(Default: 16.0)')
    p.Define(
        'use_trainable_scale_factor', True, 'Whether or not to use a trainable'
        ' scaling factor for the key score softmax calculation. This overrides '
        'the static scale_factor setting if set to True. Additionally, the '
        'scale_factor setting will be used to initialize the equivalent '
        'parameter for training purposes. (Default: True)')
    p.Define(
        'apply_l2_norm_to_keys', True, 'Should L2 length normalization be '
        'applied to the key vectors? (Default: True)')
    p.Define(
        'apply_l2_norm_to_values', False, 'Should L2 length normalization be '
        'applied to the value vectors? (Default: False)')
    p.Define(
        'apply_global_l2_norm_to_concat_form', True, 'Should L2 length '
        'normalization be applied to the (global) concatenated attention vector'
        ' result? This normalization is applied after the application of L2 '
        'normalization to keys, queries and values and after the calculation of'
        ' the attention weights. The attention formulation that is represented '
        'as a weighted combination of enrollment and test value vectors can '
        'also be represented as a single high dimensional dot product. This '
        'normalization applies L2-norm to the high dimensional vectors in the '
        'dot product formulation. (Default: True)')
    p.Define(
        'apply_softmax_per_test_key', False, 'Instead of applying softmax '
        'across all key comparisons, apply softmax for each test utterance '
        'key across all other enrollment keys. (Default: False)')

    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    # Check that the parameters are valid.
    if p.num_keys < 1:
      raise ValueError('The parameter num_keys must be an integer greater than '
                       '0.')
    if p.key_dim < 1:
      raise ValueError('The parameter key_dim must be an integer greater than '
                       '0.')
    if p.value_dim < 1:
      raise ValueError('The parameter value_dim must be an integer greater than'
                       ' 0.')
    if p.scale_factor <= 0:
      raise ValueError('The parameter scale_factor must be a real value greater'
                       ' than 0.')

    if p.use_trainable_scale_factor:
      weight_config = py_utils.WeightParams(
          shape=[],
          init=py_utils.WeightInit.Constant(scale=math.log(p.scale_factor)),
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
      self.CreateVariable('trainable_log_scale_factor', weight_config)

  def FProp(self, inputs, theta=None):
    """Performs attentive scoring given enrollment and test representations.

    Args:
      inputs: A tuple of 2 items. The first is the test representations and the
        second is the enrollment speaker representations.
        Test tensors are tf.float32 tensors of size:
        [num_test_utts, representation_dim].
        Enrollment tensors are tf.float32 tensors of size:
        [num_enroll_spks, num_enroll_utts_per_spk, representation_dim].
        Note here that the representation_dim is the total number of attributes
        to describe a single utterance. For example, for 2 keys, where there is
        a key dimensionality of 4 and a value dimensionality of 10, the
        representation_dim would be 2 * (4 + 10) = 28. The representation_dim
        contains all statistics related to a single utterance.

        Each utterance representation is composed of the concatenated result
        of num_keys sets of statistics. The first key_dim numbers of each
        set contains the key and the next value_dim numbers holds the
        corresponding values. The total dimensionality of each utterance
        representation may be calculated as num_keys * (key_dim + value_dim).

        Note that the dimensionality for both test utterance and
        enrollment utterance representations is the same.
      theta: A NestedMap containing layer weights specific to the
        AttentiveScoringLayer layer. Can be set to None (default) if a trainable
        scaling factor is not needed. Specifically, if
        use_trainable_scale_factor is set, theta must contain the trainable
        variable referenced as theta.trainable_log_scale_factor.
    Returns:
      scores: The output scores comparing all test utterance representations
        with all enrollment speaker models (comprising multiple utterances).
        The result is a tf.float32 tensor of size:
        [num_test_utts, num_enroll_spks(or num_models)]

    """

    p = self.params

    # Retrieve the test and enrollment data
    (test_data, enroll_data) = inputs

    # Determine the input parameters
    representation_dim = enroll_data.shape[2]
    test_representation_dim = test_data.shape[1]

    # Check that the sizes are valid
    if representation_dim != test_representation_dim:
      raise ValueError('The enrollment representation_dim (dim=%d) must be the '
                       'same as the test_representation_dim (dim=%d).' %
                       (representation_dim, test_representation_dim))

    if p.use_keys_and_queries:
      expected_representation_dim = p.num_keys * (2 * p.key_dim + p.value_dim)
      if expected_representation_dim != representation_dim:
        raise ValueError(
            'The expected_representation_dim (dim=%d) must be the same as the '
            'actual representation_dim (dim=%d). The '
            'expected_representation_dim is calculated as num_keys * (2 * '
            'key_dim + value_dim) or %d * (2 * %d + %d).' %
            (expected_representation_dim, representation_dim, p.num_keys,
             p.key_dim, p.value_dim))
    else:
      expected_representation_dim = p.num_keys * (p.key_dim + p.value_dim)
      if expected_representation_dim != representation_dim:
        raise ValueError(
            'The expected_representation_dim (dim=%d) must be the same as the '
            'actual representation_dim (dim=%d). The '
            'expected_representation_dim is calculated as num_keys * (key_dim +'
            ' value_dim) or %d * (%d + %d).' %
            (expected_representation_dim, representation_dim, p.num_keys,
             p.key_dim, p.value_dim))

    # Extract keys, queries and values from the utterance representations and
    # apply L2 normalization on each if enabled. If p.use_keys_and_queries is
    # True, keys and queries will be set the same.
    # Note: all tensors are composed of elements of type tf.float32.
    #
    # For enrollment data:
    # enroll_data_reshaped has size:
    # [num_enroll_spks, num_enroll_utts_per_spk, representation_dim]
    # enroll_keys has size:
    # [num_enroll_spks, num_enroll_utts_per_spk, num_keys, key_dim]
    # enroll_values has size:
    # [num_enroll_spks, num_enroll_utts_per_spk, num_keys, value_dim]
    #
    # For test data:
    # test_data has size: [num_test_utts, representation_dim]
    # test_keys has size: [num_test_utts, num_keys, key_dim]
    # test_values has size: [num_test_utts, num_keys, value_dim]
    (enroll_keys, _, enroll_values) = (self._extract_and_norm(enroll_data))
    (_, test_queries, test_values) = self._extract_and_norm(test_data)

    # Determine whether to have a static or trainable scale factor
    if p.use_trainable_scale_factor:
      # Note that exp(log_scale_factor) resolves to the regular
      # scale_factor parameter. This is to ensure that values are positive.
      if hasattr(theta, 'trainable_log_scale_factor'):
        scale_factor = tf.math.exp(theta.trainable_log_scale_factor)
      else:
        raise ValueError(
            'The trainable weight was not specified as part of the forward '
            'propagation function. A trainable scale factor was requested '
            '(i.e. use_trainable_scale_factor was set to True) without '
            'specifying theta.trainable_log_scale_factor.')
    else:
      scale_factor = p.scale_factor

    # Generate all cross scores for keys and values.
    # For the key scores, scale by scale_factor so that it is ready for the
    # softmax calculation. For both keys and values, the output shape is:
    # [num_test_utts, num_keys(test), num_enroll_spks, num_enroll_utts_per_spk,
    # num_keys(enroll)]
    keys_cross_scores_scaled = scale_factor * tf.tensordot(
        test_queries, enroll_keys, axes=((2), (3)))
    values_cross_scores = tf.tensordot(
        test_values, enroll_values, axes=((2), (3)))

    if p.apply_softmax_per_test_key:
      # Calculate the softmax across scores specific to each test key for a
      # trial. Also scale down by the number of test keys so that the final
      # summation is correct.
      scoring_softmax = tf.keras.activations.softmax(
          keys_cross_scores_scaled, axis=[3, 4]) / p.num_keys
    else:
      # Calculate softmax across all key scores specific to each trial
      scoring_softmax = tf.keras.activations.softmax(
          keys_cross_scores_scaled, axis=[1, 3, 4])

    # Calculate scores of test segments against bundled enrollment segments
    # normalized by the softmax weighting function.
    scores = tf.math.reduce_sum(
        tf.multiply(values_cross_scores, scoring_softmax), axis=(1, 3, 4))

    # Check if we will be applying an additional L2-norm for the effective
    # concatenated attention vector
    if p.apply_global_l2_norm_to_concat_form:
      concat_norm = self._calc_l2_norm_for_concat_form(scoring_softmax,
                                                       test_values,
                                                       enroll_values)
      scores = tf.math.divide(scores, concat_norm)

    return scores

  def _extract_and_norm(self, representation_data):
    """Extracts keys and values and L2 normalizes them if requested.

    Args:
      representation_data: The fixed dimensional representations for each
        utterance. Tensors can be of two different shapes. For enrollment data
        they are tf.float32 (3d) tensors of size:
        [num_enroll_spks, num_enroll_utts_per_spk, representation_dim].
        For test data, they are tf.float32 (2d) tensors of size:
        [num_test_utts, representation_dim].
        For both cases, the parameter representation_dim is equivalent to
        num_keys * (key_dim + value_dim).

    Returns:
      keys: The extracted keys. Tensors can be of two different shapes. For
        enrollment data they are tf.float32 (4d) tensors of size:
        [num_enroll_spks, num_enroll_utts_per_spk, num_keys, key_dim]
        For test data they are tf.float32 (3d) tensors of size:
        [num_test_utts, num_keys, key_dim]
        These may be L2-normalized along the key_dim axis if the
        apply_l2_norm_to_keys attribute is True and unnormalized if False.
      queries: The extracted queries. If p.use_keys_and_queries is set to True,
        then the queries will be independent of the keys. If
        p.use_keys_and_queries is False, then the queries returned will be the
        same as the keys. Tensors can be of two different shapes.
        For enrollment data they are tf.float32 (4d) tensors of size:
        [num_enroll_spks, num_enroll_utts_per_spk, num_keys, key_dim]
        For test data they are tf.float32 (3d) tensors of size:
        [num_test_utts, num_keys, key_dim]
        These may be L2-normalized along the key_dim axis if the
        apply_l2_norm_to_keys attribute is True and unnormalized if False.
      values: The extracted values. Tensors can be of two different shapes. For
        enrollment data they are tf.float32 (4d) tensors of size:
        [num_enroll_spks, num_enroll_utts_per_spk, num_keys, value_dim]
        For test data they are tf.float32 (3d) tensors of size:
        [num_test_utts, num_keys, value_dim]
        These may be L2-normalized along the value_dim axis if the
        apply_l2_norm_to_values attribute is True and unnormalized if False.
    """

    p = self.params

    if p.use_keys_and_queries:
      key_query_value_dim = 2 * p.key_dim + p.value_dim
      query_start_ndx = p.key_dim
      query_stop_ndx = 2 * p.key_dim
    else:
      key_query_value_dim = p.key_dim + p.value_dim
      query_start_ndx = 0
      query_stop_ndx = p.key_dim

    # Extract the key and value tensors specific to whether the data is test
    # data (2 axes) or enrollment data (3 axes).
    data_shape = representation_data.shape.as_list()
    if len(data_shape) == 2:
      # Test data with shape [num_test_utts, representation_dim]
      # converts to keys with shape [num_test_utts, num_keys, key_dim]
      # and values with shape [num_test_utts, num_keys, value_dim]
      num_test_utts = data_shape[0]
      data_reshaped = tf.reshape(
          representation_data,
          shape=[num_test_utts, p.num_keys, key_query_value_dim])
      keys = data_reshaped[:, :, 0:p.key_dim]
      queries = data_reshaped[:, :, query_start_ndx:query_stop_ndx]
      values = data_reshaped[:, :, query_stop_ndx:]
    else:
      # Enrollment data with shape [num_enroll_spks, num_enroll_utts_per_spk,
      # representation_dim] converts to keys with shape [num_enroll_spks,
      # num_enroll_utts_per_spk, num_keys, key_dim] and values with shape
      # [num_enroll_spks, num_enroll_utts_per_spk, num_keys, value_dim]
      (num_enroll_spks, num_enroll_utts_per_spk) = data_shape[0:2]
      data_reshaped = tf.reshape(
          representation_data,
          shape=[
              num_enroll_spks, num_enroll_utts_per_spk, p.num_keys,
              key_query_value_dim
          ])
      keys = data_reshaped[:, :, :, 0:p.key_dim]
      queries = data_reshaped[:, :, :, query_start_ndx:query_stop_ndx]
      values = data_reshaped[:, :, :, query_stop_ndx:]

    # If enabled, perform unit length normalization on keys (or queries)
    if p.apply_l2_norm_to_keys:
      keys = tf.nn.l2_normalize(keys, axis=-1)
      queries = tf.nn.l2_normalize(queries, axis=-1)

    # If enabled, perform unit length normalization on values
    if p.apply_l2_norm_to_values:
      values = tf.nn.l2_normalize(values, axis=-1)

    return (keys, queries, values)

  def _calc_l2_norm_for_concat_form(self, scoring_softmax, test_values,
                                    enroll_values):
    """Calculates the L2-norm for the effective concatenated attention vector.

    Args:
      scoring_softmax: The softmax weightings to apply to the (trial based)
        cross scores. Tensors are of tf.float32 with size: [num_test_utts,
          num_keys(test), num_enroll_spks, num_enroll_utts_per_spk,
          num_keys(enroll)].
      test_values: The test vector related values (tf.float32) of size:
        [num_test_utts, num_keys, value_dim]
      enroll_values: The enrollment vector related values (tf.float32) of size:
        [num_enroll_spks, num_enroll_utts_per_spk, num_keys, value_dim]

    Returns:
      l2_norm_concat: The effective L2-norm to divide the attention scores by.
        The result is tf.float32 of size: [num_test_utts, num_enroll_speakers].
    """

    # Test embeddings
    # Result is 2d: [num_test_utts, num_keys]
    test_values_sumsq = tf.math.reduce_sum(
        tf.math.multiply(test_values, test_values), axis=2)

    # Result is 2d: [num_test_utts, num_enroll_speakers]
    test_values_sumsq_weighted = tf.math.sqrt(
        tf.reduce_sum(
            tf.math.multiply(
                scoring_softmax, test_values_sumsq[:, :, tf.newaxis, tf.newaxis,
                                                   tf.newaxis]),
            axis=(1, 3, 4)))

    # Enroll embeddings
    # Result is 3d: [num_enroll_spks, num_enroll_utts_per_spk, num_keys]
    enroll_values_sumsq = tf.math.reduce_sum(
        tf.math.multiply(enroll_values, enroll_values), axis=3)

    # Result is 2d: [num_test_utts, num_enroll_speakers]
    enroll_values_sumsq_weighted = tf.math.sqrt(
        tf.reduce_sum(
            tf.math.multiply(
                scoring_softmax, enroll_values_sumsq[tf.newaxis,
                                                     tf.newaxis, :, :, :]),
            axis=(1, 3, 4)))

    # Determine the matrix to normalize the cross scores by
    # Result is 2d: [num_test_utts, num_enroll_speakers]
    l2_norm_concat = tf.math.multiply(test_values_sumsq_weighted,
                                      enroll_values_sumsq_weighted)

    return l2_norm_concat
