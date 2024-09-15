"""Speaker recognition loss layers."""

import enum
from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import layers as lingvo_layers
from lingvo.core import py_utils
import attentive_scoring_layer
import utils


class EmbeddingComparisonType(enum.Enum):
  """Add enum capabilities for selecting cosine or attentive scoring."""

  # Instructs the system to perform cosine scoring of the test utterance
  # embedding against the averaged speaker enrollment utterance embedding.
  COSINE = enum.auto()

  # Performs attentive scoring of the test utterance against all enrollment
  # utterances. Attentive scoring involves the use of keys/queries and values to
  # compare a test utterance with multiple enrollment utterances. For more
  # information, refer to this paper:
  # https://arxiv.org/abs/2203.05642
  ATTENTIVE_SCORING = enum.auto()


class GEnd2EndSoftmaxLayer(base_layer.BaseLayer):
  """The generalized end-to-end softmax loss layer.

  This is implemented based on Eq.(6) proposed in the GEnd2End paper:
  https://arxiv.org/pdf/1710.10467.pdf
  """

  @classmethod
  def Params(cls):
    p = super(GEnd2EndSoftmaxLayer, cls).Params()
    # Below values should match the input generator params!
    p.Define('batch_size', 128, 'The batch size.')
    p.Define('num_spks_per_batch', 16, 'The number of speakers in the batch.')
    p.Define('num_utts_per_spk', 8, 'The number of utts per speaker.')
    p.Define('vary_number_of_enrollment_utterances_averaged', False,
             ('Vary the number of enrollment utterances in training (True) or '
              'use all the available enrollment utterances (default: False).'))
    p.Define('softmax', lingvo_layers.SimpleFullSoftmax.Params(),
             'Softmax layer.')
    p.Define(
        'split_batch', True, 'Whether to split batch into enrollment and '
        'test at loss computation.')
    p.Define(
        'select_embedding_comparison_type', EmbeddingComparisonType.COSINE,
        'Determines what method should be used to compare the enrollment '
        'and test utterance embeddings. There are currently 2 options: '
        'EmbeddingComparisonType.COSINE and '
        'EmbeddingComparisonType.ATTENTIVE_SCORING. (Default: '
        'EmbeddingComparisonType.COSINE)')
    p.Define('attentive_scoring',
             attentive_scoring_layer.AttentiveScoringLayer.Params(),
             'Attentive scoring layer.')
    p.Define('transform_weight', 10.0, 'The transform weight initial value.')
    p.Define('transform_bias', -5.0, 'The transform bias initial value.')
    p.Define('mask_dup_spk_scores', False,
             'Whether to mask duplicate speaker scores in the batch.')
    p.Define(
        'mask_value', 100.0,
        'The masking value to be subtracted from the scores matrix.'
        'Note that this is done after the linear transformation of the'
        'similarity scores.')
    p.name = 'gend2end_softmax_layer'
    return p

  def __init__(self, params):
    if params.num_utts_per_spk % 2 != 0:
      raise ValueError('An even number of examples per speaker is expected, '
                       'but got %i.)' % (params.num_utts_per_spk))

    if params.batch_size != params.num_spks_per_batch * params.num_utts_per_spk:
      raise ValueError(
          'The batch size must be the same as the number of '
          'speakers times the number of examples per speaker. But '
          'batch_size != num_spks_per_batch * num_utts_per_spk, %i != %i * %i)'
          % (params.batch_size, params.num_spks_per_batch,
             params.num_utts_per_spk))

    super(GEnd2EndSoftmaxLayer, self).__init__(params)
    p = self.params
    self.CreateChild('softmax', p.softmax)
    self.CreateChild('attentive_scoring', p.attentive_scoring)

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    pc = py_utils.WeightParams(
        shape=[],
        init=py_utils.WeightInit.Constant(scale=p.transform_weight),
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('w', pc)
    pc = py_utils.WeightParams(
        shape=[],
        init=py_utils.WeightInit.Constant(scale=p.transform_bias),
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('b', pc)

  def BatchWithoutDuplicateSpeakers(self, input_batch):
    """Check if the batch has duplicate speakers.

    This tensor is multiplied on top of the default loss weight of batch size.
    If the input batch contains duplicate speakers, we return 0, so that the
    batch loss gets weighted 0. Otherwise, the batch is valid and we return 1.

    Args:
      input_batch: A NestedMap containing:
        - src.src_inputs: tensor of shape [batch, len, dim].
        - src.paddings: tensor of shape [batch, len].
        - label: int tensor of shape [batch]. The inputs has ground truth labels
          required for the loss calculation.

    Returns:
      A int32 tensor of value 0 for invalid batches with duplicate speakers,
      and 1 for valid ones without duplicate speakers.
    """
    p = self.params

    # Collect all the speakers
    speakers = tf.gather(
        tf.reshape(input_batch.label,
                   [p.num_spks_per_batch, p.num_utts_per_spk]), [0],
        axis=1)
    speakers = tf.squeeze(speakers)
    # Check the batch speaker compositions
    return tf.cast(
        tf.reduce_all(tf.equal(speakers, tf.range(p.num_spks_per_batch))),
        dtype=tf.int32)

  def MaskDuplicateSpeakerScores(self, score, input_batch):
    """Mask duplicate speaker scores in the score matrix and update the labels.

    Illustration on a 6*3 matrix of 3 speakers.
      1 [(11)  12   13*]
      2 [(21)  22   23*]
      3 [ 31  (32)  33 ]
      4 [ 41  (42)  43 ]
      5 [ 51*  52  (53)]
      6 [ 61*  62  (63)]
    The label tensor in this example is [0, 0, 1, 1, 0, 0] where the 5th and
    6th examples are from the same speaker of the 1st and 2nd examples. We mask
    the non-diagnoal, duplicate speaker target trials to be non-target, as
    shown in the matrix where *'s are. The brackets () represents for the
    target trials that remain untouched and those would be the only target
    trials that are considered in the following loss computations.

    The masking is subtracting a relatively large numeric value from the scores.

    Args:
      score: tensor of shape [batch, num_spks_per_batch] with scores of each
        example to each speaker.
      input_batch: A NestedMap containing:
        - src.src_inputs: tensor of shape [batch, len, dim].
        - src.paddings: tensor of shape [batch, len].
        - label: int tensor of shape [batch]. The inputs has ground truth labels
          required for the loss calculation.

    Returns:
      A tuple of (masked_score tensor, updated input_batch NestedMap), where
      the scores are masked and the input_batch's label tensor is also updated
      to be purely speaker indices (because the duplicate target trials have
      been masked and are now non-target trials).
    """
    p = self.params

    # Ensure the input shapes are correct
    assert len(py_utils.GetShape(score)) == 2
    assert len(py_utils.GetShape(input_batch.label)) == 1
    assert py_utils.GetShape(score)[0] == py_utils.GetShape(
        input_batch.label)[0]

    # Collect speakers
    label_2d = tf.reshape(input_batch.label,
                          [p.num_spks_per_batch, p.num_utts_per_spk])
    speakers = tf.gather(label_2d, [0], axis=1)
    speakers = tf.squeeze(speakers)

    # Create matrix for all target trials including duplicate speaker trials
    all_target_trials = tf.equal(
        tf.expand_dims(speakers, 0), tf.expand_dims(speakers, 1))

    # Create matrix for only the same speaker target trials, the diagonal
    idx = tf.cumsum(tf.ones_like(speakers)) - 1
    same_speaker_target_trials = tf.equal(
        tf.expand_dims(idx, 0), tf.expand_dims(idx, 1))

    # Calculate the maskings for the duplicate speaker target trials, which
    # we mask to treat as non-target trials.
    maskings = tf.cast(all_target_trials, tf.float32) - tf.cast(
        same_speaker_target_trials, tf.float32)

    # Repeat each row `num_utts_per_spk` times.
    maskings = tf.tile(maskings, tf.constant([1, p.num_utts_per_spk]))
    maskings = tf.reshape(
        maskings,
        [p.num_spks_per_batch * p.num_utts_per_spk, p.num_spks_per_batch])
    maskings *= p.mask_value

    # Mask the scores
    score -= maskings

    # Update the label tensor accordingly as well.
    # Note that this updated label tensor might or might not be used
    # depending on the use case. For example, for the non-MR GE2E loss, we
    # have to use this for the cross-entropy loss, but for the MR loss, this
    # is not used at all, because MR has its own transformation.
    spk_idx = tf.range(p.num_spks_per_batch)
    spk_idx = tf.reshape(spk_idx, [-1, 1])
    spk_idx = tf.tile(spk_idx, [1, p.num_utts_per_spk])
    spk_idx = tf.reshape(spk_idx, [p.num_spks_per_batch * p.num_utts_per_spk])
    assert py_utils.GetShape(input_batch.label) == py_utils.GetShape(spk_idx)
    input_batch.label = spk_idx

    # Verify that after masking there must be no duplicates.
    input_batch.label = py_utils.with_dependencies([
        py_utils.assert_equal(
            self.BatchWithoutDuplicateSpeakers(input_batch), 1)
    ], input_batch.label)

    return score, input_batch

  def ComputeSimilarity(self, logits, theta=None):
    """Top-level compute similarity score function.

    Args:
      logits: output data of network Tensor([batch, dims])
      theta: A NestedMap containing layer weights.
    Returns:
      cosine similarity matrix between each data and each speaker centroid.
      Tensor([batch, num_spks_per_batch])
    """
    p = self.params

    # [batch, num_spks_per_batch]
    if p.select_embedding_comparison_type == EmbeddingComparisonType.COSINE:
      if p.split_batch:
        score = self.ComputeSimilaritySplit(logits)
      else:
        score = self.ComputeSimilaritySimple(logits)
    elif (p.select_embedding_comparison_type ==
          EmbeddingComparisonType.ATTENTIVE_SCORING):
      score = self.ComputeAttentionScoringSimilaritySplit(logits, theta)
    else:
      raise ValueError('Invalid value for select_embedding_comparison_type.')

    return score

  def ComputeSimilaritySimple(self, logits):
    """Compute similarity score of each data to each speaker centroid.

    Each centroid is computed using all embeddings of a given speaker.

    Args:
      logits: output data of network Tensor([batch, dims])

    Returns:
      cosine similarity matrix between each data and each speaker centroid.
      Tensor([batch, num_spks_per_batch])
    """
    p = self.params
    return utils.ComputeSimilaritySimple(logits, p.num_spks_per_batch,
                                         p.num_utts_per_spk)

  def ComputeSimilaritySplit(self, logits):
    """Compute score of each data to each speaker centroid, with split batches.

    Each centroid is computed using all embeddings of a given speaker. The batch
    is split into the enrollment and test parts so that the loss is computed in
    a non-biased way.

    Args:
      logits: output data of network Tensor([batch, dims])

    Returns:
      cosine similarity matrix between each data and each speaker centroid.
      Tensor([batch, num_spks_per_batch])
    """
    p = self.params
    return utils.ComputeSimilaritySplit(
        logits, p.num_spks_per_batch, p.num_utts_per_spk,
        p.vary_number_of_enrollment_utterances_averaged)

  def ComputeAttentionScoringSimilaritySplit(self,
                                             segment_representations,
                                             theta=None):
    """Compute attention based scoring of minibatch.

    Compute the attention based score of each test segment embedding against
    each enrollment speaker (with multiple utterance embeddings), with split
    batches.

    Args:
      segment_representations: A batch of utterance representations. Tensor is
        of shape ([batch, dims]) with elements of tf.float32.
      theta: A NestedMap containing layer weights.

    Returns:
      attention based scoring similarity matrix between each test segment and
      each enrollment speaker (each enrollment speaker with their group of
      enrollment segments). Tensor([batch, num_spks_per_batch]), tf.float32.
    """
    p = self.params

    segment_representations_3d = tf.reshape(
        segment_representations, [p.num_spks_per_batch, p.num_utts_per_spk, -1])

    def _AttentionScoringGivenDataPart(part):
      """Compute attentive scoring for one of two parts.

      This helper function extracts the speaker test and enrollment
      representations according to the appropriate part selected. When part is
      0, all even numbered representations (based on index) are used as test
      segments while all odd numbered representations are used as enrollment
      speech segments. When part is 1, the assignments are switched.

      Args:
        part: Select which data part to use as test (0 or 1).

      Returns:
        scores: For a data part, calculates the attention based scores of all
          test utterances against all enrollment speakers.
          Size: [num_test_utterances, num_enroll_spks] as tf.float32
      """
      if part not in {0, 1}:
        raise ValueError('Must select an appropriate data part, either 0 or 1.')

      # Split enroll and test segments
      test_utts_3d = segment_representations_3d[:, part::2, :]
      test_utts = tf.reshape(test_utts_3d, [-1, test_utts_3d.shape[2]])

      other_part = 0 if part else 1
      enroll_utts_3d = segment_representations_3d[:, other_part::2, :]

      # Compare the test utterances against the enrollment speakers (comprising
      # multiple utterances per enrollment speaker).
      scores = self.attentive_scoring.FProp(
          inputs=(test_utts, enroll_utts_3d), theta=theta.attentive_scoring)

      return scores

    # Calculate the scores of the first set of test segments against the second
    # set of enrollment segments. Then switch them.
    scores_fst = _AttentionScoringGivenDataPart(0)
    scores_snd = _AttentionScoringGivenDataPart(1)

    # attention_similarity:
    # [batch // 2, 2, num_spks_per_batch] -> [batch, num_spks_per_batch]
    attention_similarity = tf.reshape(
        tf.stack([scores_fst, scores_snd], axis=1),
        [p.batch_size, p.num_spks_per_batch])
    return attention_similarity

  def TransformScoreAndLabel(self, theta, score, input_batch):
    """Transform the score and label tensors.

    This transformation includes the default linear transformation and masking
    if specified. This could be overridden by subclasses if there is some
    additional transformation logic, such as extended-set softmax, etc.

    Args:
      theta: A NestedMap containing layer weights.
      score: tensor of shape [batch, num_spks_per_batch] with scores of each
        example to each speaker.
      input_batch: A NestedMap containing at least:
        - label: int tensor of shape [batch]. The inputs has ground truth labels
          required for the loss calculation.

    Returns:
      A tuple of two elements:
        - A transformed score tensor, which could be of a different shape.
        - A NestedMap input_batch containing the transformed label tensor, which
          could be of a different shape.
    """
    p = self.params

    # [batch, num_spks_per_batch], linear transformation
    score = theta.w * score + theta.b

    # Mask the duplicate speaker related scores with negative values and
    # update the label tensor accordingly after masking the target trials
    # manually to be non-target trials.
    if p.mask_dup_spk_scores:
      score, input_batch = self.MaskDuplicateSpeakerScores(score, input_batch)

    return score, input_batch

  def CalculateLoss(self, theta, score, input_batch):
    """Calculate the loss given the scores and input batch labels.

    Args:
      theta: A NestedMap containing layer weights.
      score: tensor of shape [batch, num_proxy_classes] with scores of each
        example to each proxy centroid.
      input_batch: A NestedMap containing at least:
        - label: int tensor of shape [batch]. The inputs has ground truth labels
          required for the loss calculation.

    Returns:
      A tuple of two elements:
        - A transformed score tensor, which could be of a different shape.
        - A NestedMap input_batch containing the transformed label tensor, which
          could be of a different shape.
    """
    p = self.params

    # [batch]
    per_example_xent, _ = self.softmax.XentLossFromLogits(
        theta=theta.softmax,
        logits=score,
        class_weights=tf.ones_like(score.shape[1], dtype=tf.float32),
        class_ids=input_batch.label)
    avg_xent = tf.reduce_mean(per_example_xent, name='batch_loss_mean')

    metrics = py_utils.NestedMap()
    batch_size = py_utils.GetShape(input_batch.label)[0]
    # Convert the weight of batch_size into a tensor to prevent test time error
    weight = tf.constant(batch_size)
    # When we skip masking duplicate speaker scores, we need to assign zero
    # weight to the batches with duplicate speakers.
    if not p.mask_dup_spk_scores:
      weight *= self.batch_without_duplicate_speakers
    metrics.loss = (avg_xent, weight)
    return metrics

  def FProp(self, theta, predictions, input_batch):
    """Computes the GEnd2End softmax loss and other metrics.

    Args:
      theta: A NestedMap containing layer weights.
      predictions: A NestedMaps of the predictions:
        - logits: tensor of shape [batch, dvector_size].
        - paddings: tensor of shape [batch, len].
        - state1: RNN output states, a NestedMap, for each RNN layer containing:
          - m: the lstm output. [batch, output_nodes]
          - c: the lstm cell state. [batch, hidden_nodes]
      input_batch: A NestedMap containing:
        - src.src_inputs: tensor of shape [batch, len, dim].
        - src.paddings: tensor of shape [batch, len].
        - label: int tensor of shape [batch]. The inputs has ground truth labels
          required for the loss calculation.

    Returns:
      A tuple of two elements:
        - A NestedMap containing str keys and (metric, weight) values, where
          one of the keys is expected to be 'loss'.
        - A NestedMap containing arbitrary tensors describing something about
          each training example, where the first dimension of each tensor is
          the batch index.
    """
    # Inspect the batch to check duplicates before any transformation.
    self.batch_without_duplicate_speakers = self.BatchWithoutDuplicateSpeakers(
        input_batch)
    score = self.ComputeSimilarity(predictions.logits, theta)
    score, input_batch = self.TransformScoreAndLabel(theta, score, input_batch)
    metrics = self.CalculateLoss(theta, score, input_batch)
    return metrics, py_utils.NestedMap()


class GEnd2EndExtendedSetSoftmaxLayer(GEnd2EndSoftmaxLayer):
  """The generalized end-to-end extended-set softmax loss layer.

  This expands upon the standard row based softmax with localised cross-entropy.
  It provides improved score normalization (for speaker accept/reject
  thresholding purposes and better overall performance) by including non-target
  scores from other test segments. This encourages target scores to not only be
  large relative to the scores of the corresponding non-target models but also
  large relative to the scores of other test segment and model pairings. This
  code is currently designed for the standard text-independent data layout.

  To learn more about this loss, refer to this paper:
  https://arxiv.org/abs/2104.01989
  """

  @classmethod
  def Params(cls):
    p = super(GEnd2EndExtendedSetSoftmaxLayer, cls).Params()
    p.name = 'gend2end_extended_set_softmax_layer'
    return p

  def TransformScoreAndLabel(self, theta, score, input_batch):
    """Transformation with additional extended-set transformation.

    Args:
      theta: A NestedMap containing layer weights.
      score: tensor of shape [batch, num_spks_per_batch] with scores of each
        example to each speaker.
      input_batch: A NestedMap containing at least:
        - label: int tensor of shape
          [batch]. The inputs has ground truth labels required for the loss
          calculation.

    Returns:
      A tuple of two elements:
        - A transformed score tensor of shape [batch, num_proxy_classes]
        - A NestedMap input_batch containing the transformed label tensor of
          shape [batch, num_proxy_classes]
    """
    # Linear transformation and masking if specified.
    score, input_batch = super().TransformScoreAndLabel(theta, score,
                                                        input_batch)

    # Rework the scores
    refactored_scores, refactored_labels = self._RefactorExtendedSetInput(score)
    # New label tensor containing all zeros to represent that the zeroth indexed
    # column (1st column) in refactored_scores contains the target scores.
    input_batch.label = refactored_labels
    return refactored_scores, input_batch

  def _TransformStripedToStackedMatrix(self, striped_score_matrix):
    """Transforms a partially striped matrix to a stacked matrix.

    Given a striped score matrix (the striping pattern relates to the
    targets in the score matrix) it produces a stacked 'diagonal' matrix by
    swapping the rows of the input matrix.

    Args:
      striped_score_matrix: A 2-D tensor of size ([num_segments, num_models])
        containing the speaker scores.

    Returns:
      stacked_diagonal matrix: A 2-D tensor of stacked square matrices where
        each matrix consists of a diagonal of target scores and a
        non-diagonal of non-target scores.

    Explanation:
      For example, on the left is a striped score matrix with the target
        scores identified using round brackets (). On the right is the
        corresponding output from the function

          Row                     From Row
           1 [(11)  12   13 ]        1 [(11)  12   13 ]
           2 [(21)  22   23 ]        3 [ 31  (32)  33 ]
           3 [ 31  (32)  33 ]   -->  5 [ 51   52  (53)]
           4 [ 41  (42)  43 ]        2 [(21)  22   23 ]
           5 [ 51   52  (53)]        4 [ 41  (42)  43 ]
           6 [ 61   62  (63)]        6 [ 61   62  (63)]
    """

    # Determine the number of test segments and the number of models
    num_segs, num_models = [
        int(i) for i in striped_score_matrix.shape.as_list()
    ]

    # Calculate the number of test segments per speaker model
    # (verify that num_segs is a multiple of num_models)
    if num_segs % num_models != 0:
      raise ValueError(
          'Note that num_segs (%i) is not a multiple of num_models (%i).' %
          (num_segs, num_models))

    segs_per_speaker = num_segs // num_models

    # Specify the new row ordering
    row_ordering = tf.reshape(
        tf.transpose(
            tf.reshape(
                tf.range(0, num_segs, dtype=tf.int32),
                [segs_per_speaker, num_models])), [num_segs])

    # Apply the row ordering to the matrix
    stacked_diagonal_target_score_matrix = tf.scatter_nd(
        indices=tf.expand_dims(input=row_ordering, axis=1),
        updates=striped_score_matrix,
        shape=[num_segs, num_models])

    return stacked_diagonal_target_score_matrix

  def _ExtractTargetScoresFromStackedMatrix(self,
                                            stacked_diagonal_score_matrix):
    """Given a stacked 'diagonal' matrix, extracts corresponding target scores.

    Args:
      stacked_diagonal_score_matrix: A 2-D tensor of stacked square matrices
        where each matrix consists of a diagonal of target scores and a
        non-diagonal of non-target scores.

    Returns:
      tar_scores: A 1-D tensor of the target scores

    Explanation:
      For example, on the left is a stacked matrix and on the right are the
      corresponding target scores.

        Row
         1 [(11)  12   13 ]
         2 [ 31  (32)  33 ]
         3 [ 51   52  (53)]    -->   [(11) (32) (53) (21) (42) (63)]
         4 [(21)  22   23 ]
         5 [ 41  (42)  43 ]
         6 [ 61   62  (63)]
    """

    # Determine the number of test segments and the number of models
    num_segs, num_models = [
        int(i) for i in stacked_diagonal_score_matrix.shape.as_list()
    ]

    # Calculate the number of test segments per speaker model
    # (verify that num_segs is a multiple of num_models)
    if num_segs % num_models != 0:
      raise ValueError(
          'Note that num_segs (%i) is not a multiple of num_models (%i).' %
          (num_segs, num_models))

    segs_per_speaker = num_segs // num_models

    # Extract the target speaker scores
    tar_mask = tf.tile(tf.eye(num_models, dtype=tf.bool), [segs_per_speaker, 1])
    tar_scores = tf.boolean_mask(stacked_diagonal_score_matrix, tar_mask)

    return tar_scores

  def _ExtractNonTargetScoresForEachStackedMatrix(
      self, stacked_diagonal_score_matrix):
    """Extracts non-target scores from a stacked 'diagonal' matrix.

    For each stacked 'diagonal' matrix this function will produce a row of
    non-target scores. If there are N stacked matrices, there will be N rows.
    If the number of columns in the matrix is C, then the number of non-target
    scores for each row will be Cx(C-1).

    Args:
      stacked_diagonal_score_matrix: A 2-D tensor of stacked square matrices
        where each matrix consists of a diagonal of target scores and a
        non-diagonal of non-target scores.

    Returns:
      non_scores: A 1-D tensor of the non-target scores

    Explanation:
      For example, on the left is a stacked matrix and on the right are the
      corresponding non-target scores. The target scores are shown within
      brackets ().

        Row
         1 [(11)  12   13 ]
         2 [ 31  (32)  33 ]
         3 [ 51   52  (53)]    -->   [ 12  13  31  33  51  52]
         4 [(21)  22   23 ]          [ 22  23  41  43  61  62]
         5 [ 41  (42)  43 ]
         6 [ 61   62  (63)]
    """

    # Determine the number of test segments and the number of models
    num_segs, num_models = [
        int(i) for i in stacked_diagonal_score_matrix.shape.as_list()
    ]

    # Calculate the number of test segments per speaker model
    # (verify that num_segs is a multiple of num_models)
    if num_segs % num_models != 0:
      raise ValueError(
          'Note that num_segs (%i) is not a multiple of num_models (%i).' %
          (num_segs, num_models))

    segs_per_speaker = num_segs // num_models

    # Extract the non-target speaker scores
    non_mask = tf.tile(
        tf.math.logical_not(tf.eye(num_models, dtype=tf.bool)),
        [segs_per_speaker, 1])
    non_scores = tf.boolean_mask(stacked_diagonal_score_matrix, non_mask)

    # From each row of the output matrix, collect the non-target scores from
    # the same stacked 'diagonal' matrix.
    grouped_non_scores = tf.reshape(
        non_scores, [segs_per_speaker, num_models * (num_models - 1)])

    return grouped_non_scores

  def _RepeatRows(self, in_matrix, repeat_n_times):
    """Repeats each row of a matrix multiple times to produce a new matrix.

    Given a matrix this function will return a new matrix where each row
    becomes a repeated version of each row. For a PxQ input matrix, the
    output matrix will now have P times N rows and Q columns. An example
    follows below.

    Args:
      in_matrix: A PxQ matrix.
      repeat_n_times: How many times to repeat each row within the matrix.

    Returns:
      repeated_rows_matrix: A matrix calculated from in_matrix by repeating its
      rows.

    Explanation:
      For example, on the left is the input matrix (in_matrix), and
      repeat_n_times is set to 2. This produces the resulting matrix on the
      right.

        Row                        Row
         1 [(11)  12   13 ]         1 [(11)  12   13 ]
         2 [ 31  (32)  33 ]         2 [(11)  12   13 ]
         3 [ 51   52  (53)]    -->  3 [ 31  (32)  33 ]
                                    4 [ 31  (32)  33 ]
                                    5 [ 51   52  (53)]
                                    6 [ 51   52  (53)]
    """

    # Determine the number of matrix rows and columns
    num_rows, num_cols = [int(i) for i in in_matrix.shape.as_list()]

    # Horizontally stack the matrix multiple times
    stacked_matrix = tf.tile(
        input=in_matrix, multiples=tf.constant([1, repeat_n_times]))

    # Reshape the matrix to produce the final repeated rows
    repeated_rows_matrix = tf.reshape(stacked_matrix,
                                      [repeat_n_times * num_rows, num_cols])

    return repeated_rows_matrix

  def _RefactorExtendedSetInput(self, score):
    """Refactors the input score matrix into extended-set softmax form.

    Transform the scores such that the first column has the same-speaker
    scores and other columns have the (extended-set) different-speaker
    scores.

    Args:
      score: A 2-D tensor of size ([num_segments, num_models]) containing the
        speaker scores.

    Returns:
      final_scores: A 2-D tensor of scores with the first column containing
        target scores and the other columns containing the non-target scores.
        Size: ([num_segments, num_models*(num_models-1)+1])
      final_labels: A same-speaker label vector of zeros indicating that the
        same-speaker score is in the first column of final_scores.
    """

    # Determine the number of test segments and the number of models
    num_segs, num_models = [int(i) for i in score.shape.as_list()]

    # Convert the striped score matrix into a stacked "diagonal" matrix
    # Input shape: score = [num_segs x num_models]
    # Output shape: stacked_matrix = [num_segs x num_models]
    stacked_matrix = self._TransformStripedToStackedMatrix(score)

    # Extract the target scores from a stacked "diagonal" matrix
    # Input shape: stacked_matrix = [num_segs x num_models]
    # Output shape: tar_scores = [1 x num_segs]
    tar_scores = self._ExtractTargetScoresFromStackedMatrix(stacked_matrix)

    # Extract the non-target scores from a stacked "diagonal" matrix.
    # The difference this time is that the non-target scores for each
    # stacked square matrix are collected as a single row in the output
    # matrix.
    # Input shape: stacked_matrix = [num_segs x num_models]
    # Output shape: grouped_non_scores =
    #   [(num_segs/num_models) x (num_models*(num_models-1))]
    grouped_non_scores = self._ExtractNonTargetScoresForEachStackedMatrix(
        stacked_matrix)

    # Repeat the contents of each row multiple times to correspond to
    # the target trials in each stacked matrix
    # Input shape: grouped_non_scores =
    #   [(num_segs/num_models) x (num_models*(num_models-1))]
    # Output shape: repeated_non_scores =
    #   [num_segs x (num_models*(num_models-1))]
    repeated_non_scores = self._RepeatRows(grouped_non_scores, num_models)

    # Place the target scores at the beginning and the non-targets second
    # Input shapes: #1 [num_seg x 1]
    #               #2 [num_segs x (num_models*(num_models-1))]
    # Output shape: final_scores [num_seg x (num_models*(num_models-1)+1)]
    final_scores = tf.concat(
        values=[tf.expand_dims(tar_scores, 1), repeated_non_scores], axis=1)

    # Target trials are in column 0, so create a label key with a zeros
    final_labels = tf.zeros([num_segs], dtype=tf.int32)

    return final_scores, final_labels
