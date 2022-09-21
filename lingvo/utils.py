"""Helper functions for modeling."""

from lingvo import compat as tf
from lingvo.core import py_utils


def GetLastSeqOutput(sequence_output, padding):
  """Gets the last sequence frame based on the paddings.

  Args:
    sequence_output: a tf.float32 tensor of size [time, batch, feature_dim]
    padding: a tf.float32 tensor of size [time, batch]

  Returns:
    last_frame: a tf.float32 tensor of size [batch, feature_dim]
  """
  sequence_output = py_utils.HasRank(sequence_output, 3)
  padding = py_utils.HasRank(padding, 2)

  sequence_output = py_utils.with_dependencies([
      py_utils.assert_equal(
          py_utils.GetShape(sequence_output, 2),
          py_utils.GetShape(padding, 2),
          message=('The first two dimensions of the two tensors '
                   f'must be the same: {py_utils.GetShape(sequence_output, 2)},'
                   f'{py_utils.GetShape(padding, 2)}')),
  ], sequence_output)

  # Last frame index
  idx_offset = 1
  # [batch]
  last_output_idx = tf.cast(
      tf.reduce_sum(1 - padding, axis=[0]) - idx_offset, dtype=tf.int32)

  batch_size = tf.shape(last_output_idx)[0]
  # [batch, 2]
  gather_idx = tf.stack(
      [last_output_idx, tf.range(batch_size, dtype=tf.int32)], axis=1)
  # [batch, feature_dim]
  last_frame = tf.gather_nd(sequence_output, gather_idx)
  return last_frame


def ComputeSimilaritySimple(logits, num_spks_per_batch, num_utts_per_spk):
  """Compute similarity score of each data to each speaker centroid.

  Each centroid is computed using all embeddings of a given speaker.

  Args:
    logits: output data of network Tensor([batch, dims])
    num_spks_per_batch: int, the number of speakers per batch
    num_utts_per_spk: int, the number of utterances per speaker

  Returns:
    cosine similarity matrix between each data and each speaker centroid.
    Tensor([batch, num_spks_per_batch])
  """
  # Normalize logits: [batch, dims]
  logits = tf.nn.l2_normalize(logits, axis=1)

  # logits_3d: [num_spks_per_batch, num_utts_per_spk, dims]
  logits_3d = tf.reshape(logits, [num_spks_per_batch, num_utts_per_spk, -1])

  # Compute centroid for each spk: [num_spks_per_batch, dims]
  centroid = tf.reduce_mean(logits_3d, axis=1)
  centroid = tf.nn.l2_normalize(centroid, axis=1)

  # Cosine similarity: [batch, num_spks_per_batch]
  cos_similarity = tf.matmul(logits, centroid, transpose_b=True)
  return cos_similarity


def ComputeSimilaritySplit(logits,
                           num_spks_per_batch,
                           num_utts_per_spk,
                           vary_number_of_enrollment_utterances_averaged=False):
  """Compute score of each data to each speaker centroid, with split batches.

  Each centroid is computed using all embeddings of a given speaker. The batch
  is split into the enrollment and test parts so that the loss is computed in
  a non-biased way.

  Args:
    logits: output data of network Tensor([batch, dims])
    num_spks_per_batch: int, the number of speakers per batch
    num_utts_per_spk: int, the number of utterances per speaker
    vary_number_of_enrollment_utterances_averaged: bool. If True, will
      randomize how many utterances are selected for enrollment. This makes the
      training challenge more difficult. If False (default), the average of all
      appropriate enrollment utterances is taken.

  Returns:
    cosine similarity matrix between each data and each speaker centroid.
    Tensor([batch, num_spks_per_batch])
  """
  batch_size = num_spks_per_batch * num_utts_per_spk

  # Normalize logits: [batch, dims]
  logits = tf.nn.l2_normalize(logits, axis=1)

  # logits_3d: [num_spks_per_batch, num_utts_per_spk, dims]
  logits_3d = tf.reshape(logits, [num_spks_per_batch, num_utts_per_spk, -1])

  def _GetCentroidAndData(part):
    """Compute Centroid and Data for one of two parts with length norm.

    This helper function extracts the speaker d-vectors and the speaker
    d-vector means for half of each speaker's data. When part is 0,
    all even numbered d-vector INDEXES are used for processing the d-vectors
    into test d-vectors and centroid models. When part is 1, all odd numbered
    d-vector INDEXES are utilized in the same manner.

    Args:
      part: Select which data part to use (0 or 1).

    Returns:
      centroid_part: For a data part, calculates the speaker specific mean.
      logits_part: The actual data part.
    """
    if part not in {0, 1}:
      raise ValueError('Must select an appropriate data part, either 0 or 1.')

    # logits_3d_part: [num_spks_per_batch, num_utts_per_spk // 2, dims]
    logits_3d_part = logits_3d[:, part::2, :]

    # logits_part: [batch // 2, dims]
    logits_part = tf.reshape(logits_3d_part, [batch_size // 2, -1])

    if vary_number_of_enrollment_utterances_averaged:
      # Compute centroid based on randomly selecting the number of utterances
      # to use followed by their random selection. The number of utterances used
      # is defined according to a uniform distribution from 1 up to the maximum
      # number of utterances.

      max_utt = num_utts_per_spk // 2

      # Randomly select the number of enrollment utterances to use for each
      # speaker and generate the sequence mask.
      lengths = tf.random.uniform(
          shape=(num_spks_per_batch,),
          minval=1,
          maxval=max_utt + 1,
          dtype=tf.dtypes.int32)
      mask = tf.sequence_mask(lengths, maxlen=max_utt, dtype=tf.dtypes.float32)

      # Randomize the sequence mask on each row.
      indices = tf.argsort(
          tf.random.uniform(shape=(num_spks_per_batch, max_utt)))
      mask_row_randomized = tf.gather(mask, indices, batch_dims=-1)

      # Calculate the average based on the mask
      sum_x = tf.reduce_sum(
          tf.math.multiply(logits_3d_part, mask_row_randomized[:, :,
                                                               tf.newaxis]),
          axis=1)
      sum_n = tf.cast(lengths, dtype=tf.float32)
      centroid_part = tf.math.divide(sum_x, sum_n[:, tf.newaxis])
    else:
      # Compute centroid for each spk: [num_spks_per_batch, dims]
      centroid_part = tf.reduce_mean(logits_3d_part, axis=1)

    centroid_part = tf.nn.l2_normalize(centroid_part, axis=1)
    return centroid_part, logits_part

  centroid_fst, logits_fst = _GetCentroidAndData(0)
  centroid_snd, logits_snd = _GetCentroidAndData(1)

  # cos_similarity_fst: [batch // 2, num_spks_per_batch]
  # cos_similarity_snd: [batch // 2, num_spks_per_batch]
  cos_similarity_fst = tf.matmul(logits_fst, centroid_snd, transpose_b=True)
  cos_similarity_snd = tf.matmul(logits_snd, centroid_fst, transpose_b=True)

  # cos_similarity:
  # [batch // 2, 2, num_spks_per_batch] -> [batch, num_spks_per_batch]
  cos_similarity = tf.reshape(
      tf.stack([cos_similarity_fst, cos_similarity_snd], axis=1),
      [batch_size, num_spks_per_batch])

  return cos_similarity
