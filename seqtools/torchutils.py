import logging

import numpy as np

import torch
import torch.utils.data
import torch_struct


logger = logging.getLogger(__name__)


# -=( MISC )==-----------------------------------------------------------------
def tensorFromSequence(sequence, **tensor_kwargs):
    """ Stack each item of a sequence along the final dimension.

    If the input has shape (num_features,), the output will have shape
    (num_features, num_samples).

    Parameters
    ----------
    sequence : iterable( numpy array )

    Returns
    -------
    tensor : torch.Tensor, shape (..., num_samples)
    """

    after_last_dim = len(sequence[0].shape)

    tensor = torch.stack(
        tuple(torch.tensor(item, **tensor_kwargs) for item in sequence),
        dim=after_last_dim
    )

    return tensor


# -=( TRAINING & EVALUATION )=-------------------------------------------------
def guessStructure(log_potentials):
    """ Guess the structure associated with a table of log potentials.

    This structure is used in dynamic programs for training and decoding.

    Parameters
    ----------
    log_potentials : torch.Tensor of float

    Returns
    -------
    structure : torch_struct.StructDistribution, optional
        This functions returns None of there is no structure in the output
        (i.e. outputs are independent).
    """

    # Ignore axis 0 (batch size) when examining the output shape
    table_shape = log_potentials.shape[1:]
    num_table_dims = len(table_shape)

    if num_table_dims in (1, 2):
        return None
    if num_table_dims == 3 and table_shape[-2] == table_shape[-1]:
        return torch_struct.LinearChainCRF
    elif num_table_dims == 4 and table_shape[-2] == table_shape[-1]:
        return torch_struct.SemiMarkovCRF

    err_str = f"Can't guess structure for input with shape {log_potentials.shape}"
    raise ValueError(err_str)


def StructuredNLLLoss(log_potentials, label, lengths=None, structure=None):
    """ Efficiently compute structured negative-log-likelihood loss using dynamic programming.

    Parameters
    ----------
    log_potentials : torch.Tensor of float
        Shape depends on `structure`.
    label : torch.Tensor of int, shape (batch_size, num_samples)
    structure : torch_struct.StructDistribution, optional
        If this argument is omitted, the function will try to choose a CRF-style
        structure that is compatible with the shape of the input.

    Returns
    -------
    loss : float
    """

    num_classes = log_potentials.shape[-1]

    if structure is None:
        structure = guessStructure(log_potentials)

    dist = structure(log_potentials, lengths=lengths)

    label_events = (
        structure.struct.to_parts(label, num_classes, lengths=lengths)
        .type_as(dist.log_potentials)
    )
    loss = -dist.log_prob(label_events).sum()

    # logger.info(f"labels: {label[0]}")

    return loss


def fillSegments(pred_batch, in_place=False):
    """ Fill the interior of segments decoded by torch_struct.SemiMarkovCRF

    Parameters
    ----------
    pred_batch :
    in_place : bool, optional
    """

    if not in_place:
        raise NotImplementedError()

    for sample_index in range(pred_batch.shape[0]):
        segment_bounds = (pred_batch[sample_index] != -1).nonzero().squeeze()
        for seg_start, next_seg_start in zip(segment_bounds[:-1], segment_bounds[1:]):
            seg_label = pred_batch[sample_index, seg_start]
            pred_batch[sample_index, seg_start:next_seg_start] = seg_label


# -=( DATASETS )=--------------------------------------------------------------
class SequenceDataset(torch.utils.data.Dataset):
    """ A dataset wrapping sequences of numpy arrays stored in memory.

    Attributes
    ----------
    _data : tuple(np.ndarray, shape (num_samples, num_dims))
    _labels : tuple(np.ndarray, shape (num_samples,))
    _device : torch.Device
    """

    def __init__(
            self, data, labels, device=None, labels_type=None, sliding_window_args=None,
            transpose_data=False):
        """
        Parameters
        ----------
        data : iterable( array_like of float, shape (sequence_len, num_dims) )
        labels : iterable( array_like of int, shape (sequence_len,) )
        device :
        labels_type : string in {'float'}, optional
            If passed, labels will be converted to float type.
        sliding_window_args : tuple(int, int, int), optional
            A tuple specifying parameters for extracting sliding windows from
            the data sequences. This should be ``(dimension, size, step)``---i.e.
            the input to ``torch.unfold``. The label of each sliding window is
            taken to be the median over the labels in that window.
        """

        self.num_obsv_dims = data[0].shape[1]

        if len(labels[0].shape) == 2:
            self.num_label_types = labels[0].shape[1]
        elif len(labels[0].shape) < 2:
            self.num_label_types = np.unique(np.hstack(labels)).shape[0]
        else:
            err_str = f"Labels have a weird shape: {labels[0].shape}"
            raise ValueError(err_str)

        self.sliding_window_args = sliding_window_args
        self.transpose_data = transpose_data

        self._device = device
        self._labels_type = labels_type

        self._data = data
        self._labels = labels

        logger.info('Initialized ArrayDataset.')
        logger.info(
            f"Data has dimension {self.num_obsv_dims}; "
            f"{self.num_label_types} unique labels"
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        data_seq = torch.tensor(self._data[i], dtype=torch.float)
        label_seq = torch.tensor(self._labels[i])

        if self._device is not None:
            data_seq = data_seq.to(device=self._device)
            label_seq = label_seq.to(device=self._device)

        if self._labels_type == 'float':
            label_seq = label_seq.float()

        # shape (sequence_len, num_dims) --> (num_dims, sequence_len)
        if self.transpose_data:
            data_seq = data_seq.transpose(0, 1)

        if self.sliding_window_args is not None:
            # Unfold gives shape (sequence_len, window_len);
            # after transpose, data_seq has shape (window_len, sequence_len)
            data_seq = data_seq.unfold(*self.sliding_window_args).transpose(-1, -2)
            label_seq = label_seq.unfold(*self.sliding_window_args).median(dim=-1).values

        return data_seq.contiguous(), label_seq.contiguous()


# -=( MODELS )=----------------------------------------------------------------
class LinearChainScorer(object):
    """ Pytorch mixin that creates a linear-chain score table.

    Attributes
    ----------
    transition_weights : torch.Tensor of float, shape (num_classes, num_classes)
        Element (i, j) corresponds to (cur segment == i, prev segment == j);
        i.e. the transpose of how a HMM's transition array is usually arranged.
    start_weights : torch.Tensor of float, shape (num_classes,), optional
    end_weights : torch.Tensor of float, shape (num_classes,), optional
    """

    def __init__(
            self, *super_args,
            transition_weights=None, initial_weights=None, final_weights=None,
            requires_grad=False,
            **super_kwargs):
        """
        Parameters
        ----------
        transitions : torch.Tensor of float, shape (num_classes, num_classes), optional
            Element (i, j) corresponds to (cur segment == i, prev segment == j);
            i.e. the transpose of how a HMM's transition array is usually arranged.
        initial_states : torch.Tensor of float, shape (num_classes,), optional
        final_states : torch.Tensor of float, shape (num_gestures,), optional
        requires_grad : bool, optional
            If True, start, end, and transition weights will be updated during
            training.
        *super_args, **super_kwargs : optional
            Remaining arguments are passed to super's init method.
        """

        super().__init__(*super_args, **super_kwargs)

        if transition_weights is None:
            return

        if final_weights is None:
            final_weights = torch.full(transition_weights.shape[0:1], 0)

        if initial_weights is None:
            initial_weights = torch.full(transition_weights.shape[0:1], 0)

        self.transition_weights = torch.nn.Parameter(
            transition_weights, requires_grad=requires_grad
        )
        self.initial_weights = torch.nn.Parameter(
            initial_weights, requires_grad=requires_grad
        )
        self.final_weights = torch.nn.Parameter(
            final_weights, requires_grad=requires_grad
        )

    def predict(self, outputs):
        """

        Parameters
        ----------

        Returns
        -------
        """

        dist = torch_struct.LinearChainCRF(outputs)
        preds, _ = dist.struct.from_parts(dist.argmax)

        return preds

    def forward(self, input_seq):
        """ Construct a table of Markov (i.e. linear-chain) scores.

        These scores can be used to instantiate pytorch-struct's LinearChainCRF.

        Parameters
        ----------
        attribute_scores : torch.Tensor of float, shape (batch_size, ..., num_samples)
            NOTE: This method doesn't handle inputs with batch size > 1 (yet).

        Returns
        -------
        log_potentials : torch.Tensor of float,
                shape (batch_size, num_samples - 1, max_duration, num_classes, num_classes)
        """

        # Should be shape (batch_size, seq_len, vocab_size)
        scores = self.scoreSamples(input_seq)

        # Should be shape (batch_size, seq_len - 1, vocab_size, vocab_size)
        log_potentials = scores[:, 1:, ..., None].expand(-1, -1, -1, scores.shape[-1])
        log_potentials += self.transition_weights
        log_potentials[:, 0, ...] += (self.initial_weights + scores[:, 0])[:, :, None]
        log_potentials[:, -1, ...] += self.final_weights[:, :, None]

        """
        batch_size = scores.shape[0]  # input_seq.shape[0]
        num_samples = scores.shape[-1]
        num_labels = scores.shape[1]  # self.initial_weights.shape[0]

        scores_shape = (batch_size, num_samples - 1, num_labels, num_labels)
        log_potentials = torch.full(scores_shape, -float("Inf"), device=input_seq.device)

        for sample_index in range(1, num_samples):
            # FIXME: Line below only looks at first sample in batch. I think
            #   pytorch's default array broadcasting should make full-batch
            #   computations work just fine, but I haven't tried it.
            sample = input_seq[0:1, ..., sample_index]
            scores = self.scoreSample(sample)

            if sample_index == 1:
                scores += self.initial_weights + self.scoreSample(input_seq[0:1, ..., 0])
                # Arrange scores as a column vector so tensor broadcasting
                # replicates scores across columns---scores array needs to
                # have shape matching (cur segment) x (prev segment)
                scores = scores[:, :, None]
            elif sample_index == num_samples - 1:
                scores += self.final_weights
                # Arrange scores as a column vector so tensor broadcasting
                # replicates scores across columns---scores array needs to
                # have shape matching (cur segment) x (prev segment)
                scores = scores[:, :, None]
            # FIXME: Make sure transition dimensions are broadcast
            #   correctly for batch sizes > 1
            scores = scores + self.transition_weights

            log_potentials[0:1, sample_index - 1, :, :] = scores
        """

        return log_potentials

    def scoreSamples(self, input_seq):
        """ Score a sample.

        Parameters
        ----------
        input_seq : ???

        Returns
        -------
        scores : torch.Tensor of float, shape (batch_size, seq_len, num_classes)
        """

        # This flattens the segment if it was a sliding window of features
        # sample = sample.contiguous().view(sample.shape[0], sample.shape[1], -1)
        # return super().forward(sample)  # .sum(dim=-1)

        # super().forward should return a tensor of shape (batch_size, num_classes)
        seq_scores = tuple(
            super(LinearChainScorer, self).forward(sample)
            for sample in input_seq
        )
        return torch.stack(seq_scores, dim=1)


class SemiMarkovScorer(object):
    """ Pytorch mixin that creates a semi-Markov (segmental) score table.

    NOTE: Segment scores are implemented as a sum over the data scores of each
        frame in the segment. However, you can implement more complex scoring
        functions by inheriting from SemiMarkovScorer and overriding `scoreSegment`,
        or by monkey-patching it at runtime if you're a real daredevil.

    Attributes
    ----------
    transition_weights : torch.Tensor of float, shape (num_classes, num_classes)
        Element (i, j) corresponds to (cur segment == i, prev segment == j);
        i.e. the transpose of how a HMM's transition array is usually arranged.
    start_weights : torch.Tensor of float, shape (num_classes,), optional
    end_weights : torch.Tensor of float, shape (num_classes,), optional
    max_duration : int, optional
        If the duration is not bounded, the runtime complexity of most inference
        algorithms is quadratic in the sequence length instead of linear (times
        a constant factor), so it's better to bound the duration if you can.
    """

    def __init__(
            self, *super_args, transitions=None, initial_states=None, final_states=None,
            max_duration=None, update_transition_params=False,
            **super_kwargs):
        """
        Parameters
        ----------
        transitions : torch.Tensor of float, shape (num_classes, num_classes), optional
            Element (i, j) corresponds to (cur segment == i, prev segment == j);
            i.e. the transpose of how a HMM's transition array is usually arranged.
        initial_states : torch.Tensor of float, shape (num_classes,), optional
        final_states : torch.Tensor of float, shape (num_gestures,), optional
        update_transition_params : bool, optional
            If True, start, end, and transition weights will be updated during
            training.
        max_duration : int, optional
            Default is `num_samples`---note that this makes the runtime complexity of
            most inference algorithms scale quadratically with the sequence length
            instead of linearly, so it's better to bound the duration if you can
            (and you probably can).
        *super_args, **super_kwargs : optional
            Remaining arguments are passed to super's init method.
        """

        super().__init__(*super_args, **super_kwargs)

        transition_weights = torch.tensor(transitions).float().log()
        start_weights = torch.tensor(initial_states).float().log()
        end_weights = torch.tensor(final_states).float().log()

        self.transition_weights = torch.nn.Parameter(
            transition_weights, requires_grad=update_transition_params
        )
        self.start_weights = torch.nn.Parameter(
            start_weights, requires_grad=update_transition_params
        )
        self.end_weights = torch.nn.Parameter(
            end_weights, requires_grad=update_transition_params
        )
        self.max_duration = max_duration

    def predict(self, outputs):
        dist = torch_struct.SemiMarkovCRF(outputs)
        preds, _ = dist.struct.from_parts(dist.argmax)
        if (preds == -1).any():
            fillSegments(preds, in_place=True)
        # The first predicted label is spurious---it's the state from which the
        # initial state transitioned.
        preds = preds[:, 1:]
        return preds

    def forward(self, attribute_scores):
        """ Construct a table of semi-Markov (i.e. segmental) scores.

        These scores can be used to instantiate pytorch-struct's SemiMarkovCRF.

        Parameters
        ----------
        attribute_scores : torch.Tensor of float, shape (batch_size, ..., num_samples)
            NOTE: This method doesn't handle inputs with batch size > 1 (yet).

        Returns
        -------
        log_potentials : torch.Tensor of float,
                shape (batch_size, num_samples, max_duration, num_classes, num_classes)
        """

        batch_size = attribute_scores.shape[0]
        num_samples = attribute_scores.shape[-1]
        num_labels = self.start_weights.shape[0]
        max_duration = self.max_duration
        if max_duration is None:
            max_duration = num_samples

        scores_shape = (batch_size, num_samples, max_duration, num_labels, num_labels)
        log_potentials = torch.full(scores_shape, -float("Inf"), device=attribute_scores.device)

        for seg_start_index in range(num_samples):
            for seg_duration in range(1, max_duration):
                seg_end_index = seg_start_index + seg_duration
                if seg_end_index > num_samples:
                    continue

                # FIXME: Line below only looks at first sample in batch. I think
                #   pytorch's default array broadcasting should make full-batch
                #   computations work just fine, but I haven't tried it.
                segment = attribute_scores[0:1, ..., seg_start_index:seg_end_index]
                scores = self.scoreSegment(segment)

                if not seg_start_index:
                    scores += self.start_weights
                    # Arrange scores as a column vector so tensor broadcasting
                    # replicates scores across columns---scores array needs to
                    # have shape matching (cur segment) x (prev segment)
                    scores = scores[:, :, None]
                else:
                    if seg_end_index == num_samples:
                        scores += self.end_weights
                        # Arrange scores as a column vector so tensor broadcasting
                        # replicates scores across columns---scores array needs to
                        # have shape matching (cur segment) x (prev segment)
                        scores = scores[:, :, None]
                    # FIXME: Make sure transition dimensions are broadcast
                    #   correctly for batch sizes > 1
                    scores = scores + self.transition_weights

                log_potentials[0:1, seg_start_index, seg_duration, :, :] = scores

        return log_potentials

    def scoreSegment(self, segment):
        """ Score a segment.

        Parameters
        ----------
        segment : torch.Tensor of float, shape (batch_size, ..., segment_len)

        Returns
        -------
        scores : torch.Tensor of float, shape (batch_size, num_classes)
        """

        # This flattens the segment if it was a sliding window of features
        segment = segment.contiguous().view(segment.shape[0], segment.shape[1], -1)
        return super().forward(segment).sum(dim=-1)


class Hmm(torch_struct.HMM):
    pass
