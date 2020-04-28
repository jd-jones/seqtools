import logging

import torch
import torch.utils.data


from mathtools import utils

from . import torch_lctm


logger = logging.getLogger(__name__)


try:
    import torch_struct
except ImportError:
    torch_struct = None
    logger.warning(
        "No torch_struct installation detected "
        "(This is fine if you won't use any structured models)"
    )
    pass


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

        super(LinearChainScorer, self).__init__(*super_args, **super_kwargs)

        self.obsv_model = super(LinearChainScorer, self)

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

    def forward(self, *args, **kwargs):
        return self._create_sequence_scores(*args, **kwargs)

    def _create_sequence_scores(self, input_seq, scores_as_input=False):
        """ Construct a table of Markov (i.e. linear-chain) scores.

        These scores can be used to instantiate pytorch-struct's LinearChainCRF.

        Parameters
        ----------
        input_seq : torch.Tensor of float, shape (batch_size, ..., num_samples)
            NOTE: This method doesn't handle inputs with batch size > 1 (yet).
        scores_as_input : bool, optional

        Returns
        -------
        log_potentials : torch.Tensor of float,
                shape (batch_size, num_samples - 1, max_duration, num_classes, num_classes)
        """

        if scores_as_input:
            scores = input_seq
        else:
            # Should be shape (batch_size, seq_len, vocab_size)
            scores = self.scoreSamples(input_seq)

        # Should be shape (batch_size, seq_len - 1, vocab_size, vocab_size)
        log_potentials = scores[:, 1:, ..., None].expand(-1, -1, -1, scores.shape[-1])
        log_potentials = log_potentials + self.transition_weights
        log_potentials[:, 0, ...] += (self.initial_weights[None, :] + scores[:, 0])[:, :, None]
        log_potentials[:, -1, ...] += self.final_weights[None, :, None]

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

        seq_scores = utils.batchProcess(
            super(LinearChainScorer, self).forward,
            input_seq
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
            max_duration=None, update_transition_params=False, use_standard_dp=False,
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

        self.update_transition_params = update_transition_params
        self.max_duration = max_duration
        self.use_standard_dp = use_standard_dp

        if transitions is not None:
            transition_weights = torch.tensor(transitions).float().log()
            self.transition_weights = torch.nn.Parameter(
                transition_weights, requires_grad=self.update_transition_params
            )

        if initial_states is not None:
            start_weights = torch.tensor(initial_states).float().log()
            self.start_weights = torch.nn.Parameter(
                start_weights, requires_grad=self.update_transition_params
            )

        if final_states is not None:
            end_weights = torch.tensor(final_states).float().log()
            self.end_weights = torch.nn.Parameter(
                end_weights, requires_grad=self.update_transition_params
            )

    def predict(
            self, log_potentials, use_standard_dp=None, decode_method='viterbi',
            arc_labels=None, return_scores=False):
        """
        Parameters
        ----------
        log_potentials :
        return_edges : bool, optional
        use_standard_dp : bool, optional
            pytorch_struct uses a parallel-scan dynamic program to reduce the number
            of sequential semiring calls from `O(seq_len)` to `O(log seq_len)`.
            This increases the memory complexity of the algorithm, though. So if
            you are decoding on a big graph and you run out of memory, you can try
            using the standard DP.
        """

        if use_standard_dp is None:
            use_standard_dp = self.use_standard_dp

        dist = torch_struct.SemiMarkovCRF(log_potentials)
        if use_standard_dp:
            dist.struct._dp = _dp_standard_sm

        if decode_method == 'viterbi':
            preds, argmax = self._viterbi(dist, arc_labels=arc_labels)
            pred_scores = log_potentials.cpu()[argmax.byte()]
        elif decode_method == 'marginal':
            preds, marginals = self._marginalDecode(dist, arc_labels=arc_labels)
            pred_scores = marginals
        else:
            err_str = f"decode method {decode_method} not recognized"
            raise AssertionError(err_str)

        if return_scores:
            return preds, pred_scores

        return preds

    def _viterbi(self, dist, arc_labels=None):
        argmax = dist.argmax.cpu()
        preds, _ = dist.struct.from_parts(argmax)
        if (preds == -1).any():
            fillSegments(preds, in_place=True)

        if arc_labels is not None:
            preds = self._viterbiArcs(preds, arc_labels)

        return preds, argmax

    def _viterbiArcs(self, pred_state_idxs, arc_labels):
        pred_state_idxs = pred_state_idxs[0]
        state_segs, _ = utils.computeSegments(pred_state_idxs.tolist())

        pred_arc_idxs = -torch.ones_like(pred_state_idxs)
        for next_state_idx, cur_state_idx in zip(state_segs[1:], state_segs[:-1]):
            pred_arc_idx = arc_labels[next_state_idx, cur_state_idx]
            is_cur_state = pred_state_idxs == cur_state_idx
            pred_arc_idxs[is_cur_state] = pred_arc_idx
        pred_arc_idxs = pred_arc_idxs[:-1]

        if (pred_arc_idxs < 0).any():
            err_str = f"Impossible transition decoded: {pred_arc_idxs}"
            raise AssertionError(err_str)

        return pred_arc_idxs

    @staticmethod
    def _dist_marginals(dist):
        struct = dist._struct(torch_struct.semirings.LogSemiring)
        v, _, marginals = _dp_standard_sm(
            struct, dist.log_potentials, dist.lengths, force_grad=True,
            return_marginals=True
        )
        # Get edge marginals by summing over duration
        edge_marginals = struct.semiring.sum(marginals.cpu(), dim=2)
        return edge_marginals

    def _marginalDecode(self, dist, arc_labels=None):
        marginals = self._dist_marginals(dist).cpu()

        if arc_labels is not None:
            marginals = self._arcMarginals(marginals, arc_labels)
            preds = marginals.argmax(dim=-1)

        return preds[0], marginals[0]

    def _arcMarginals(self, edge_probs, arc_labels):
        is_arc = arc_labels >= 0
        arcs = is_arc.nonzero()
        s_next = arcs[:, 0]
        s_cur = arcs[:, 1]

        num_arcs = arc_labels.max() + 1
        arc_probs = torch.zeros((edge_probs.shape[0], edge_probs.shape[1], num_arcs))
        arc_probs[:, :, arc_labels[s_next, s_cur]] = edge_probs[:, :, s_next, s_cur]

        return arc_probs

    def forward(self, obsv, dur_scores=None):
        """ Construct a table of semi-Markov (i.e. segmental) scores.

        These scores can be used to instantiate pytorch-struct's SemiMarkovCRF.

        Parameters
        ----------
        obsv : torch.Tensor of float, shape (batch_size, ..., num_samples)
            NOTE: This method doesn't handle inputs with batch size > 1 (yet).

        Returns
        -------
        log_potentials : torch.Tensor of float,
                shape (batch_size, num_samples, max_duration, num_classes, num_classes)
            [t, d, i, j] = score(x[t:t+d], s_j -> s_i)
        """

        batch_size = obsv.shape[0]
        num_samples = obsv.shape[-1]
        num_labels = self.transition_weights.shape[0]
        max_duration = self.max_duration
        if max_duration is None:
            max_duration = num_samples

        scores_shape = (batch_size, num_samples, max_duration, num_labels, num_labels)
        log_potentials = torch.full(scores_shape, -float("Inf"), device=obsv.device)

        duration_scores = self.scoreDurations(torch.arange(max_duration), dur_scores=dur_scores)

        # for seg_start_index in range(num_samples):
        for seg_end_index in range(num_samples - 1):
            for seg_duration in range(1, max_duration):
                # seg_end_index = seg_start_index + seg_duration - 1
                next_start_index = seg_end_index + 1
                seg_start_index = next_start_index - seg_duration
                if seg_start_index < 0:
                    continue
                # if next_start_index > num_samples:
                #     break

                segment = obsv[:, ..., seg_start_index:next_start_index]
                assert(segment.shape[-1] == seg_duration)

                scores = self.scoreSegment(segment)
                scores = scores + duration_scores[seg_duration]
                scores = scores + self.transition_weights
                if seg_start_index == 0:
                    # Add initial weight to each state s_cur --- this is the
                    # score we give to being in that state when the sequence
                    # begins.
                    scores += self.start_weights
                if next_start_index == num_samples:
                    # Add final weight to each state s_next --- this is the score
                    # we give to being in that state after the sequence ends.
                    scores += self.end_weights[:, None]

                log_potentials[:, seg_end_index, seg_duration, :, :] = scores
                # log_potentials[:, seg_start_index, seg_duration, :, :] = scores
        return log_potentials

    def scoreDurations(self, durations, dur_scores=None):
        return torch.zeros_like(durations, dtype=torch.float)

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
        scores = super().forward(segment).sum(dim=-1)
        # Arrange scores as a column vector so tensor broadcasting replicates
        # scores across columns---scores array needs to have shape matching
        # (cur segment) x (prev segment)
        scores = scores[:, :, None]
        return scores


class MarkovScorer(SemiMarkovScorer):
    def predict(
            self, log_potentials, use_standard_dp=False, decode_method='viterbi',
            arc_labels=None):
        """
        Parameters
        ----------
        log_potentials :
        return_edges : bool, optional
        use_standard_dp : bool, optional
            pytorch_struct uses a parallel-scan dynamic program to reduce the number
            of sequential semiring calls from `O(seq_len)` to `O(log seq_len)`.
            This increases the memory complexity of the algorithm, though. So if
            you are decoding on a big graph and you run out of memory, you can try
            using the standard DP.
        """

        dist = torch_struct.LinearChainCRF(log_potentials)
        if use_standard_dp:
            dist.struct._dp = _dp_standard_lc
            dist.struct._dp_backward = _dp_backward_lc

        if decode_method == 'viterbi':
            preds, argmax = self._viterbi(dist, arc_labels=arc_labels)
            # FIXME: don't sum along zero-dimension
            pred_scores = log_potentials.cpu()[argmax.byte()]
        elif decode_method == 'marginal':
            preds, marginals = self._marginalDecode(dist, arc_labels=arc_labels)
            pred_scores = marginals
        else:
            err_str = f"decode method {decode_method} not recognized"
            raise AssertionError(err_str)

        # pdb.set_trace()

        return preds, pred_scores

    def _viterbi(self, dist, arc_labels=None):
        argmax = dist.argmax.cpu()
        preds, _ = dist.struct.from_parts(argmax)

        if arc_labels is not None:
            preds = self._viterbiArcs(preds, arc_labels)

        return preds, argmax

    def _viterbiArcs(self, pred_state_idxs, arc_labels):
        pred_state_idxs = pred_state_idxs[0]

        pred_arc_idxs = -torch.ones_like(pred_state_idxs)
        for next_state_idx, cur_state_idx in zip(pred_state_idxs[1:], pred_state_idxs[:-1]):
            pred_arc_idx = arc_labels[next_state_idx, cur_state_idx]
            is_cur_state = pred_state_idxs == cur_state_idx
            pred_arc_idxs[is_cur_state] = pred_arc_idx
        pred_arc_idxs = pred_arc_idxs[:-1]

        if (pred_arc_idxs < 0).any():
            err_str = f"Impossible transition decoded: {pred_arc_idxs}"
            raise AssertionError(err_str)

        return pred_arc_idxs

    @staticmethod
    def _dist_marginals(dist):
        struct = dist._struct(torch_struct.semirings.LogSemiring)
        v, _, alpha = _dp_standard_lc(struct, dist.log_potentials, dist.lengths, force_grad=True)
        ret = _dp_backward_lc(struct, dist.log_potentials, dist.lengths, alpha)
        return ret

    def _marginalDecode(self, dist, arc_labels=None):
        marginals = self._dist_marginals(dist).cpu()

        if arc_labels is not None:
            marginals = self._arcMarginals(marginals, arc_labels)
            preds = marginals.argmax(dim=-1)

        return preds[0], marginals[0]

    def _arcMarginals(self, edge_probs, arc_labels):
        is_arc = arc_labels >= 0
        arcs = is_arc.nonzero()
        s_next = arcs[:, 0]
        s_cur = arcs[:, 1]

        num_arcs = arc_labels.max() + 1
        arc_probs = torch.zeros((edge_probs.shape[0], edge_probs.shape[1], num_arcs))
        arc_probs[:, :, arc_labels[s_next, s_cur]] = edge_probs[:, :, s_next, s_cur]

        return arc_probs

    def forward(self, obsv, dur_scores=None):
        """ Construct a table of Markov (i.e. linear-chain) scores.

        These scores can be used to instantiate pytorch-struct's LinearChainCRF.

        Parameters
        ----------
        obsv : torch.Tensor of float, shape (batch_size, ..., num_samples)
            NOTE: This method doesn't handle inputs with batch size > 1 (yet).

        Returns
        -------
        log_potentials : torch.Tensor of float,
                shape (batch_size, num_samples, num_classes, num_classes)
            [t, d, i, j] = score(x[t:t+d], s_j -> s_i)
        """

        batch_size = obsv.shape[0]
        num_samples = obsv.shape[-1]
        num_labels = self.transition_weights.shape[0]

        scores_shape = (batch_size, num_samples, num_labels, num_labels)
        log_potentials = torch.full(scores_shape, -float("Inf"), device=obsv.device)

        for sample_index in range(num_samples):
            next_sample_index = sample_index + 1
            if next_sample_index > num_samples:
                break

            segment = obsv[:, ..., sample_index:next_sample_index]

            scores = self.scoreSegment(segment)
            scores += self.transition_weights
            if sample_index == 0:
                # Add initial weight to each state s_cur --- this is the
                # score we give to being in that state when the sequence
                # begins.
                scores += self.start_weights
            if next_sample_index == num_samples:
                # Add final weight to each state s_next --- this is the score
                # we give to being in that state after the sequence ends.
                scores += self.end_weights[:, None]

            log_potentials[:, sample_index, :, :] = scores
        return log_potentials


class LeaSegmentalScorer(object):
    def __init__(
            self, *super_args, transition_probs=None, initial_probs=None, final_probs=None,
            max_segs=None, update_transition_params=False, **super_kwargs):
        """
        Parameters
        ----------
        transition_probs : torch.Tensor of float, shape (num_classes, num_classes), optional
            Element (i, j) corresponds to (cur segment == j, prev segment == i);
        initial_probs : torch.Tensor of float, shape (num_classes,), optional
        final_probs : torch.Tensor of float, shape (num_gestures,), optional
        update_transition_params : bool, optional
            If True, start, end, and transition weights will be updated during
            training.
        max_segs : int, optional
        *super_args, **super_kwargs : optional
            Remaining arguments are passed to super's init method.
        """

        super().__init__(*super_args, **super_kwargs)

        self.update_transition_params = update_transition_params
        self.max_segs = max_segs

        if transition_probs is not None:
            transition_weights = torch.tensor(transition_probs).float().log()
            self.transition_weights = torch.nn.Parameter(
                transition_weights, requires_grad=self.update_transition_params
            )

        if initial_probs is not None:
            start_weights = torch.tensor(initial_probs).float().log()
            self.start_weights = torch.nn.Parameter(
                start_weights, requires_grad=self.update_transition_params
            )

        if final_probs is not None:
            end_weights = torch.tensor(final_probs).float().log()
            self.end_weights = torch.nn.Parameter(
                end_weights, requires_grad=self.update_transition_params
            )

    def predict(self, data_scores):
        """
        Parameters
        ----------
        data_scores : torch.tensor of float, shape (batch_size, num_dims, seq_length)
        """

        preds = tuple(
            torch_lctm.segmental_inference(
                scores, self.max_segs,
                pw=self.transition_weights
            )
            for scores in data_scores.transpose(1, 2)
        )

        # FIXME: should be a batch of sequences
        return torch.stack(preds)

    def nllloss(self, data_scores, y, reduction='mean'):
        losses = tuple(
            -torch_lctm.log_prob_eccv(
                scores, y_i, self.max_segs,
                pw=self.transition_weights
            )
            for scores, y_i in zip(data_scores.transpose(1, 2), y)
        )

        losses = torch.stack(losses)
        if reduction == 'none':
            return losses
        if reduction == 'mean':
            return losses.mean()
        if reduction == 'sum':
            return losses.sum()
        raise ValueError()


# -=( DP HELPER FUNCTIONS )==--------------------------------------------------
def _viterbi_forward(self, log_potentials):
    """ Construct a table of semi-Markov (i.e. segmental) scores.

    These scores can be used to instantiate pytorch-struct's SemiMarkovCRF.

    Parameters
    ----------
    obsv : torch.Tensor of float, shape (batch_size, ..., num_samples)
        NOTE: This method doesn't handle inputs with batch size > 1 (yet).

    Returns
    -------
    log_potentials : torch.Tensor of float,
            shape (batch_size, num_samples, max_duration, num_classes, num_classes)
        [t, d, i, j] = score(x[t:t+d], s_j -> s_i)
    """

    batch_size, num_samples, max_duration, num_labels = log_potentials.shape[:-1]

    argmax = torch.full(log_potentials.shape, dtype=torch.byte)

    # prev_score = None
    for start_idx in range(num_samples):
        for duration_idx in range(max_duration):
            if (start_idx + duration_idx) > num_samples:
                break
            for state_idx in range(num_labels):
                log_potentials[:, start_idx, duration_idx, :, :]

    return argmax


def _dp_standard_lc(self, edge, lengths=None, force_grad=False):
    semiring = self.semiring
    ssize = semiring.size()
    edge, batch, N, C, lengths = self._check_potentials(edge, lengths)

    alpha = self._make_chart(N, (batch, C), edge, force_grad)
    edge_store = self._make_chart(N - 1, (batch, C, C), edge, force_grad)

    semiring.one_(alpha[0].data)

    for n in range(1, N):
        edge_store[n - 1][:] = semiring.times(
            alpha[n - 1].view(ssize, batch, 1, C),
            edge[:, :, n - 1].view(ssize, batch, C, C),
        )
        alpha[n][:] = semiring.sum(edge_store[n - 1])

    for n in range(1, N):
        edge_store[n - 1][:] = semiring.times(
            alpha[n - 1].view(ssize, batch, 1, C),
            edge[:, :, n - 1].view(ssize, batch, C, C),
        )
        alpha[n][:] = semiring.sum(edge_store[n - 1])

    ret = [alpha[lengths[i] - 1][:, i] for i in range(batch)]
    ret = torch.stack(ret, dim=1)
    v = semiring.sum(ret)
    return v, edge_store, alpha


def _dp_backward_lc(self, edge, lengths, alpha_in, v=None):
    def logdivide(a, b):
        return (a - b)

    semiring = self.semiring
    _, batch, N, C, lengths = self._check_potentials(edge, lengths)

    alpha = self._make_chart(N, (batch, C), edge, force_grad=False)
    edge_store = self._make_chart(N - 1, (batch, C, C), edge, force_grad=False)

    for n in range(N - 1, 0, -1):
        for b, l in enumerate(lengths):
            # alpha[l - 1][b].data.fill_(semiring.one_())
            semiring.one_(alpha[l - 1][b].data)

        edge_store[n - 1][:] = semiring.times(
            alpha[n].view(batch, C, 1), edge[:, n - 1]
        )
        alpha[n - 1][:] = semiring.sum(edge_store[n - 1], dim=-2)
    v = semiring.sum(
        torch.stack([alpha[0][i] for i, l in enumerate(lengths)]), dim=-1
    )
    edge_marginals = self._make_chart(
        1, (batch, N - 1, C, C), edge, force_grad=False
    )[0]

    for n in range(N - 1):
        edge_marginals[0][:, n] = logdivide(
            semiring.times(
                alpha_in[n].view(batch, 1, C),
                edge[:, n],
                alpha[n + 1].view(batch, C, 1),
            ),
            v.view(batch, 1, 1),
        )

    return edge_marginals[0]


def _dp_standard_sm(self, log_potentials, lengths=None, force_grad=False, return_marginals=False):
    """ Standard dynamic program for semi-markov CRFs.

    Copied from pytorch_struct.semimarkov.py.

    This algorithm has worse runtime complexity than pytorch_struct's builtin
    method _dp, but it has better memory complexity. Useful for decoding long
    sequences (i.e. video streams instead of sentences).

    Parameters
    ----------
    log_potentials : torch.tensor of float, shape (n_batch, n_samples, max_dur, n_class, n_class)

    Returns
    -------
    v :
    ??? : list(tensor)
        A list containing only one element --- the input, `log_potentials`.
    beta :
        Weight of all paths finishing at sample N, with label C.
    """
    semiring = self.semiring
    ssize = semiring.size()
    log_potentials, batch, N, K, C, lengths = self._check_potentials(log_potentials, lengths)
    log_potentials.requires_grad_(True)

    # Init
    #   alpha: All paths starting at N of len K
    #   beta:  All paths finishing at N with label C
    alpha = self._make_chart(1, (batch, N, K, C), log_potentials, force_grad)[0]
    beta = self._make_chart(N, (batch, C), log_potentials, force_grad)
    semiring.one_(beta[0].data)

    # Main
    for n in range(1, N):
        alpha[:, :, n - 1] = semiring.dot(
            beta[n - 1].view(ssize, batch, 1, 1, C),
            log_potentials[:, :, n - 1].view(ssize, batch, K, C, C),
        )

        t = max(n - K, -1)
        f1 = torch.arange(n - 1, t, -1)
        f2 = torch.arange(1, len(f1) + 1)
        beta[n][:] = semiring.sum(
            torch.stack([alpha[:, :, a, b] for a, b in zip(f1, f2)], dim=-1)
        )
    v = semiring.sum(
        torch.stack([beta[l - 1][:, i] for i, l in enumerate(lengths)], dim=1)
    )

    if return_marginals:
        def logdivide(a, b):
            return (a - b)
        edge_marginals = self._make_chart(
            1, (batch, N - 1, K, C, C), log_potentials, force_grad=False
        )[0]

        for n in range(N - 1):
            k_max = min(K, N - n)
            for k in range(k_max):
                edge_marginals[0][:, n, k] = logdivide(
                    semiring.times(
                        alpha[:, :, n, k].view(batch, 1, C),
                        log_potentials[:, :, n, k],
                        beta[n + k].view(batch, C, 1),
                    ),
                    v.view(batch, 1, 1),
                )
        return v, [log_potentials], edge_marginals[0]

    # pdb.set_trace()
    return v, [log_potentials], beta
