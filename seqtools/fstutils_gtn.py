import logging
import os

import numpy as np
import torch
import gtn


logger = logging.getLogger(__name__)


EPSILON = gtn.epsilon
BOS = EPSILON - 1
EOS = BOS - 1
EPSILON_STRING = 'ε'
BOS_STRING = '<bos>'
EOS_STRING = '<eos>'


def vocab_mapper(base=None):
    mapper = {EPSILON: EPSILON_STRING, BOS: BOS_STRING, EOS: EOS_STRING}
    if base is not None:
        mapper.update(base)
    return mapper


# Log-semiring constants
zero = -np.inf
one = 0


def __test_basic(base_dir=None, draw=False):
    if base_dir is None:
        base_dir = os.path.expanduser('~')

    vocabulary = ['a', 'b', 'c', 'd', 'e', 'f']
    vocab_size = len(vocabulary)
    transitions = np.random.randn(vocab_size, vocab_size)
    transition_vocab, transition_ids = makeTransitionVocabulary(transitions)

    seq = range(vocab_size)
    seq_transitions = [transition_ids[t] for t in toTransitionSeq(seq)]
    seq_fst = fromSequence(seq_transitions)

    num_samples = len(seq_transitions) - 2
    # samples_mapper = vocab_mapper({i: str(i) for i in range(num_samples)})
    seq_mapper = vocab_mapper({i: s for i, s in enumerate(vocabulary)})
    tx_mapper = vocab_mapper(
        {i: ','.join((seq_mapper[j] for j in t)) for i, t in enumerate(transition_vocab)}
    )

    tx_fst = fromTransitions(transitions, transition_ids=transition_ids)

    scores = torch.tensor(np.random.randn(num_samples, vocab_size ** 2))
    scores_fst = linearFstFromArray(scores)

    denom = gtn.compose(scores_fst, tx_fst)
    num = gtn.compose(denom, seq_fst)

    fsts = (scores_fst, tx_fst, seq_fst, denom, num)
    names = ('SCORES', 'TRANSITIONS', 'SEQUENCE', 'DENOMINATOR', 'NUMERATOR')

    if draw:
        for fst, name in zip(fsts, names):
            gtn.draw(
                fst, os.path.join(base_dir, f'TEST_FROM{name}.png'),
                isymbols=tx_mapper, osymbols=tx_mapper
            )

    return fsts, names


class LatticeCrf(torch.nn.Module):
    def __init__(
            self, vocabulary,
            transition_weights=None, initial_weights=None, final_weights=None,
            duration_weights=None,
            requires_grad=True, debug_output_dir=None):
        super().__init__()

        self._vocabulary = vocabulary
        self._vocab_size = len(vocabulary)
        self._vocab_symbols = vocab_mapper({i: symbol for i, symbol in enumerate(vocabulary)})

        if transition_weights is None:
            transition_weights = torch.zeros(self._vocab_size, self._vocab_size, dtype=torch.float)

        if initial_weights is None:
            initial_weights = torch.zeros(self._vocab_size, dtype=torch.float)

        if final_weights is None:
            final_weights = torch.zeros(self._vocab_size, dtype=torch.float)

        if duration_weights is None:
            duration_weights = torch.zeros(
                self._vocab_size, self._vocab_size, 1,
                dtype=torch.float
            )

        self._arc_to_transition, self._transition_to_arc = makeTransitionVocabulary(
            transition_weights
        )

        self._arc_symbols = vocab_mapper(
            {
                arc_idx: ','.join([self._vocab_symbols[state_idx] for state_idx in state_pair])
                for arc_idx, state_pair in enumerate(self._arc_to_transition)
            }
        )

        self._transition_fst = fromTransitions(
            transition_weights, init_weights=initial_weights, final_weights=final_weights,
            transition_ids=self._transition_to_arc
        )

        self._duration_fst = fromDurations(
            duration_weights, init_weights=None, final_weights=None,
            transition_ids=self._transition_to_arc
        )

        transition_params = torch.from_numpy(self._transition_fst.weights_to_numpy())
        duration_params = torch.from_numpy(self._duration_fst.weights_to_numpy())

        self._params = torch.nn.ParameterDict({
            'transition': torch.nn.Parameter(transition_params, requires_grad=requires_grad),
            'duration': torch.nn.Parameter(duration_params, requires_grad=requires_grad),
        })

    def labels_to_arc(self, labels):
        arc_labels = torch.stack([
            torch.tensor([
                self._transition_to_arc[t]
                for t in toTransitionSeq(label_seq.tolist())
            ])
            for label_seq in labels
        ])
        return arc_labels

    def scores_to_arc(self, scores):
        arc_scores = torch.stack([
            x.view(x.shape[0], -1)
            for x in scores
        ])
        return arc_scores

    def seq_fst(self, transition_params=None, duration_params=None):
        if transition_params is None:
            transition_params = self._params['transition']

        if duration_params is None:
            duration_params = self._params['duration']

        self._transition_fst.set_weights(transition_params.tolist())
        self._transition_fst.zero_grad()

        self._duration_fst.set_weights(duration_params.tolist())
        self._duration_fst.zero_grad()

        # seq_fst = gtn.compose(self._duration_fst, self._transition_fst)
        return self._duration_fst, self._transition_fst

    def nllLoss(self, inputs, targets):
        return -self.log_prob(inputs, targets)

    def log_prob(self, inputs, targets):
        class GtnNllLoss(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs, targets, w_tx, w_dur):
                losses = self._log_prob(inputs, targets, w_tx, w_dur)
                return losses

            @staticmethod
            def backward(ctx, grad_output):
                gradients = self._log_prob_gradient(grad_output)
                return gradients

        logprob = GtnNllLoss.apply(
            inputs, targets,
            self._params['transition'], self._params['duration']
        )
        return logprob

    def _log_prob(self, inputs, targets, transition_params, duration_params):
        seq_fsts = self.seq_fst(
            transition_params=transition_params,
            duration_params=duration_params
        )

        device = inputs.device
        arc_scores = self.scores_to_arc(inputs)
        arc_labels = self.labels_to_arc(targets)

        arc_scores = arc_scores.cpu()
        arc_labels = arc_labels.cpu()

        batch_size, num_samples, num_classes = arc_scores.shape

        losses = [None] * batch_size
        obs_fsts = [None] * batch_size

        def seq_loss(batch_index):
            obs_fst = linearFstFromArray(arc_scores[batch_index].reshape(num_samples, -1))
            gt_fst = fromSequence(arc_labels[batch_index])

            # Compose each sequence fst individually: it seems like composition
            # only works for lattices
            denom_fst = obs_fst
            for seq_fst in seq_fsts:
                denom_fst = gtn.compose(denom_fst, seq_fst)
                denom_fst = gtn.project_output(denom_fst)

            num_fst = gtn.compose(denom_fst, gt_fst)

            loss = gtn.subtract(gtn.forward_score(num_fst), gtn.forward_score(denom_fst))

            losses[batch_index] = loss
            obs_fsts[batch_index] = obs_fst

        gtn.parallel_for(seq_loss, range(batch_size))

        self.auxiliary_data = losses

        losses = torch.tensor([lp.item() for lp in losses]).to(device)

        return losses

    def _log_prob_gradient(self, grad_output):
        """
        Parameters
        ----------
        grad_output

        Returns
        -------
        input_grad
        target_grad
        transition_grad
        """

        losses = self.auxiliary_data

        # Compute the gradients for each example:
        def seq_grad(b):
            gtn.backward(losses[b])

        # Compute gradients in parallel over the batch:
        gtn.parallel_for(seq_grad, range(len(losses)))

        transition_grad = self._transition_fst.grad().weights_to_numpy()
        duration_grad = self._duration_fst.grad().weights_to_numpy()

        input_grad = None
        target_grad = None
        transition_grad = torch.from_numpy(transition_grad) * grad_output.cpu()
        duration_grad = torch.from_numpy(duration_grad) * grad_output.cpu()

        return input_grad, target_grad, transition_grad, duration_grad

    def predict(self, inputs):
        return self.argmax(inputs)

    def argmax(self, inputs):
        seq_fsts = self.seq_fst()

        arc_scores = self.scores_to_arc(inputs)

        device = arc_scores.device
        arc_scores = arc_scores.cpu()

        batch_size, num_samples, num_classes = arc_scores.shape

        best_paths = [None] * batch_size

        def pred_seq(batch_index):
            obs_fst = linearFstFromArray(arc_scores[batch_index].reshape(num_samples, -1))

            # Compose each sequence fst individually: it seems like composition
            # only works for lattices
            denom_fst = obs_fst
            for seq_fst in seq_fsts:
                denom_fst = gtn.compose(denom_fst, seq_fst)

            viterbi_path = gtn.viterbi_path(denom_fst)
            best_paths[batch_index] = gtn.remove(gtn.project_output(viterbi_path))

        gtn.parallel_for(pred_seq, range(batch_size))

        best_paths = torch.tensor([self._getOutputString(p) for p in best_paths]).to(device)
        return best_paths

    def _getOutputString(self, path):
        arc_labels = path.labels_to_list(ilabel=False)
        transition_seq = tuple(self._arc_to_transition[a] for a in arc_labels)
        seq = toStateSeq(transition_seq)
        return seq

    def forward(self, inputs):
        outputs = inputs
        return outputs


# -=( MISC UTILS )==-----------------------------------------------------------
def toTransitionSeq(state_seq):
    transition_seq = (
        ((BOS, state_seq[0]),)
        + tuple(zip(state_seq[:-1], state_seq[1:]))
        + ((state_seq[-1], EOS),)
    )
    return transition_seq


def toStateSeq(transition_seq):
    state_seq = tuple(transition[1] for transition in transition_seq[:-1])
    return state_seq


def makeTransitionVocabulary(transition_weights):
    num_states = transition_weights.shape[0]

    transition_vocab = []
    for s_cur in range(num_states):
        for s_next in range(num_states):
            transition_vocab.append((s_cur, s_next))
    for s in range(num_states):
        transition_vocab.append((BOS, s))
    for s in range(num_states):
        transition_vocab.append((s, EOS))

    transition_ids = {t: i for i, t in enumerate(transition_vocab)}

    return transition_vocab, transition_ids


# -=( FST CREATION )==---------------------------------------------------------
def linearFstFromArray(weights):
    seq_len, vocab_size = weights.shape
    fst = gtn.linear_graph(seq_len, vocab_size, calc_grad=weights.requires_grad)

    # Set FST weights
    data = weights.contiguous()
    fst.set_weights(data.data_ptr())

    fst.arc_sort(olabel=True)
    return fst


def fromSequence(seq, is_acceptor=True, calc_grad=True):
    if not is_acceptor:
        raise AssertionError()

    fst = gtn.Graph(calc_grad=calc_grad)

    init_state = fst.add_node(start=True)
    final_state = fst.add_node(accept=True)

    prev_state = init_state
    for sample_index, label_index in enumerate(seq):
        cur_state = fst.add_node()
        fst.add_arc(prev_state, cur_state, label_index)
        prev_state = cur_state
    fst.add_arc(cur_state, final_state, gtn.epsilon)

    fst.arc_sort(olabel=True)
    return fst


def fromDurations(
        duration_weights, init_weights=None, final_weights=None,
        transition_ids=None, calc_grad=True):
    if init_weights is None:
        init_weights = np.full(duration_weights.shape[:-1], one)

    if final_weights is None:
        final_weights = np.full(duration_weights.shape[:-1], one)

    fst = gtn.Graph(calc_grad=calc_grad)
    init_state = fst.add_node(start=True)
    final_state = fst.add_node(accept=True)
    fst.add_arc(final_state, init_state, gtn.epsilon, gtn.epsilon, one)

    def makeStateDuration(i_cur, i_next):
        weights = duration_weights[i_cur, i_next]
        transition_id = transition_ids[i_cur, i_next]
        init_weight = init_weights[i_cur, i_next]
        final_weight = final_weights[i_cur, i_next]

        cur_state = fst.add_node()
        fst.add_arc(init_state, cur_state, gtn.epsilon, transition_id, init_weight)

        for weight in weights:
            next_state = fst.add_node()
            fst.add_arc(cur_state, next_state, transition_id, gtn.epsilon, weight)
            cur_state = next_state
        fst.add_arc(cur_state, final_state, gtn.epsilon, gtn.epsilon, final_weight)

    num_states = duration_weights.shape[0]
    for i_cur in range(num_states):
        for i_next in range(num_states):
            makeStateDuration(i_cur, i_next)

    fst.arc_sort(olabel=True)
    return fst


def fromTransitions(
        transition_weights, init_weights=None, final_weights=None,
        transition_ids=None, calc_grad=True):
    """ Instantiate a state machine from state transitions.

    Parameters
    ----------

    Returns
    -------
    """

    num_states = transition_weights.shape[0]

    if transition_ids is None:
        _, transition_ids = makeTransitionVocabulary(transition_weights)

    if init_weights is None:
        init_weights = tuple(float(one) for __ in range(num_states))

    if final_weights is None:
        final_weights = tuple(float(one) for __ in range(num_states))

    fst = gtn.Graph(calc_grad=calc_grad)
    init_state = fst.add_node(start=True)
    final_state = fst.add_node(accept=True)

    def makeState(i):
        state = fst.add_node()

        transition_id = transition_ids[BOS, i]
        initial_weight = init_weights[i]
        if initial_weight != zero:
            fst.add_arc(init_state, state, gtn.epsilon, transition_id, initial_weight)

        transition_id = transition_ids[i, EOS]
        final_weight = final_weights[i]
        if final_weight != zero:
            fst.add_arc(state, final_state, gtn.epsilon, transition_id, final_weight)

        return state

    states = tuple(makeState(i) for i in range(num_states))
    for i_cur, row in enumerate(transition_weights):
        for i_next, weight in enumerate(row):
            cur_state = states[i_cur]
            next_state = states[i_next]
            transition_id = transition_ids[i_cur, i_next]
            if weight != zero:
                fst.add_arc(cur_state, next_state, transition_id, transition_id, weight)

    fst.arc_sort(olabel=True)
    return fst
