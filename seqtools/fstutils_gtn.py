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
            requires_grad=True, debug_output_dir=None):
        super().__init__()

        self._vocabulary = vocabulary
        self._vocab_size = len(vocabulary)

        if transition_weights is None:
            transition_weights = torch.zeros(self._vocab_size, self._vocab_size, dtype=torch.float)

        if initial_weights is None:
            initial_weights = torch.zeros(self._vocab_size, dtype=torch.float)

        if final_weights is None:
            final_weights = torch.zeros(self._vocab_size, dtype=torch.float)

        self._params = torch.nn.ParameterDict({
            'transition': torch.nn.Parameter(transition_weights, requires_grad=requires_grad),
            'initial': torch.nn.Parameter(initial_weights, requires_grad=requires_grad),
            'final': torch.nn.Parameter(final_weights, requires_grad=requires_grad)
        })

        self._arc_to_transition, self._transition_to_arc = makeTransitionVocabulary(
            transition_weights
        )
        self._arc_symbols = {
            arc_idx: ','.join([str(self._vocabulary[state_idx]) for state_idx in state_pair])
            for arc_idx, state_pair in enumerate(self._arc_to_transition)
        }

        self._seq_fst = self._makeSeqFst()

        self._debug_output_dir = debug_output_dir

        if self._debug_output_dir is not None:
            gtn.draw(
                self._seq_fst, os.path.join(self._debug_output_dir, 'seq.png'),
                isymbols=self._arc_symbols, osymbols=self._arc_symbols
            )

    def _makeSeqFst(self):
        transition_fst = fromTransitions(
            self._params['transition'], self._params['initial'], self._params['final'],
            transition_ids=self._transition_to_arc
        )
        return transition_fst

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

    def nllLoss(self, inputs, targets):
        raise NotImplementedError()

    def log_prob(self, inputs, targets):
        class GtnNllLoss(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs, targets):
                losses = self._log_prob(inputs, targets)
                return losses

            @staticmethod
            def backward(ctx, grad_output):
                return self._log_prob_gradient()

        return GtnNllLoss.apply(inputs, targets)

    def _log_prob(self, inputs, targets):
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

            denom_fst = gtn.compose(obs_fst, self._seq_fst)
            num_fst = gtn.compose(denom_fst, gt_fst)

            loss = gtn.subtract(
                gtn.forward_score(num_fst),
                gtn.forward_score(denom_fst)
            )

            losses[batch_index] = loss
            obs_fsts[batch_index] = obs_fst

        gtn.parallel_for(seq_loss, range(batch_size))

        losses = torch.tensor([lp.item() for lp in losses]).to(device)
        self.auxiliary_data = (losses, obs_fsts, arc_scores.shape)

        return losses

    def _log_prob_gradient(self):
        raise NotImplementedError()

    def predict(self, inputs):
        return self.argmax(inputs)

    def argmax(self, inputs):
        arc_scores = self.scores_to_arc(inputs)

        device = arc_scores.device
        arc_scores = arc_scores.cpu()

        batch_size, num_samples, num_classes = arc_scores.shape

        best_paths = [None] * batch_size

        def pred_seq(batch_index):
            obs_fst = linearFstFromArray(arc_scores[batch_index].reshape(num_samples, -1))
            denom_fst = gtn.compose(obs_fst, self._seq_fst)
            best_paths[batch_index] = gtn.viterbi_path(denom_fst)

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


class DurationCrf(LatticeCrf):
    def __init__(self, labels, num_states=2, transition_weights=None, self_weights=None):
        super().__init__()

        if transition_weights is None:
            transition_weights = [0.6] * num_states

        if self_weights is None:
            self_weights = [0.4] * num_states

        self._transition_weights = transition_weights
        self._self_weights = self_weights
        self._num_states = num_states
        self._labels = labels

    def _makeSeqFst(self):
        raise NotImplementedError()


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


# -=( CREATION AND CONVERSION )==----------------------------------------------
def linearFstFromArray(weights):
    # Create a linear FST
    seq_len, vocab_size = weights.shape
    fst = gtn.linear_graph(seq_len, vocab_size, calc_grad=weights.requires_grad)

    # Set FST weights
    data = weights.contiguous()
    fst.set_weights(data.data_ptr())

    return fst


def fromArray(weights, final_weight=None, calc_grad=True):
    """ Instantiate a state machine from an array of weights.

    Parameters
    ----------
    weights : array_like, shape (num_inputs, num_outputs)
        Needs to implement `.shape`, so it should be a numpy array or a torch
        tensor.
    final_weight : arc_types.AbstractSemiringWeight, optional
        Should have the same type as `arc_type`. Default is `arc_type.zero`

    Returns
    -------
    fst : fsm.FST
        The transducer's arcs have input labels corresponding to the state
        they left, and output labels corresponding to the state they entered.
    """
    if weights.ndim == 3:
        is_lattice = False
    elif weights.ndim == 2:
        is_lattice = True
    else:
        raise AssertionError(f"weights have unrecognized shape {weights.shape}")

    fst = gtn.Graph(calc_grad=calc_grad)

    if final_weight is None:
        final_weight = one

    init_state = fst.add_node(start=True)
    final_state = fst.add_node(accept=True)

    if is_lattice:
        prev_state = init_state
        for sample_index, row in enumerate(weights):
            cur_state = fst.add_node()
            for i, weight in enumerate(row):
                input_label_index = sample_index
                output_label_index = i
                if weight != zero:
                    fst.add_arc(
                        prev_state, cur_state,
                        input_label_index, output_label_index,
                        weight
                    )
            prev_state = cur_state
        if final_weight != zero:
            fst.add_arc(
                cur_state, final_state,
                gtn.epsilon, gtn.epsilon,
                final_weight
            )
    else:
        prev_state = init_state
        for sample_index, input_output in enumerate(weights):
            cur_state = fst.add_node()
            for i, outputs in enumerate(input_output):
                for j, weight in enumerate(outputs):
                    input_label_index = i
                    output_label_index = j
                    if weight != zero:
                        fst.add_arc(
                            prev_state, cur_state,
                            input_label_index, output_label_index,
                            weight
                        )
            prev_state = cur_state
        if final_weight != zero:
            fst.add_arc(
                cur_state, final_state,
                gtn.epsilon, gtn.epsilon,
                final_weight
            )

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

    return fst


def toArray(lattice):
    num_states = lattice.num_states()
    num_outputs = lattice.output_symbols().num_symbols()
    weights = np.full((num_states, num_outputs), float(zero))

    for state in lattice.states():
        for arc in lattice.arcs(state):
            weights[state, arc.olabel] = float(arc.weight)

    return weights, lattice.weight_type()


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

    return fst


def viterbi(fst):
    raise NotImplementedError()
