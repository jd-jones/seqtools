import logging

import numpy as np
import gtn


logger = logging.getLogger(__name__)


EPSILON = gtn.epsilon
EPSILON_STRING = 'ε'

zero = -np.inf
one = 0


def __test(num_samples=5):
    def sampleGT(transition_probs, initial_probs):
        cur_state = np.random.choice(initial_probs.shape[0], p=initial_probs)
        gt_seq = [cur_state]
        while True:
            transitions = transition_probs[cur_state, :]
            cur_state = np.random.choice(transitions.shape[0], p=transitions)
            if cur_state == transitions.shape[0] - 1:
                return gt_seq
            gt_seq.append(cur_state)

    def sampleScores(gt_seq, num_states):
        """ score[i, j, k] := weight(sample i | state j -> state k) """
        num_samples = len(gt_seq) - 1
        scores = np.random.random_sample(size=(num_samples, num_states, num_states))
        return scores

    def samplePair(transition_probs, initial_probs):
        gt_seq = sampleGT(transition_probs, initial_probs)
        score_seq = sampleScores(gt_seq, initial_probs.shape[0])
        return gt_seq, score_seq

    def simulate(num_samples, transition, initial, final):
        transition_probs = np.hstack((transition, final[:, None]))
        transition_probs /= transition_probs.sum(axis=1)[:, None]
        initial_probs = initial.copy()
        initial_probs /= initial_probs.sum()

        simulated_dataset = tuple(
            samplePair(transition_probs, initial_probs)
            for __ in range(num_samples)
        )
        return simulated_dataset

    transition = np.array(
        [[0, 1, 0, 0, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 0, 0, 1],
         [0, 1, 0, 0, 1],
         [0, 0, 0, 0, 0]], dtype=float
    )
    initial = np.array([1, 0, 1, 0, 0], dtype=float)
    final = np.array([0, 1, 0, 0, 1], dtype=float) / 10

    num_states = len(initial)
    transition_to_arc = {}
    for s_cur in range(num_states):
        for s_next in range(num_states):
            transition_to_arc[(s_cur, s_next)] = len(transition_to_arc)
    for s in range(num_states):
        transition_to_arc[(-1, s)] = len(transition_to_arc)
    arc_to_transition = {v: k for k, v in transition_to_arc.items()}

    seq_params = (transition, initial, final)
    simulated_dataset = simulate(num_samples, *seq_params)
    seq_params = tuple(map(lambda x: -np.log(x), seq_params))

    return seq_params, simulated_dataset, arc_to_transition, transition_to_arc


class LatticeCrf(object):
    def __init__(self, transition_weights=None, initial_weights=None, final_weights=None):
        super().__init__()

        self._params = {
            'transition': transition_weights,
            'initial': initial_weights,
            'final': final_weights
        }

        num_states = self._params['initial'].shape[0]
        self._transition_to_arc = {}
        for s_cur in range(num_states):
            for s_next in range(num_states):
                self._transition_to_arc[(s_cur, s_next)] = len(self._transition_to_arc)
        for s in range(num_states):
            self._transition_to_arc[(-1, s)] = len(self._transition_to_arc)
        self._arc_to_transition = {v: k for k, v in self._transition_to_arc.items()}

    def fit(self, train_samples, train_labels, observation_scores=None, num_epochs=1):
        if observation_scores is None:
            observation_scores = self.score(train_samples)

        obs_fsts = tuple(
            fromArray(scores.reshape(scores.shape[0], -1))
            for scores in observation_scores
        )

        train_labels = tuple(
            [self._transition_to_arc[t] for t in toTransitionSeq(label)]
            for label in train_labels
        )
        gt_fsts = tuple(
            fromSequence(labels)
            for labels in train_labels
        )

        losses, params = self._fit(obs_fsts, gt_fsts)

        return losses, params

    def _fit(self, obs_fsts, gt_fsts):
        raise NotImplementedError()

    def _makeSeqFst(self):
        transition_fst = fromTransitions(
            self._params['transition'], self._params['initial'], self._params['final'],
            transition_ids=self._transition_to_arc
        )
        return transition_fst

    def predict(self, test_samples, observation_scores=None):
        if observation_scores is None:
            observation_scores = self.score(test_samples)

        transition_fst = fromArray(self._transition_weights)
        observation_fsts = tuple(
            fromArray(scores.reshape(scores.shape[0], -1))
            for scores in observation_scores
        )

        all_preds = []
        for observation_fst in observation_fsts:
            decode_graph = gtn.compose(observation_fst, transition_fst)
            pred_labels = viterbi(decode_graph)
            all_preds.append(pred_labels)

        return all_preds

    def score(self, train_samples):
        raise NotImplementedError()


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
    transition_seq = ((-1, state_seq[0]),) + tuple(zip(state_seq[:-1], state_seq[1:]))
    return transition_seq


def toStateSeq(transition_seq):
    state_seq = tuple(transition[1] for transition in transition_seq)
    return state_seq


def iteratePaths(fst):
    paths = [tuple(fst.arcs(fst.start()))]
    while paths:
        path = paths.pop()
        state = path[-1].nextstate
        if state == -1:
            yield path

        for arc in fst.arcs(state):
            new_path = path + (arc,)
            paths.append(new_path)


def outputLabels(fst):
    for path in iteratePaths(fst):
        yield tuple(arc.olabel for arc in path)


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
            cur_state = fst.add_state()
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
            cur_state = fst.add_state()
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
        cur_state = fst.add_state()
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
        transition_ids = {}
        for s_cur in range(num_states):
            for s_next in range(num_states):
                transition_ids[(s_cur, s_next)] = len(transition_ids)
        for s in range(num_states):
            transition_ids[(-1, s)] = len(transition_ids)

    if init_weights is None:
        init_weights = tuple(float(one) for __ in range(num_states))

    if final_weights is None:
        final_weights = tuple(float(one) for __ in range(num_states))

    fst = gtn.Graph(calc_grad=calc_grad)
    init_state = fst.add_node(start=True)
    final_state = fst.add_node(accept=True)

    def makeState(i):
        transition_id = transition_ids[-1, i]
        state = fst.add_state()

        initial_weight = init_weights[i]
        if initial_weight != zero:
            fst.add_arc(init_state, state, gtn.epsilon, transition_id, initial_weight)

        final_weight = final_weights[i]
        if final_weight != zero:
            fst.add_arc(state, final_state, gtn.epsilon, gtn.epsilon, final_weight)

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
