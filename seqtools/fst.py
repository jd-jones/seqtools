import logging

import numpy as np
import pywrapfst as openfst


logger = logging.getLogger(__name__)


EPSILON = 0


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
    def __init__(
            self, transition_weights=None, initial_weights=None, final_weights=None,
            update_method='simple'):
        self._transition_weights = transition_weights
        self._initial_weights = initial_weights
        self._final_weights = final_weights

        num_states = self._initial_weights.shape[0]
        self._transition_to_arc = {}
        for s_cur in range(num_states):
            for s_next in range(num_states):
                self._transition_to_arc[(s_cur, s_next)] = len(self._transition_to_arc)
        for s in range(num_states):
            self._transition_to_arc[(-1, s)] = len(self._transition_to_arc)
        self._arc_to_transition = {v: k for k, v in self._transition_to_arc.items()}

        if update_method == 'simple':
            self._updateWeights = gradientStep
        else:
            raise NotImplementedError()

    def fit(self, train_samples, train_labels, observation_scores=None, num_epochs=1):
        if observation_scores is None:
            observation_scores = self.score(train_samples)

        train_labels = tuple(
            [self._transition_to_arc[t] for t in toTransitionSeq(label)]
            for label in train_labels
        )
        observation_fsts = tuple(
            fromArray(scores.reshape(scores.shape[0], -1), output_labels=self._transition_to_arc)
            for scores in observation_scores
        )
        gt_fsts = tuple(
            fromSequence(labels, symbol_table=o_fst.output_symbols())
            for labels, o_fst in zip(train_labels, observation_fsts)
        )

        losses = []
        for i in range(num_epochs):
            transition_fst = fromTransitions(
                self._transition_weights, self._initial_weights, self._final_weights,
                transition_ids=self._transition_to_arc
            )
            batch_loss, batch_grad = batchLoss(
                gt_fsts, observation_fsts, transition_fst, self._arc_to_transition
            )

            transition_grad, init_grad, final_grad = batch_grad
            self._transition_weights = self._updateWeights(
                self._transition_weights, transition_grad
            )
            self._initial_weights = self._updateWeights(self._initial_weights, init_grad)
            self._final_weights = self._updateWeights(self._final_weights, final_grad)
            losses.append(batch_loss)

        return np.array(losses)

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
            decode_graph = openfst.compose(observation_fst, transition_fst)
            pred_labels = viterbi(decode_graph)
            all_preds.append(pred_labels)

        return all_preds

    def score(self, train_samples):
        raise NotImplementedError()


# -=( MISC UTILS )==-----------------------------------------------------------
def toTransitionSeq(state_seq):
    transition_seq = ((-1, state_seq[0]),) + tuple(zip(state_seq[:-1], state_seq[1:]))
    return transition_seq


def toStateSeq(transition_seq):
    state_seq = tuple(transition[1] for transition in transition_seq)
    return state_seq


def isLattice(fst):
    for state in fst.states():
        input_labels = tuple(arc.ilabel for arc in fst.arcs(state))
        input_label = input_labels[0]
        if not all(i == input_label for i in input_labels):
            return False

    # TODO
    return True


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
def fromArray(weights, final_weight=None, arc_type=None, input_labels=None, output_labels=None):
    """ Instantiate a state machine from an array of weights.

    Parameters
    ----------
    weights : array_like, shape (num_inputs, num_outputs)
        Needs to implement `.shape`, so it should be a numpy array or a torch
        tensor.
    final_weight : arc_types.AbstractSemiringWeight, optional
        Should have the same type as `arc_type`. Default is `arc_type.zero`
    arc_type : {'standard', 'log'}, optional
        Default is 'standard' (ie the tropical arc_type)
    input_labels :
    output_labels :

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

    if arc_type is None:
        arc_type = 'standard'

    if output_labels is None:
        output_labels = {str(i): i for i in range(weights.shape[1])}

    if input_labels is None:
        if is_lattice:
            input_labels = {str(i): i for i in range(weights.shape[0])}
        else:
            input_labels = {str(i): i for i in range(weights.shape[2])}

    input_table = openfst.SymbolTable()
    for in_symbol, index in input_labels.items():
        input_table.add_symbol(str(in_symbol), key=index + 1)

    output_table = openfst.SymbolTable()
    for out_symbol, index in output_labels.items():
        output_table.add_symbol(str(out_symbol), key=index + 1)

    fst = openfst.VectorFst(arc_type=arc_type)
    fst.set_input_symbols(input_table)
    fst.set_output_symbols(output_table)

    zero = openfst.Weight.zero(fst.weight_type())
    one = openfst.Weight.one(fst.weight_type())

    if final_weight is None:
        final_weight = one
    else:
        final_weight = openfst.Weight(fst.weight_type(), final_weight)

    init_state = fst.add_state()
    fst.set_start(init_state)

    if is_lattice:
        prev_state = init_state
        for sample_index, row in enumerate(weights):
            cur_state = fst.add_state()
            for i, weight in enumerate(row):
                input_label_index = sample_index + 1
                output_label_index = i + 1
                weight = openfst.Weight(fst.weight_type(), weight)
                if weight != zero:
                    arc = openfst.Arc(
                        input_label_index, output_label_index,
                        weight, cur_state
                    )
                    fst.add_arc(prev_state, arc)
            prev_state = cur_state
        fst.set_final(cur_state, final_weight)
    else:
        prev_state = init_state
        for sample_index, input_output in enumerate(weights):
            cur_state = fst.add_state()
            for i, outputs in enumerate(input_output):
                for j, weight in enumerate(outputs):
                    input_label_index = i + 1
                    output_label_index = j + 1
                    weight = openfst.Weight(fst.weight_type(), weight)
                    if weight != zero:
                        arc = openfst.Arc(
                            input_label_index, output_label_index,
                            weight, cur_state
                        )
                        fst.add_arc(prev_state, arc)
            prev_state = cur_state
        fst.set_final(cur_state, final_weight)

    return fst


def toArray(lattice):
    zero = openfst.Weight.zero(lattice.weight_type())
    # one = openfst.Weight.one(lattice.weight_type())

    num_states = lattice.num_states()
    num_outputs = lattice.output_symbols().num_symbols()
    weights = np.full((num_states, num_outputs), float(zero))

    for state in lattice.states():
        for arc in lattice.arcs(state):
            weights[state, arc.olabel] = float(arc.weight)

    return weights, lattice.weight_type()


def fromSequence(seq, arc_type='standard', symbol_table=None):
    fst = openfst.VectorFst(arc_type=arc_type)
    one = openfst.Weight.one(fst.weight_type())

    if symbol_table is not None:
        fst.set_input_symbols(symbol_table)
        fst.set_output_symbols(symbol_table)

    init_state = fst.add_state()
    fst.set_start(init_state)

    cur_state = init_state
    for i, label in enumerate(seq):
        next_state = fst.add_state()
        arc = openfst.Arc(label + 1, label + 1, one, next_state)
        fst.add_arc(cur_state, arc)
        cur_state = next_state

    fst.set_final(cur_state, one)

    return fst


def fromTransitions(
        transition_weights, init_weights=None, final_weights=None,
        arc_type='standard', transition_ids=None):
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

    output_table = openfst.SymbolTable()
    for transition, index in transition_ids.items():
        output_table.add_symbol(str(transition), key=index + 1)

    input_table = openfst.SymbolTable()
    for transition, index in transition_ids.items():
        input_table.add_symbol(str(transition), key=index + 1)

    fst = openfst.VectorFst(arc_type=arc_type)
    fst.set_input_symbols(input_table)
    fst.set_output_symbols(output_table)

    zero = openfst.Weight.zero(fst.weight_type())
    one = openfst.Weight.one(fst.weight_type())

    if init_weights is None:
        init_weights = tuple(float(one) for __ in range(num_states))

    if final_weights is None:
        final_weights = tuple(float(one) for __ in range(num_states))

    fst.set_start(fst.add_state())

    def makeState(i):
        state = fst.add_state()

        initial_weight = openfst.Weight(fst.weight_type(), init_weights[i])
        if initial_weight != zero:
            transition = transition_ids[-1, i] + 1
            arc = openfst.Arc(EPSILON, transition, initial_weight, state)
            fst.add_arc(fst.start(), arc)

        final_weight = openfst.Weight(fst.weight_type(), final_weights[i])
        if final_weight != zero:
            fst.set_final(state, final_weight)

        return state

    states = tuple(makeState(i) for i in range(num_states))
    for i_cur, row in enumerate(transition_weights):
        for i_next, tx_weight in enumerate(row):
            cur_state = states[i_cur]
            next_state = states[i_next]
            weight = openfst.Weight(fst.weight_type(), tx_weight)
            transition = transition_ids[i_cur, i_next] + 1
            if weight != zero:
                arc = openfst.Arc(transition, transition, weight, next_state)
                fst.add_arc(cur_state, arc)

    return fst


def leftToRightAcceptor(input_seq, semiring=None, string_mapper=None):
    """ Construct a left-to-right finite-state acceptor from an input sequence.

    The input is usually a sequence of segment-level labels, and this machine
    is used to align labels with sample-level scores.

    Parameters
    ----------
    input_seq : iterable(int or string), optional
    semiring : semirings.AbstractSemiring, optional
        Default is semirings.BooleanSemiringWeight.
    string_mapper : function, optional
        A function that takes an integer as input and returns a string as output.

    Returns
    -------
    acceptor : fsm.FST
        A linear-chain finite-state acceptor with `num_states` states. Each state
        has one self-transition and one transition to its right neighbor. i.e.
        the topology looks like this (self-loops are omitted in the diagram
        below because they're hard to draw in ASCII style):

            [START] s1 --> s2 --> s3 [END]

        All edge weights are `semiring.one`.
    """
    raise NotImplementedError()


# -=( TRAINING )==-------------------------------------------------------------
def batchLoss(gt_fsts, observation_fsts, transition_fst, arc_to_transition):
    batch_loss = 0
    batch_gradient = None
    for observation_fst, gt_fst in zip(observation_fsts, gt_fsts):
        neg_log_prob, gradient = fstLogProb(
            observation_fst, gt_fst, transition_fst, arc_to_transition
        )

        batch_loss += neg_log_prob
        if batch_gradient is None:
            batch_gradient = gradient
        else:
            batch_gradient = tuple(cur + new for cur, new in zip(batch_gradient, gradient))

    return batch_loss, batch_gradient


def gradientStep(weights, gradient, step_size=1e-3):
    new_weights = weights + step_size * gradient
    return new_weights


# -=( CRF/HMM ALGORITHMS )==---------------------------------------------------
def fstLogProb(observation_fst, gt_fst, transition_fst, arc_to_transition):
    denominator_graph = openfst.compose(observation_fst, transition_fst)
    denom_betas = backward(denominator_graph, neglog_to_log=True)
    denom_weight = denom_betas[denominator_graph.start()]

    numerator_graph = openfst.compose(denominator_graph, gt_fst)
    num_betas = backward(numerator_graph, neglog_to_log=True)
    num_weight = num_betas[numerator_graph.start()]

    # LOSS FUNCTION
    log_prob = -float(openfst.divide(num_weight, denom_weight))

    # LOSS GRADIENT
    num_arcgrad = fstArcGradient(numerator_graph, betas=num_betas)
    denom_arcgrad = fstArcGradient(denominator_graph, betas=denom_betas)
    num_seqgrad = seqGradient(num_arcgrad, arc_to_transition)
    denom_seqgrad = seqGradient(denom_arcgrad, arc_to_transition)
    seq_gradient = tuple(grad_n - grad_d for grad_n, grad_d in zip(num_seqgrad, denom_seqgrad))

    return log_prob, seq_gradient


def seqGradient(arc_gradient, arc_to_transition):
    """ TODO: Generalize this so it maps arcs to their partial gradients """

    if arc_gradient.arc_type() != 'log':
        raise AssertionError()

    zero = openfst.Weight.zero(arc_gradient.weight_type())

    num_states = max(v[1] for v in arc_to_transition.values()) + 1
    transition_grad = np.zeros((num_states, num_states))
    initial_grad = np.zeros(num_states)
    final_grad = np.zeros(num_states)

    if arc_gradient.final(arc_gradient.start()) != zero:
        raise AssertionError("Lattice start states should never be final states!")

    for state in arc_gradient.states():
        for arc in arc_gradient.arcs(state):
            prev_out, cur_out = arc_to_transition[arc.olabel - 1]
            # Arc weights are negative log-probs, but we need probs for the gradient
            arc_prob = np.exp(-float(arc.weight))
            if state == arc_gradient.start():
                initial_grad[cur_out] += arc_prob
            else:
                transition_grad[prev_out, cur_out] += arc_prob

            next_final_weight = arc_gradient.final(arc.nextstate)
            if next_final_weight != zero:
                # Arc weights are negative log-probs, but we need probs for the gradient
                final_prob = np.exp(-float(next_final_weight))
                final_grad[cur_out] += final_prob

    return transition_grad, initial_grad, final_grad


def fstArcGradient(lattice, alphas=None, betas=None):
    if lattice.arc_type() != 'log':
        lattice = openfst.arcmap(lattice, map_type='to_log')

    if alphas is None:
        alphas = forward(lattice, neglog_to_log=True)

    if betas is None:
        betas = backward(lattice, neglog_to_log=True)

    total_weight = betas[lattice.start()]
    zero = openfst.Weight.zero(lattice.weight_type())

    arc_gradient = lattice.copy()
    for state in arc_gradient.states():
        w_incoming = alphas[state]
        arc_iterator = arc_gradient.mutable_arcs(state)
        while not arc_iterator.done():
            arc = arc_iterator.value()

            w_outgoing = betas[arc.nextstate]
            weight_thru_arc = openfst.times(w_incoming, w_outgoing)
            arc_neglogprob = openfst.divide(total_weight, weight_thru_arc)

            arc.weight = arc_neglogprob
            arc_iterator.set_value(arc)
            arc_iterator.next()

        if lattice.final(state) != zero:
            # w_outgoing = one --> final weight = w_in \otimes one = w_in
            weight_thru_arc = alphas[state]
            arc_neglogprob = openfst.divide(total_weight, weight_thru_arc)
            arc_gradient.set_final(state, arc_neglogprob)

    return arc_gradient


def forward(lattice, neglog_to_log=False):
    if lattice.arc_type() != 'log':
        lattice = openfst.arcmap(lattice, map_type='to_log')

    if neglog_to_log:
        inverted = openfst.arcmap(lattice, map_type='invert')
        one = openfst.Weight.one(lattice.weight_type())
        alphas = [openfst.divide(one, a) for a in forward(inverted)]
        return alphas

    alphas = openfst.shortestdistance(lattice)
    return alphas


def backward(lattice, neglog_to_log=False):
    if lattice.arc_type() != 'log':
        lattice = openfst.arcmap(lattice, map_type='to_log')

    if neglog_to_log:
        inverted = openfst.arcmap(lattice, map_type='invert')
        one = openfst.Weight.one(lattice.weight_type())
        betas = [openfst.divide(one, a) for a in backward(inverted)]
        return betas

    betas = openfst.shortestdistance(lattice, reverse=True)
    return betas


def viterbi(lattice):
    if lattice.arc_type() != 'standard':
        lattice = openfst.arcmap(lattice, map_type='to_std')

    shortest_paths = openfst.shortestpath(lattice)

    path_outputs = outputLabels(shortest_paths)
    if len(path_outputs) != 1:
        raise AssertionError()
    output = path_outputs[0]
    return output


def align(scores, label_seq):
    """ Align (ie segment) a sequence of scores, given a known label sequence.

    NOTE: I don't know if this works with score tables that have more than 9
        columns.

    Parameters
    ----------
    scores : array_like of float, shape (num_samples, num_labels)
        Log probabilities (possibly un-normalized).
    label_seq : iterable(string or int)
        The segment-level label sequence.

    Returns
    -------
    aligned_labels : tuple(int)
    alignment_score : semirings.TropicalSemiringWeight
        Score of the best path through the alignment graph (possible un-normalized)
    """
    raise NotImplementedError()
