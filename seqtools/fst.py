import collections
import logging
import itertools

import numpy as np
import pywrapfst as openfst

from mathtools import utils


logger = logging.getLogger(__name__)


class FstIntegerizer(object):
    def __init__(self, iterable=[], prepend_epsilon=True):
        # OpenFST hardcodes zero to represent epsilon transitions---so make sure
        # our integerizer is consistent with that.
        if prepend_epsilon:
            iterable = type(iterable)(['epsilon']) + iterable

        super().__init__(iterable)

    def updateFromSequences(self, sequences):
        self.update(itertools.chain(*sequences))

    def integerizeSequence(self, sequence):
        return tuple(self.index(x) for x in sequence)

    def deintegerizeSequence(self, index_sequence):
        return tuple(self[i] for i in index_sequence)


class HashableFstIntegerizer(FstIntegerizer, utils.Integerizer):
    pass


class UnhashableFstIntegerizer(FstIntegerizer, utils.UnhashableIntegerizer):
    pass


def Lattice(object):
    def __init__(self, transition_weights=None, final_weights=None, update_method='gradient'):
        transition_weights = transition_weights[None, ...]
        transition_weights = transition_weights.reshape(transition_weights.shape[0], -1)

        self._transition_weights = transition_weights
        self._final_weights = final_weights
        if update_method == 'gradient':
            self._updateWeights = gradientStep

    def fit(self, train_samples, train_labels, observation_scores=None, num_epochs=1):
        if observation_scores is None:
            observation_scores = self.score(train_samples)

        observation_scores = observation_scores.reshape(transition_weights.shape[0], -1)

        losses = []
        for i in num_epochs:
            batch_loss, batch_grad = batchLoss(
                train_labels, observation_scores, self._transition_weights, self._final_weights
            )

            self._transition_weights = self._updateWeights(self._transition_weights, batch_grad)
            losses.append(batch_loss)
        return np.array(losses)

    def predict(self, test_samples, observation_scores=None):
        if observation_scores is None:
            observation_scores = self.score(test_samples)
      
        all_preds = []
        for score_seq in observation_scores:
            observation_scores = observation_scores.reshape(transition_weights.shape[0], -1)
            observation_lattice = fromArray(observation_scores)
            transition_fst = fromArray(transition_weights)

            decode_graph = openfst.compose(observation_lattice, transition_fst)
            pred_labels = viterbi(decode_graph)

            all_preds.append(pred_labels)

        return all_preds

    def score(self, train_samples):
        raise NotImplementedError()


# -=( MISC UTILS )==-----------------------------------------------------------
def isLattice(fst):
    for state in fst.states():
        input_labels = arc.ilabel for arc in fst.arcs(state)
        input_label = input_labels[0]
        if not all(i == input_label for i in input_labels):
            return False
    return is_lattice


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
        output_labels = tuple(str(i) for i in range(weights.shape[1]))

    if input_labels is None:
        if is_lattice:
            input_labels = tuple(str(i) for i in range(weights.shape[0]))
        else:
            input_labels = tuple(str(i) for i in range(weights.shape[2]))

    fst = openfst.VectorFst(arc_type=arc_type)
    zero = openfst.Weight.zero(fst.weight_type())
    one = openfst.Weight.one(fst.weight_type())

    if final_weight is None:
        final_weight = one

    init_state = fst.add_state()
    fst.set_start(init_state)

    if is_lattice:
        prev_state = init_state
        for sample_index, row in enumerate(weights):
            cur_state = fst.add_state()
            for i, weight in enumerate(row):
                input_label_index = sample_index
                output_label_index = i
                if weight != zero:
                    arc = openfst.Arc(
                        input_label_index, output_label_index,
                        weight, cur_state_index
                    )
                    fst.add_arc(prev_state_index, arc)
            prev_state = cur_state
        fst.set_final(cur_state, openfst.Weight(fst.weight_type(), final_weight))
    else:
        prev_state = init_state
        for sample_index, input_output in enumerate(weights):
            cur_state = fst.add_state()
            for i, outputs in enumerate(input_output):
                for j, weight in enumerate(outputs):
                    input_label_index = i
                    output_label_index = j
                    weight = openfst.Weight(fst.weight_type(), weight)
                    if weight != zero:
                        arc = openfst.Arc(
                            input_label_index, output_label_index,
                            weight, cur_state_index
                        )
                        fst.add_arc(prev_state_index, arc)
            prev_state = cur_state
        fst.set_final(cur_state, openfst.Weight(fst.weight_type(), final_weight)

    return fst


def toArray(lattice):
    zero = openfst.Weight.zero(lattice.weight_type())
    # one = openfst.Weight.one(lattice.weight_type())

    num_states = lattice.num_states()
    num_outputs = lattice.output_symbols().num_symbols()
    weights = np.full((num_states, num_outputs), float(zero))

    for state in states:
        for arc in lattice.arcs(state):
            weights[state, arc.olabel] = float(arc.weight)

    return weights, fst.weight_type()


def fromSequence(seq, arc_type='standard'):
    fst = openfst.VectorFst(arc_type=arc_type)
    zero = openfst.Weight.zero(fst.weight_type())
    one = openfst.Weight.one(fst.weight_type())

    init_state = fst.add_state()
    fst.set_start(init_state)

    cur_state = init_state
    for i in seq:
        next_state = fst.add_state()
        arc = openfst.Arc(i, i, one, next_state)
        fst.add_arc(cur_state, arc)
        cur_state = next_state

    fst.set_final(cur_state, openfst.Weight(fst.weight_type(), one)

    return sequence_acceptor


def fromTransitions(
        transition_weights, init_weights, final_weights, integerizer,
        index_names=None, semiring=None, as_dict=False):
    """ Instantiate a state machine from state transitions.

    Parameters
    ----------

    Returns
    -------
    """

    if semiring is None:
        semiring = semirings.LogSemiringWeight

    if as_dict:
        transitions = transition_weights.keys()
        final_states = final_weights.keys()
        init_states = init_weights.keys()
    else:
        transitions = (transition_weights != semiring.zero.value).nonzero().tolist()
        final_states = (final_weights != semiring.zero.value).nonzero().squeeze(1).tolist()
        init_states = (init_weights != semiring.zero.value).nonzero().squeeze(1).tolist()

    fst = FST(semiring, string_mapper=integerizer.__getitem__)
    init_state = fst.add_state()
    fst.set_initial_state(init_state)

    fst_states = {}
    for (prev, cur) in transitions:
        weight = transition_weights[prev, cur]

        prev_state = fst_states.get(prev, None)
        if prev_state is None:
            prev_state = fst.add_state()
            fst_states[prev] = prev_state

        cur_state = fst_states.get(cur, None)
        if cur_state is None:
            cur_state = fst.add_state()
            fst_states[cur] = cur_state

        if index_names is not None:
            prev = integerizer.index(index_names[prev])
            cur = integerizer.index(index_names[cur])

        fst.add_arc(
            prev_state, cur_state,
            input_label=prev, output_label=cur,
            weight=weight
        )

    for state in init_states:
        weight = init_weights[state]
        state_idx = fst_states[state]
        fst.add_arc(
            init_state, state_idx,
            input_label=0, output_label=0, weight=weight
        )

    for state in final_states:
        weight = final_weights[state]
        state_idx = fst_states[state]
        fst.set_final_weight(state_idx, weight)

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

    if semiring is None:
        semiring = semirings.BooleanSemiringWeight

    acceptor = FST(semiring, string_mapper=string_mapper, acceptor=True)
    init_state = acceptor.add_state()
    acceptor.set_initial_state(init_state)

    prev_state = init_state
    for token in input_seq:
        cur_state = acceptor.add_state()
        acceptor.add_arc(cur_state, cur_state, input_label=token, weight=semiring.one)
        acceptor.add_arc(prev_state, cur_state, input_label=token, weight=semiring.one)
        prev_state = cur_state
    acceptor.set_final_weight(prev_state, semiring.one)

    acceptor.display_dir = 'LR'

    return acceptor


# -=( CRF/HMM ALGORITHMS )==---------------------------------------------------
def logprob(observation_weights, gt_labels, transition_weights, final_weights):
    observation_lattice = fromArray(observation_weights)
    gt_lattice = fromSequence(gt_label)
    transition_fst = fromArray(transition_weights)

    denominator_graph = openfst.compose(observation_lattice, self._seq_fst)
    denom_betas, denominator = backward(denominator_graph)

    numerator_graph = openfst.intersect(gt_lattice, denominator_graph)
    num_betas, numerator = backward(numerator_graph)

    # numerator and denominator are in the log semiring
    log_prob = float(numerator) - float(denominator)

    grad = logprob_gradient(denominator_graph, betas=denom_betas)

    return log_prob, grad


def logprob_gradient(lattice, alphas=None, betas=None):
    if alphas is None:
        alphas = forward(lattice)

    if betas is None:
        betas = backward(lattice)

    num_states = lattice.num_states()
    num_outputs = lattice.output_symbols().num_symbols()
    zero = openfst.Weight.zero(lattice.weight_type())
    grad = np.full((num_states, num_outputs), float(zero))

    for state in lattice.states:
        for arc in lattice.arcs(state):
            w_incoming = alphas[state]
            w_outgoing = betas[arc.nextstate]
            w_arc = openfst.times(w_incoming, w_outgoing)

            # key = (state, arc.nextstate, arc.ilabel, arc.olabel)
            grad[state, arc.olabel] = float(w_arc)

    return grad.sum(axis=0), final_grad


def forward(lattice):
    if lattice.arc_type() != 'log':
        lattice = openfst.arcmap(lattice, map_type='to_log')
    alphas = openfst.shortestdistance(lattice)
    return alphas


def backward(lattice):
    if lattice.arc_type() != 'log':
        lattice = openfst.arcmap(lattice, map_type='to_log')
    betas = openfst.shortestdistance(lattice, reverse=True)
    log_Z = betas[lattice.start()]
    return betas, log_Z


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

    scores_fst = fromArray(-scores, semiring=semirings.TropicalSemiringWeight)
    label_fst = leftToRightAcceptor(label_seq, semiring=semirings.TropicalSemiringWeight)
    aligner = scores_fst.compose(label_fst)

    best_path_lattice = aligner.shortest_path()
    aligned_labels = best_path_lattice.get_unique_output_string()
    if aligned_labels is not None:
        aligned_labels = tuple(int(c) for c in aligned_labels)
    # NOTE: this gives the negative log probability of the single best path,
    #   not that of all paths, because best_path_lattice uses the tropical semiring.
    #   In this case it's fine because we only have one path anyway. If we want
    #   to marginalize over the k-best paths in the future, we will need to lift
    #   best_path_lattice to the real or log semiring before calling sum_paths.
    alignment_score = -(best_path_lattice.sum_paths().value)

    return aligned_labels, alignment_score


# -=( TRAINING )==-------------------------------------------------------------
def gradientStep(weights, gradient, step_size=1e-3):
    new_weights = weights + step_size * gradient
    return new_weights


def batchLoss(train_labels, observation_scores, transition_scores, final_weights):
    batch_loss = 0
    batch_grad = 0
    for score_seq, label_seq in zip(observation_scores, train_labels):
        log_prob, grad = logprob(
            score_seq, label_seq,
            transition_scores, final_weights
        )

        batch_loss += -log_prob
        batch_grad += -grad

    return batch_loss, batch_grad


# -=( MOVE TO ANOTHER MODULE )==-----------------------------------------------
def smoothCounts(
        edge_counts, state_counts, init_states, final_states,
        init_regularizer=0, final_regularizer=0,
        uniform_regularizer=0, diag_regularizer=0,
        override_transitions=False, structure_only=False, as_numpy=False):

    num_states = max(state_counts.keys()) + 1

    bigram_counts = np.zeros((num_states, num_states))
    for (i, j), count in edge_counts.items():
        bigram_counts[i, j] = count

    unigram_counts = np.zeros(num_states)
    for i, count in state_counts.items():
        unigram_counts[i] = count

    initial_counts = np.zeros(num_states)
    for i, count in init_states.items():
        initial_counts[i] = count

    final_counts = np.zeros(num_states)
    for i, count in final_states.items():
        final_counts[i] = count

    # Regularize the heck out of these counts
    initial_states = initial_counts.nonzero()[:, 0]
    for i in initial_states:
        bigram_counts[i, i] += init_regularizer

    final_states = final_counts.nonzero()[:, 0]
    for i in final_states:
        bigram_counts[i, i] += final_regularizer

    bigram_counts += uniform_regularizer
    diag_indices = np.diag_indices(bigram_counts.shape[0])
    bigram_counts[diag_indices] += diag_regularizer

    if override_transitions:
        logger.info('Overriding bigram_counts with an array of all ones')
        bigram_counts = np.ones_like(bigram_counts)

    if structure_only:
        bigram_counts = (bigram_counts > 0).float()
        initial_counts = (initial_counts > 0).float()
        final_counts = (final_counts > 0).float()

    denominator = bigram_counts.sum(1)
    transition_probs = bigram_counts / denominator[:, None]
    transition_probs[np.isnan(transition_probs)] = 0
    initial_probs = initial_counts / initial_counts.sum()
    final_probs = (final_counts > 0).float()

    if as_numpy:
        def to_numpy(x):
            return x.numpy().astype(float)
        return tuple(map(to_numpy, (transition_probs, initial_probs, final_probs)))

    return transition_probs, initial_probs, final_probs


def countSeqs(seqs):
    """ Count n-gram statistics on a collection of sequences.

    Parameters
    ----------
    seqs : iterable( iterable(Hashable) )

    Returns
    -------
    bigram_counts : collections.defaultdict((Hashable, Hashable) -> int)
    unigram_counts : collections.defaultdict(Hashable -> int)
    initial_counts : collections.defaultdict(Hashable -> int)
    final_counts : collections.defaultdict(Hashable -> int)
    """

    bigram_counts = collections.defaultdict(int)
    unigram_counts = collections.defaultdict(int)
    initial_counts = collections.defaultdict(int)
    final_counts = collections.defaultdict(int)

    for seq in seqs:
        initial_counts[seq[0]] += 1
        final_counts[seq[-1]] += 1
        for state in seq:
            unigram_counts[state] += 1
        for prev, cur in zip(seq[:-1], seq[1:]):
            bigram_counts[prev, cur] += 1

    return bigram_counts, unigram_counts, initial_counts, final_counts


# -=( DEPRECATED )==-----------------------------------------------------------
def argmax(decode_graph, count=1, squeeze=True):
    """ """

    # FIXME

    if decode_graph.semiring is not semirings.TropicalSemiringWeight:
        if decode_graph.semiring is semirings.LogSemiringWeight:
            def converter(weight):
                return -weight.value
        elif decode_graph.semiring is semirings.RealSemiringWeight:
            def converter(weight):
                return -weight.value.log()
        else:
            raise NotImplementedError("Conversion to tropical semiring isn't implemented yet")
        decode_graph = decode_graph.lift(
            semiring=semirings.TropicalSemiringWeight, converter=converter
        )

    lattice = decode_graph.shortest_path(count=count)
    best_paths = tuple(path.output_path for path in lattice.iterate_paths())

    if squeeze and len(best_paths) == 1:
        return best_paths[0]

    return best_paths
