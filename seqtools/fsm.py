import collections
import logging
import math
import itertools

import graphviz
import torch

import mfst
from . import semirings, utils
from ._fsm_utils.expectation_semiring import ExpectationSemiringWeight


logger = logging.getLogger(__name__)


class FST(mfst.FST):
    EXPECTATION_USES_FB = False

    def __init__(
            self, *args,
            display_dir='LR', float_precision=None, expectation_uses_fb=None,
            **kwargs):
        self.display_dir = display_dir
        self.float_precision = float_precision

        super().__init__(*args, **kwargs)

        if expectation_uses_fb is None:
            self.expectation_uses_fb = FST.EXPECTATION_USES_FB

        if expectation_uses_fb:
            self._sum_paths = self.sum_paths
            self.sum_paths = self.sum_paths_with_fb

        if self._string_mapper is not None:
            self.__str__ = self.prettyPrint

    def lift(self, semiring=None, converter=None):
        """ Wrap mfst.FST.lift so it returns an FST of the same type as `self`. """
        lifted = super().lift(semiring=semiring, converter=converter)
        params = dict(
            _fst=lifted._fst,
            semiring_class=lifted._semiring_class,
            acceptor=lifted._acceptor,
            string_mapper=lifted._string_mapper
        )
        return type(self)(**params)

    def sum_paths_fb(self):
        """ Use forward-backward to find the pathsum of an FST in the expectation semiring. """

        assert self.semiring is ExpectationSemiringWeight

        self_real = self.lift(ExpectationSemiringWeight, converter=lambda w: w.dropvalue())
        alpha = self_real.shortest_distance()
        beta = self_real.shortest_distance(reverse=True)
        Z = beta[self_real.initial_state]

        total = ExpectationSemiringWeight(0)
        for s in self.states:
            # if s is final, then get_arcs will yield an arc to state -1 with the final-state weight
            for e in self.get_arcs(s):
                multiplier = alpha[s] * (
                    beta[e.nextstate] if e.nextstate >= 0 else ExpectationSemiringWeight.one
                )
                # avoid multiplying the big `e.weight` by alpha and beta separately
                total += multiplier * e.weight

        # The second element of total will now be correct, but we need to replace
        # the first element with Z. Here's a slightly hacky approach that remains
        # safe (even if total and Z are encoded with different multipliers).
        return total + (Z.dropvalue() - total.dropvalue())

    def sum_paths_with_fb(self, *args, **kwargs):
        if self.expectation_uses_fb and self.semiring is ExpectationSemiringWeight:
            return self.sum_paths_fb(*args, **kwargs)
        else:
            return self._sum_paths(*args, **kwargs)

    def determinize_with_merging(self, *args, **kwargs):
        assert self.semiring is ExpectationSemiringWeight
        # temporarily modifies behavior of `quantize` on expectation semirings
        # (WARNING: not thread-safe)
        ExpectationSemiringWeight.aggressive_quantization = True
        # must push first for aggressive quantization to be correct
        result = self.push().determinize(*args, **kwargs)
        ExpectationSemiringWeight.aggressive_quantization = False
        return result

    def create_from_observable(self, oo, alphabet, wildcard='?'):
        """
        Return an FSA that accepts all sequences compatible with `oo`.  The `wildcard`
        symbol in `oo` is allowed to match any element of `alphabet`.  The FSA
        uses the same semiring and other parameters as `self`.
        """

        fsa = self.constructor(acceptor=True)
        start_idx = fsa.add_state()
        fsa.initial_state = start_idx

        prev_state_idx = start_idx
        for o in oo:
            state_idx = fsa.add_state()
            if o == wildcard:
                for x in alphabet:
                    fsa.add_arc(prev_state_idx, state_idx, input_label=x)
            else:
                fsa.add_arc(prev_state_idx, state_idx, input_label=o)
            prev_state_idx = state_idx
        end_idx = state_idx
        fsa.set_final_weight(end_idx)
        return fsa

    def toGraphviz(self, action_dict=None, state_dict=None):
        """
        When returned from an ipython cell, this will generate the FST visualization
        """

        fst = graphviz.Digraph("finite state machine", filename="fsm.gv")
        fst.attr(rankdir=self.display_dir)

        # here we are actually going to read the states from the FST and generate nodes for them
        # in the output source code
        zero = self._make_weight('__FST_ZERO__')
        one = self._make_weight('__FST_ONE__')
        initial_state = self.initial_state

        for sid in range(self.num_states):
            finalW = ''
            is_final = False
            ww = self._fst.FinalWeight(sid)
            # look at the raw returned value to see if it is zero (unset)
            if ww is not None and (not isinstance(ww, str) or '__FST_ONE__' == ww):
                ww = self._make_weight(ww)
                if zero != ww:
                    is_final = True
                    if not (one == ww and sid != initial_state):
                        if isinstance(ww, semirings.BooleanSemiringWeight):
                            weight_str = f"{ww}"
                        else:
                            weight_str = f'{ww:.2f}'
                        finalW = f'\n({weight_str})'
            label = f'{sid}{finalW}'
            if is_final:
                fst.node(str(sid), label=label, shape='doublecircle')

        if self._string_mapper:
            if self._string_mapper is chr:
                def make_label(x):
                    if x == 32:
                        return '(spc)'
                    elif x < 32:
                        return str(x)
                    else:
                        return chr(x)
            else:
                def make_label(x):
                    return str(self._string_mapper(x))
        else:
            make_label = str

        fst.attr('node', shape='circle')
        for sid in range(self.num_states):
            to = collections.defaultdict(list)
            for arc in self.get_arcs(sid):
                if arc.nextstate == -1:
                    continue

                # Make the arc label
                label = ''
                if arc.input_label == 0:
                    label += '\u03B5'  # epsilon
                else:
                    label += make_label(arc.input_label)
                if arc.input_label != arc.output_label:
                    label += ':'
                    if arc.output_label == 0:
                        label += '\u03B5'
                    else:
                        label += make_label(arc.output_label)
                if one != arc.weight:
                    if isinstance(arc.weight, semirings.BooleanSemiringWeight):
                        weight_str = f"/{arc.weight}"
                    else:
                        weight_str = f'/{arc.weight:.2f}'
                    label += weight_str
                to[arc.nextstate].append(label)

            # Draw the arc
            for dest, values in to.items():
                if len(values) > 3:
                    values = values[0:2] + ['. . .']
                label = '\n'.join(values)
                fst.edge(str(sid), str(dest), label=label)

        if initial_state >= 0:
            # mark the start state
            fst.node('', shape='point')
            fst.edge('', str(initial_state))

        return fst

    def prettyPrint(self):
        for state in self.states:
            for arc in self.get_arcs(state):
                in_label = self._string_mapper(arc.input_label)
                out_label = self._string_mapper(arc.output_label)
                print(f"{state} -> {arc.nextstate} {in_label} : {out_label} / {arc.weight}")

    def _repr_html_(self):
        if self.num_states == 0:
            return '<code>Empty FST</code>'

        # if the machine is too big, do not attempt to make ipython display it
        # otherwise it ends up crashing and stuff...
        if self.num_states > 1000:
            string = (
                f'FST too large to draw graphic, use fst.full_str()<br />'
                f'<code>FST(num_states={self.num_states})</code>'
            )
            return string

        fst = self.toGraphviz()

        return fst._repr_svg_()


class FstIntegerizer(object):
    def __init__(self, iterable=[], prepend_epsilon=True):
        # OpenFST hardcodes zero to represent epsilon transitions---so make sure
        # our integerizer is consistent with that.
        if prepend_epsilon:
            iterable = ['epsilon'] + iterable

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


class FstScorer(object):
    """ Modify arbitrary data scores using a sequence model implemented as an FST.

    You can use this mixin class to easily make a BiLSTM-CRF, for example.

    Attributes
    ----------
    """

    def __init__(
            self, transition_weights, *super_args,
            initial_weights=None, final_weights=None, semiring=None, requires_grad=False,
            forward_uses_max=False,
            integerizer=None, vocabulary=None, device=None,
            **super_kwargs):
        """
        Parameters
        ----------
        """

        super().__init__(*super_args, **super_kwargs)

        if device is None:
            device = transition_weights.device

        if vocabulary is None:
            vocabulary = tuple(range(transition_weights.shape[0]))

        if integerizer is None:
            integerizer = HashableFstIntegerizer(vocabulary)

        if semiring is None:
            semiring = semirings.LogSemiringWeight

        if final_weights is None:
            final_weights = torch.full(transition_weights.shape[0:1], semiring.one.value)

        if initial_weights is None:
            initial_weights = torch.full(transition_weights.shape[0:1], semiring.one.value)

        self.vocabulary = vocabulary

        self.integerizer = integerizer

        self.semiring = semiring

        self.forward_uses_max = forward_uses_max

        self.device = device

        # Register semiring zero and one elements so they will be on the same
        # device as the model's params
        self._semiring_zero = torch.nn.Parameter(
            self.semiring.zero.value.to(device),
            requires_grad=False
        )
        self._semiring_one = torch.nn.Parameter(
            self.semiring.one.value.to(device),
            requires_grad=False
        )
        self.semiring.zero = self.semiring(self._semiring_zero)
        self.semiring.one = self.semiring(self._semiring_one)

        self.transition_weights = torch.nn.Parameter(
            transition_weights.to(device), requires_grad=requires_grad
        )
        self.initial_weights = torch.nn.Parameter(
            initial_weights.to(device), requires_grad=requires_grad
        )
        self.final_weights = torch.nn.Parameter(
            final_weights.to(device), requires_grad=requires_grad
        )

        self.label_scorer = fromTransitions(
            self.transition_weights, self.initial_weights, self.final_weights, self.mapper,
            semiring=self.semiring
        )

        self._label_scorer_tropical_sr = None

    def mapper(self, obj):
        index = self.integerizer.index(obj)

        if index is None:
            raise KeyError()

        return index

    @property
    def label_scorer_tropical_sr(self):
        if self._label_scorer_tropical_sr is None:
            self._label_scorer_tropical_sr = self.label_scorer.lift(
                semiring=semirings.TropicalSemiringWeight, converter=lambda x: -(x.value)
            )

        return self._label_scorer_tropical_sr

    def forward(self, batch, use_tropical_semiring=None):
        if use_tropical_semiring is None:
            use_tropical_semiring = self.forward_uses_max

        # Predict all data scores in batch mode
        batch_data_scores = super().forward(batch)

        if use_tropical_semiring:
            label_scorer = self.label_scorer_tropical_sr
            semiring = semirings.TropicalSemiringWeight
            batch_data_scores = -batch_data_scores
        else:
            label_scorer = self.label_scorer
            semiring = self.semiring

        # Construct decode graphs sequence-by-sequence
        decode_graphs = tuple(
            fromArray(
                data_scores, semiring=semiring,
                input_labels=tuple(range(1, data_scores.shape[1] + 1)),
                output_labels=tuple(self.mapper(x) for x in self.vocabulary)
            ).compose(label_scorer)
            for data_scores in batch_data_scores
        )

        return decode_graphs

    def predict(self, decode_graphs):
        x = tuple(
            torch.tensor(argmax(decode_graph, count=1)) - 1
            for decode_graph in decode_graphs
        )

        return x


class AbstractFstSequenceModel(object):
    def __init__(self, *args, **kwargs):
        self._process_fst = None
        self._integerizer = None
        self._initialize(*args, **kwargs)

    def fit(
            self, label_seqs, *feat_seqs, process_only=False, observation_only=False,
            **super_kwargs):

        if not process_only:
            self._fitObservationModel(label_seqs, *feat_seqs)
            label_seqs = self._preprocessLabelSeqs(label_seqs)
            self._integerizer.updateFromSequences(label_seqs)

        if not observation_only:
            self._process_fst = self._fitProcessModel(label_seqs)

    def score(self, *feat_seqs):
        observation_scores = self._scoreObservations(*feat_seqs)
        scores = observation_scores.compose(self._process_fst)
        return scores

    def predictSeq(self, *feat_seqs):
        score_lattice = self.score(*feat_seqs)
        shortest_path = score_lattice.shortest_path()
        predictions = shortest_path.get_unique_output_string()
        return predictions

    @property
    def num_states(self):
        # FIXME
        return None

    def _initialize(self):
        raise NotImplementedError

    def _fitIntegerizer(self, *label_seqs):
        raise NotImplementedError

    def _fitObservationModel(self, *feat_seqs):
        raise NotImplementedError

    def _fitProcessModel(self, *feat_seqs):
        raise NotImplementedError

    def _scoreObservations(self, *feat_seqs):
        raise NotImplementedError


# -=( LIBRARY FUNCTIONS )==-----------------------------------------------------
def fstNLLLoss(decode_graphs, label_seqs):
    """ Compute structured negative-log-likelihood loss using FST methods.

    NOTE: Right now this library handles mini-batches by iterating over each
        item in the batch, so batch operations are not vectorized.

    Parameters
    ----------
    decode_graphs : iterable(fsm.FST)
    label_seqs : torch.Tensor of int, shape (batch_size, num_samples)

    Returns
    -------
    loss : float
    """

    def generateLosses(decode_graphs, label_seqs):
        for decode_graph, label_seq in zip(decode_graphs, label_seqs):
            # FIXME: We need a more consistent way of mapping to/from the arc labels
            gt_path_acceptor = FST(decode_graph.semiring).create_from_string(label_seq + 1)
            gt_path_score = decode_graph.compose(gt_path_acceptor).sum_paths().value
            total_path_score = decode_graph.sum_paths().value

            if decode_graph.semiring is semirings.LogSemiringWeight:
                yield -(gt_path_score - total_path_score)
            elif decode_graph.semiring is semirings.RealSemiringWeight:
                yield -(torch.log(gt_path_score) - torch.log(total_path_score))
            elif decode_graph.semiring is semirings.TropicalSemiringWeight:
                yield gt_path_score - total_path_score
            else:
                raise AssertionError(f"{decode_graph.semiring} is not supported by fstNLLLoss")

    return torch.stack(tuple(generateLosses(decode_graphs, label_seqs)))


def toArray(fst, input_labels=None, output_labels=None, array_constructor=None):
    """ Return an array representing an FST's edge weights.

    This function is meant to be used to create an instance of pytorch-struct's
    LinearChainCRF, which we can then backpropagate through during training.

    Parameters
    ----------
    fst : FST
    input_labels : iterable(int or str), optional
        If this argument is not provided, it is taken to be the input labels
        in `FST`.
    output_labels : iterable(int or str) , optional
        If this argument is not provided, it is taken to be the output labels
        in `FST`.
    array_constructor : function, optional
        Use this to decide the return type of the array. Returns ``dict`` by
        default.

    Returns
    -------
    weights : array_like or dict, shape (num_states, num_input_labels, num_output_labels)
        The FST's edge weights, arranged as an array.
    semiring : semirings.AbstractSemiringWeight
        The FST's semiring.
    """

    weights_dict, _input_labels, _output_labels = traverse(fst)

    if array_constructor is None:
        def array_constructor(*args):
            return {}

    if input_labels is None:
        input_labels = list(_input_labels)

    if output_labels is None and not fst._acceptor:
        output_labels = list(_output_labels)

    if fst._acceptor:
        weights = array_constructor((fst.num_states, len(input_labels)), fst.semiring_zero.value)
        for (state_index, input_label), weight in weights_dict.items():
            input_index = input_labels.index(input_label)
            weights[state_index, input_index] = weight.value
    else:
        weights = array_constructor(
            (fst.num_states, len(input_labels), len(output_labels)),
            fst.semiring_zero.value
        )
        for (state_index, input_label, output_label), weight in weights_dict.items():
            input_index = input_labels.index(input_label)
            output_index = output_labels.index(output_label)
            weights[state_index, input_index, output_index] = weight.value

    return weights, fst.semiring, input_labels, output_labels


def fromArray(
        weights, final_weight=None, semiring=None, string_mapper=None,
        input_labels=None, output_labels=None):
    """ Instantiate a state machine from an array of weights.

    TODO: Right now this only takes input arrays that create linear-chain state
       machines, but it can be extended to create arbitrary arrays by taking an
       input with shape (num_states, num_input_labels, num_output_labels).

    Parameters
    ----------
    weights : array_like, shape (num_inputs, num_outputs)
        Needs to implement `.shape`, so it should be a numpy array or a torch
        tensor.
    final_weight : semirings.AbstractSemiringWeight, optional
        Should have the same type as `semiring`. Default is `semiring.zero`
    semiring : semirings.AbstractSemiringWeight, optional
        Default is `semirings.BooleanSemiringWeight`.
    string_mapper : function, optional
    is_linear_chain : bool, optional

    Returns
    -------
    fst : fsm.FST
        The transducer's arcs have input labels corresponding to the state
        they left, and output labels corresponding to the state they entered.
    """

    if semiring is None:
        semiring = semirings.LogSemiringWeight

    if final_weight is None:
        final_weight = semiring.one

    if len(weights.shape) == 3:
        is_acceptor = False
    elif len(weights.shape) == 2:
        is_acceptor = True
    else:
        raise AssertionError(f"weights have unrecognized shape {weights.shape}")

    if input_labels is None:
        input_labels = tuple(str(i) for i in range(weights.shape[1]))

    if output_labels is None and not is_acceptor:
        output_labels = tuple(str(i) for i in range(weights.shape[2]))

    fst = FST(semiring, string_mapper=string_mapper, acceptor=is_acceptor)
    init_state = fst.add_state()
    fst.set_initial_state(init_state)

    if is_acceptor:
        prev_state = init_state
        for sample_index, row in enumerate(weights):
            cur_state = fst.add_state()
            for i, weight in enumerate(row):
                if semiring(weight) != semiring.zero:
                    fst.add_arc(
                        prev_state, cur_state, input_label=input_labels[i],
                        weight=weight
                    )
            prev_state = cur_state
        fst.set_final_weight(cur_state, final_weight)
    else:
        prev_state = init_state
        for sample_index, input_output in enumerate(weights):
            cur_state = fst.add_state()
            for i, outputs in enumerate(input_output):
                for j, weight in enumerate(outputs):
                    if semiring(weight) != semiring.zero:
                        fst.add_arc(
                            prev_state, cur_state,
                            input_label=input_labels[i], output_label=output_labels[j],
                            weight=weight
                        )
            prev_state = cur_state
        fst.set_final_weight(cur_state, final_weight)

    fst.display_dir = 'LR'

    return fst


def fromTransitions(
        transition_weights, init_weights, final_weights, label_mapper,
        semiring=None, as_dict=False):
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
    else:
        transitions = (transition_weights != semiring.zero.value).nonzero().tolist()
        final_states = (final_weights != semiring.zero.value).nonzero().squeeze(1).tolist()
        init_states = (init_weights != semiring.zero.value).nonzero().squeeze(1).tolist()

    fst = FST(semiring)
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

        fst.add_arc(
            prev_state, cur_state,
            input_label=label_mapper(prev), output_label=label_mapper(cur),
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


def sequenceFsa(seqs, integerizer):
    sequence_acceptor = FST(
        semirings.TropicalSemiring, string_mapper=lambda i: str(integerizer[i])
    ).create_from_string(seqs[0])

    for seq in seqs[1:]:
        sequence_acceptor = sequence_acceptor.union(
            FST(
                semirings.TropicalSemiring, string_mapper=lambda i: str(integerizer[i])
            ).create_from_string(seq)
        )

    sequence_acceptor = sequence_acceptor.determinize().minimize()
    return sequence_acceptor


# -=( HELPER FUNCTIONS )==-----------------------------------------------------
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


def printGradNorm(grad):
    logger.info(f'grad: {type(grad)}')
    logger.info(f'grad[0]: {type(grad[0])}')
    logger.info(f'gradsize: {grad[0].size()}')
    logger.info(f'grad norm: {grad[0].norm()}')


def argmax(decode_graph, count=1, squeeze=True):
    """ """

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


def traverse(fst):
    """ Traverse a transducer, accumulating edge weights and labels.

    Parameters
    ----------
    fst : FST

    Returns
    -------
    weights_dict : dict((int, int) -> semirings.AbstractSemiringWeight)
    all_input_labels : set(int)
    all_output_labels : set(int)
    """

    # FIXME: make this a method in the FST or something
    def make_label(x):
        if x == 0:
            x = 'epsilon'
        if x == 32:
            x = '(spc)'
        elif x < 32:
            x = str(x)
        else:
            # OpenFST will encode characters as integers
            x = int(chr(x))
        return x

    weights_dict = {}
    all_input_labels = set()
    all_output_labels = set()

    state = fst.initial_state
    to_visit = [state]
    queued_states = set(to_visit)
    while to_visit:
        state = to_visit.pop()
        for edge in fst.get_arcs(state):
            # Final weights are implemented as arcs whose inputs and outputs
            # are epsilon, and whose next state is -1 (ie an impossible next
            # state). I account for final weights using fst.get_final_weight,
            # so we can skip them here.
            if edge.nextstate < 0:
                continue

            if edge.nextstate not in queued_states:
                queued_states.add(edge.nextstate)
                to_visit.append(edge.nextstate)

            weight = edge.weight

            input_label = make_label(edge.input_label)
            output_label = make_label(edge.output_label)
            if fst._acceptor:
                edge_labels = (state, input_label)
            else:
                edge_labels = (state, input_label, output_label)
            if edge_labels in weights_dict:
                weights_dict[edge_labels] += weight
            else:
                weights_dict[edge_labels] = weight
            all_input_labels.add(input_label)
            all_output_labels.add(output_label)

    return weights_dict, all_input_labels, all_output_labels


# -=( DEPRECATED )==-----------------------------------------------------------
def actionTransitionFsa(seqs, integerizer, semiring=None):
    if semiring is None:
        def semiring_transform(w):
            return -math.log(w)
        semiring = semirings.TropicalSemiring

    else:
        def transform_weight(w):
            return semiring_transform(w)

    edge_counts, state_counts, init_states, final_states = countSeqs(seqs)

    action_acceptor = FST(semiring, string_mapper=lambda i: str(integerizer[i]))

    fst_states = {}
    for (prev, cur), transition_count in edge_counts.items():
        action_id = cur

        prev_state = fst_states.get(prev, None)
        if prev_state is None:
            prev_state = action_acceptor.add_state()
            fst_states[prev] = prev_state

        cur_state = fst_states.get(cur, None)
        if cur_state is None:
            cur_state = action_acceptor.add_state()
            fst_states[cur] = cur_state

        weight = transform_weight(transition_count / state_counts[prev])
        action_acceptor.add_arc(
            prev_state, cur_state,
            input_label=action_id, output_label=action_id, weight=weight
        )

    for state in init_states:
        state_idx = fst_states[state]
        action_acceptor.set_initial_state(state_idx)

    for state in final_states:
        state_idx = fst_states[state]
        weight = transform_weight(1)
        action_acceptor.set_final_weight(state_idx, weight)

    return action_acceptor


def stateTransitionFsa(seqs, integerizer):
    edge_counts, state_counts, init_states, final_states = countSeqs(seqs)

    fst_states = {}

    action_acceptor = FST(semirings.TropicalSemiring, string_mapper=lambda i: str(integerizer[i]))

    for (prev, cur), transition_count in edge_counts.items():
        if prev == cur:
            action = ''
        else:
            action, = cur.assembly_state - prev.assembly_state
        action_id = integerizer.index(action)

        prev_state = fst_states.get(prev, None)
        if prev_state is None:
            prev_state = action_acceptor.add_state()
            fst_states[prev] = prev_state

        cur_state = fst_states.get(cur, None)
        if cur_state is None:
            cur_state = action_acceptor.add_state()
            fst_states[cur] = cur_state

        weight = -math.log(transition_count / state_counts[prev])
        action_acceptor.add_arc(
            prev_state, cur_state,
            input_label=action_id, output_label=action_id, weight=weight
        )

    for state in init_states:
        state_idx = fst_states[state]
        action_acceptor.set_initial_state(state_idx)

    for state in final_states:
        state_idx = fst_states[state]
        weight = -math.log(1)
        action_acceptor.set_final_weight(state_idx, weight)

    return action_acceptor
