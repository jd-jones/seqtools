import logging

import numpy as np
from sklearn import metrics

from . import utils


logger = logging.getLogger(__name__)


class RationalPerformanceMetric():
    """
    Performance metric with a numerator and a denominator.
    """

    def __init__(self):
        self.initializeCounts()

    def initializeCounts(self):
        self._numerator = 0
        self._denominator = 0

    def evaluate(self):
        return utils.safeDivide(self._numerator, self._denominator)

    def __str__(self):
        return f'{self.evaluate():.5f}'

    def accumulate(self, outputs=None, labels=None, loss=None):
        self._numerator += self._count_numerator(outputs, labels, loss)
        self._denominator += self._count_denominator(outputs, labels, loss)

    def _count_numerator(self, outputs=None, labels=None, loss=None):
        raise NotImplementedError()

    def _count_denominator(self, outputs=None, labels=None, loss=None):
        raise NotImplementedError()


class AverageLoss(RationalPerformanceMetric):
    def _count_numerator(self, outputs=None, labels=None, loss=None):
        return loss

    def _count_denominator(self, outputs=None, labels=None, loss=None):
        return 1

    def __str__(self):
        name = 'loss'
        return name + ': ' + super().__str__()


class ConfusionPerformanceMetric(RationalPerformanceMetric):
    def __init__(self):
        self.initializeCounts()

    def initializeCounts(self):
        self._true_positives = 0
        self._true_negatives = 0
        self._false_positives = 0
        self._false_negatives = 0

    def __str__(self):
        return f'{self.evaluate() * 100:5.2f}%'

    def accumulate(self, predicted=None, true=None, loss=None):
        self._accumulate_confusions(predicted, true)

    def _accumulate_confusions(self, predicted, true):
        self._true_positives += truePositives(predicted, true)
        self._true_negatives += trueNegatives(predicted, true)
        self._false_positives += falsePositives(predicted, true)
        self._false_negatives += falseNegatives(predicted, true)

    @property
    def _numerator(self):
        raise NotImplementedError()

    @property
    def _denominator(self):
        raise NotImplementedError()


class Fmeasure(ConfusionPerformanceMetric):
    def __init__(self, beta=1):
        super().__init__()
        self._beta = beta

    @property
    def _numerator(self):
        return (1 + self._beta ** 2) * self._true_positives

    @property
    def _denominator(self):
        denom = (
            self._false_positives
            + (self._beta ** 2) * self._false_negatives
            + (1 + self._beta ** 2) * self._true_positives
        )
        return denom

    def __str__(self):
        name = f'F_{self._beta}'
        return name + ': ' + super().__str__()


class Recall(ConfusionPerformanceMetric):
    @property
    def _numerator(self):
        return self._true_positives

    @property
    def _denominator(self):
        return self._true_positives + self._false_negatives

    def __str__(self):
        name = 'rec'
        return name + ': ' + super().__str__()


class Precision(ConfusionPerformanceMetric):
    @property
    def _numerator(self):
        return self._true_positives

    @property
    def _denominator(self):
        return self._true_positives + self._false_positives

    def __str__(self):
        name = 'prc'
        return name + ': ' + super().__str__()


class Accuracy(ConfusionPerformanceMetric):
    @property
    def _numerator(self):
        return self._true_positives + self._true_negatives

    @property
    def _denominator(self):
        data_positives = self._true_positives + self._false_negatives
        data_negatives = self._false_positives + self._true_negatives
        return data_positives + data_negatives

    def __str__(self):
        name = 'acc'
        return name + ': ' + super().__str__()


def truePositives(predicted, true):
    try:
        return sum(p == t for p, t in zip(predicted, true) if t)
    except TypeError:
        return int(predicted == true) if true else 0


def trueNegatives(predicted, true):
    try:
        return sum(p == t for p, t in zip(predicted, true) if not t)
    except TypeError:
        return int(predicted == true) if not true else 0


def falsePositives(predicted, true):
    try:
        return sum(p != t for p, t in zip(predicted, true) if t)
    except TypeError:
        return int(predicted != true) if true else 0


def falseNegatives(predicted, true):
    try:
        return sum(p != t for p, t in zip(predicted, true) if not t)
    except TypeError:
        return int(predicted != true) if not true else 0


def classAccuracy(true_label_seqs, predicted_label_seqs):
    avg_accuracies = utils.iterate(
        seqClassAccuracy,
        true_label_seqs,
        predicted_label_seqs
    )
    true_label_seq_lens = [len(s) for s in true_label_seqs]

    avg_accuracies = np.array(avg_accuracies)
    true_label_seq_lens = np.array(true_label_seq_lens)

    num_classes = utils.numClasses([l for ls in true_label_seqs for l in ls])

    avg_accy = np.average(avg_accuracies, weights=true_label_seq_lens)
    chance = 1 / num_classes

    return avg_accy, chance


def seqClassAccuracy(true_label_seq, predicted_label_seq):
    num_correct = 0
    for true, predicted in zip(true_label_seq, predicted_label_seq):
        num_correct += int((true == predicted).all())
    total = len(true_label_seq)

    return num_correct / total


def avgAccuracy(true_label_seqs, predicted_label_seqs):
    true_labels = utils.toFlattenedArray(true_label_seqs)
    predicted_labels = utils.toFlattenedArray(predicted_label_seqs)

    metric = metrics.accuracy_score(true_labels.ravel(), predicted_labels.ravel())
    chance = 0.5

    return metric, chance


def avgPrecision(true_label_seqs, predicted_label_seqs):
    true_labels = utils.toFlattenedArray(true_label_seqs)
    predicted_labels = utils.toFlattenedArray(predicted_label_seqs)

    metric = metrics.precision_score(true_labels.ravel(), predicted_labels.ravel())
    chance = true_labels.ravel().sum() / true_labels.size

    return metric, chance


def avgRecall(true_label_seqs, predicted_label_seqs):
    true_labels = utils.toFlattenedArray(true_label_seqs)
    predicted_labels = utils.toFlattenedArray(predicted_label_seqs)

    metric = metrics.recall_score(true_labels.ravel(), predicted_labels.ravel())
    chance = 0.5

    return metric, chance


def edgeDistance(true_label_seqs, predicted_label_seqs):
    avg_distances = utils.iterate(seqEdgeDistance, true_label_seqs, predicted_label_seqs)
    true_label_seq_lens = [len(s) for s in true_label_seqs]

    avg_distances = np.array(avg_distances)
    true_label_seq_lens = np.array(true_label_seq_lens)

    metric = np.average(avg_distances, weights=true_label_seq_lens)
    chance = -1

    return metric, chance


def seqEdgeDistance(true_label_seq, predicted_label_seq):
    sum_dist = 0
    for true, predicted in zip(true_label_seq, predicted_label_seq):
        sum_dist += (true != predicted).sum()
    total = len(true_label_seq)

    return sum_dist / total


def nonempty(assembly_state):
    return assembly_state.any()


def countTrue(true, pred, precision='state', denom_mode='accuracy'):
    if precision == 'block':
        p_blocks = set(pred.blocks.keys())
        t_blocks = set(true.blocks.keys())
        num_true = len(p_blocks & t_blocks)
        if denom_mode == 'accuracy':
            num_total = len(p_blocks | t_blocks)
        elif denom_mode == 'precision':
            num_total = len(p_blocks)
        elif denom_mode == 'recall':
            num_total = len(t_blocks)
    elif precision == 'edge':
        p_edges = pred.connections
        t_edges = true.connections
        num_true = np.sum(p_edges & t_edges)
        if denom_mode == 'accuracy':
            num_total = np.sum(p_edges | t_edges)
        elif denom_mode == 'precision':
            num_total = np.sum(p_edges)
        elif denom_mode == 'recall':
            num_total = np.sum(t_edges)
    elif precision == 'state':
        num_true = int(pred == true)
        if denom_mode == 'accuracy':
            num_total = 1
        elif denom_mode == 'precision':
            num_total = int(nonempty(pred))
        elif denom_mode == 'recall':
            num_total = int(nonempty(true))
        # Don't count empty states in precision/recall mode
        num_true *= num_total

    return num_true, num_total


def countSeq(true_seq, pred_seq, precision='states', denom_mode='accuracy'):
    len_true = len(true_seq)
    len_pred = len(pred_seq)
    if len_true != len_pred:
        err_str = f'{len_true} samples in true_seq'
        err_str += f' != {len_pred} samples in pred_seq'
        raise ValueError(err_str)

    num_correct = 0
    num_total = 0
    for true, pred in zip(true_seq, pred_seq):
        cur_correct, cur_total = countTrue(true, pred, precision=precision, denom_mode=denom_mode)
        num_correct += cur_correct
        num_total += cur_total

    return num_correct, num_total


def numberCorrect(
        true_seq, predicted_seq,
        ignore_empty_true=False, ignore_empty_pred=False, precision='states'):

    len_true = len(true_seq)
    len_pred = len(predicted_seq)
    if len_true != len_pred:
        err_str = f'{len_true} samples in true_seq != {len_pred} samples in predicted_seq'
        raise ValueError(err_str)

    if ignore_empty_true:
        indices = tuple(i for i, s in enumerate(true_seq) if nonempty(s))
        predicted_seq = tuple(predicted_seq[i] for i in indices)
        true_seq = tuple(true_seq[i] for i in indices)
        if not len(true_seq):
            warn_str = 'All ground-truth sequences were empty!'
            logger.warning(warn_str)

    if ignore_empty_pred:
        indices = tuple(i for i, s in enumerate(predicted_seq) if nonempty(s))
        predicted_seq = tuple(predicted_seq[i] for i in indices)
        true_seq = tuple(true_seq[i] for i in indices)

    num_correct = 0
    num_total = 0
    for p, t in zip(predicted_seq, true_seq):
        if precision == 'states':
            num_correct += int(p == t)
            num_total += 1
        elif precision == 'edges':
            num_correct += int(np.all(p.connections == t.connections))
            num_total += 1
        elif precision == 'vertices':
            num_correct += int(p.blocks == t.blocks)
            num_total += 1
        elif precision == 'structure':
            if not (nonempty(p) and nonempty(t)):
                num_correct += int(p == t)
            else:
                num_correct += int(p < t or p > t or p == t)
            num_total += 1
        elif precision == 'blocks':
            p_blocks = set(p.blocks.keys())
            t_blocks = set(t.blocks.keys())
            num_correct += len(p_blocks & t_blocks)
            num_total += len(p_blocks | t_blocks)
        elif precision == 'blocks_recall':
            p_blocks = set(p.blocks.keys())
            t_blocks = set(t.blocks.keys())
            num_correct += len(p_blocks & t_blocks)
            num_total += len(t_blocks)
        elif precision == 'blocks_precision':
            p_blocks = set(p.blocks.keys())
            t_blocks = set(t.blocks.keys())
            num_correct += len(p_blocks & t_blocks)
            num_total += len(p_blocks)
        elif precision == 'avg_edge':
            p_edges = p.connections
            t_edges = t.connections
            num_correct += np.sum(p_edges & t_edges)
            num_total += np.sum(p_edges | t_edges)
        elif precision == 'avg_edge_precision':
            p_edges = p.connections
            t_edges = t.connections
            num_correct += np.sum(p_edges & t_edges)
            num_total += np.sum(p_edges)
        elif precision == 'avg_edge_recall':
            p_edges = p.connections
            t_edges = t.connections
            num_correct += np.sum(p_edges & t_edges)
            num_total += np.sum(t_edges)

    return num_correct, num_total


def stateOverlap(true_seq, predicted_seq, ignore_empty=False):
    len_true = len(true_seq)
    len_pred = len(predicted_seq)
    if len_true != len_pred:
        err_str = f'{len_true} samples in true_seq != {len_pred} samples in predicted_seq'
        # raise ValueError(err_str)
        logger.warn(err_str)

    if ignore_empty:
        predicted_seq = tuple(filter(nonempty, predicted_seq))
        true_seq = tuple(filter(nonempty, true_seq))

    size_intersect = 0
    size_union = 0

    for p, t in zip(predicted_seq, true_seq):
        p_blocks = set(p.blocks.keys())
        t_blocks = set(t.blocks.keys())
        size_intersect += len(p_blocks & t_blocks)
        size_union += len(p_blocks | t_blocks)

    return size_intersect, size_union


def vertexOverlap(state1, state2):
    pass


def edgeOverlap(state1, state2):
    pass


def levenshtein(
        reference, candidate, normalized=False, segment_level=False,
        # reduce_reference=True, reduce_candidate=False,
        deletion_cost=1, insertion_cost=1, substitution_cost=1,
        return_num_elems=False, corpus=None, resolution=None):
    """ Compute the Levenshtein (edit) distance between two sequences.

    Parameters
    ----------
    reference : iterable(object)
    candidate : iterable(object)
    normalized : bool, optional
    reduce_true : bool, optional
    deletion_cost : int, optional
        Cost of deleting an element from the `candidate`.
    insertion_cost : int, optional
        Cost of inserting an element from the `reference`.
    substitution_cost : int, optional
        Cost of substituting an element in the `reference` for an element in
        the `candidate`.

    Returns
    -------
    dist : int
        NOTE: `dist` has type `float` if `normalized == True`
    """

    if corpus is None:
        def size(state, resolution):
            return 1

        def difference(state, other, resolution):
            return int(state != other)
    elif corpus == 'airplane':
        def size(state, resolution):
            if resolution == 'state':
                return 1
            elif resolution == 'block':
                return len(state.assembly_state)
            raise NotImplementedError

        def difference(state, other, resolution):
            if resolution == 'state':
                return int(state != other)
            elif resolution == 'block':
                return len(state.assembly_state ^ other.assembly_state)
            raise NotImplementedError
    elif corpus in ('easy', 'child'):
        def size(state, resolution):
            if resolution == 'state':
                return 1
            elif resolution == 'edge':
                return state.connections.sum()
            elif resolution == 'block':
                state_blocks = set(state.blocks.keys())
                return len(state_blocks)
            raise NotImplementedError

        def difference(state, other, resolution):
            if resolution == 'state':
                return int(state != other)
            elif resolution == 'edge':
                edge_diff = state.connections ^ other.connections
                return edge_diff.sum()
            elif resolution == 'block':
                state_blocks = set(state.blocks.keys())
                other_blocks = set(other.blocks.keys())
                return len(state_blocks ^ other_blocks)
            raise NotImplementedError

    if segment_level:
        reference, _ = utils.computeSegments(reference)
        candidate, _ = utils.computeSegments(candidate)

    num_true = 1 if not reference else len(reference)
    num_pred = 1 if not candidate else len(candidate)

    prefix_dists = np.zeros((num_pred, num_true), dtype=int)

    # Cost for deleting all elements of candidate
    for i in range(1, num_pred):
        candidate_size = size(candidate[i], resolution=resolution)
        prefix_dists[i, 0] = prefix_dists[i - 1, 0] + deletion_cost * candidate_size

    # Cost for inserting all elements of reference
    for j in range(1, num_true):
        reference_size = size(reference[j], resolution=resolution)
        prefix_dists[0, j] = prefix_dists[0, j - 1] + insertion_cost * reference_size

    for i in range(1, num_pred):
        for j in range(1, num_true):
            # needs_sub = int(reference[i] != candidate[j])
            candidate_size = size(candidate[i], resolution=resolution)
            reference_size = size(reference[j], resolution=resolution)
            sub_size = difference(candidate[i], reference[j], resolution=resolution)

            prefix_dists[i, j] = min(
                prefix_dists[i - 1, j] + deletion_cost * candidate_size,
                prefix_dists[i, j - 1] + insertion_cost * reference_size,
                prefix_dists[i - 1, j - 1] + substitution_cost * sub_size,
            )

    dist = prefix_dists[num_pred - 1, num_true - 1]

    if normalized:
        size_pred = sum(size(state, resolution) for state in candidate)
        size_true = sum(size(state, resolution) for state in reference)
        dist /= max(size_pred, size_true)

    if return_num_elems:
        return dist, (num_true, num_pred)

    return dist


def avgLevenshtein(true_seqs, predicted_seqs, normalized=False):
    num_true_seqs = len(true_seqs)
    num_pred_seqs = len(predicted_seqs)
    if num_true_seqs != num_pred_seqs:
        err_str = f'{num_true_seqs} ground-truth sequences but {num_pred_seqs} prediction sequences'
        raise ValueError(err_str)

    dist = 0
    for t, p in zip(true_seqs, predicted_seqs):
        dist += levenshtein(t, p, normalized=normalized)

    return dist / num_true_seqs


def blockAccuracy(true_seqs, predicted_seqs, ignore_empty=False):
    total_intersect = 0
    total_union = 0

    for p_seq, t_seq in zip(predicted_seqs, true_seqs):
        size_intersect, size_union = stateOverlap(p_seq, t_seq, ignore_empty=ignore_empty)
        total_intersect += size_intersect
        total_union += size_union

    return total_intersect / total_union, -1


def stateAccuracy(true_seqs, predicted_seqs, precision='states'):
    total_states = 0
    total_correct = 0

    num_true_seqs = len(true_seqs)
    num_pred_seqs = len(predicted_seqs)
    if num_true_seqs != num_pred_seqs:
        err_str = f'{num_true_seqs} ground-truth sequences but {num_pred_seqs} prediction sequences'
        raise ValueError(err_str)

    for p_seq, t_seq in zip(predicted_seqs, true_seqs):
        num_correct, num_states = numberCorrect(p_seq, t_seq, precision=precision)
        total_correct += num_correct
        total_states += num_states

    return total_correct / total_states, -1


def statePrecision(true_seqs, predicted_seqs, precision='states'):
    total_states = 0
    total_correct = 0

    for p_seq, t_seq in zip(predicted_seqs, true_seqs):
        num_correct, num_states = numberCorrect(
            t_seq, p_seq, ignore_empty_pred=True, precision=precision
        )
        total_correct += num_correct
        total_states += num_states

    if total_states:
        return total_correct / total_states, -1
    if total_correct:
        return np.inf, -1
    return np.nan, -1


def stateRecall(true_seqs, predicted_seqs, precision='states'):
    total_states = 0
    total_correct = 0

    for p_seq, t_seq in zip(predicted_seqs, true_seqs):
        num_correct, num_states = numberCorrect(
            t_seq, p_seq, ignore_empty_true=True, precision=precision
        )
        total_correct += num_correct
        total_states += num_states

    if total_states:
        return total_correct / total_states, -1
    if total_correct:
        return np.inf, -1
    return np.nan, -1
