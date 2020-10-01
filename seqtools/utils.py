import warnings
import collections
import logging

import numpy as np
from matplotlib import pyplot as plt


logger = logging.getLogger(__name__)


def smoothCounts(
        edge_counts, state_counts, init_states, final_states,
        init_regularizer=0, final_regularizer=0,
        uniform_regularizer=0, diag_regularizer=0,
        override_transitions=False, structure_only=False, as_numpy=False, as_scores=False):

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
    initial_states = initial_counts.nonzero()[0]
    for i in initial_states:
        bigram_counts[i, i] += init_regularizer

    final_states = final_counts.nonzero()[0]
    for i in final_states:
        bigram_counts[i, i] += final_regularizer

    bigram_counts += uniform_regularizer
    diag_indices = np.diag_indices(bigram_counts.shape[0])
    bigram_counts[diag_indices] += diag_regularizer

    if override_transitions:
        logger.info('Overriding bigram_counts with an array of all ones')
        bigram_counts = np.ones_like(bigram_counts)

    if structure_only:
        bigram_counts = (bigram_counts > 0).astype(float)
        initial_counts = (initial_counts > 0).astype(float)
        final_counts = (final_counts > 0).astype(float)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='divide by zero')
        denominator = bigram_counts.sum(1)
        transition_probs = bigram_counts / denominator[:, None]
        transition_probs[np.isnan(transition_probs)] = 0
    # FIXME: these aren't the real final probs in general
    final_probs = (final_counts > 0).astype(float)
    initial_probs = initial_counts / initial_counts.sum()

    if as_numpy:
        def to_numpy(x):
            return x.numpy().astype(float)
        return tuple(map(to_numpy, (transition_probs, initial_probs, final_probs)))

    if as_scores:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='divide by zero')
            transition_probs = np.log(transition_probs)
            initial_probs = np.log(initial_probs)
            final_probs = np.log(final_probs)

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


def plot_transitions(fn, transition_probs, initial_probs, final_probs):
    plt.figure()
    plt.matshow(transition_probs)
    plt.title('Transitions')
    plt.savefig(fn)
    plt.close()
