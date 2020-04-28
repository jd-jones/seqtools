import torch
# from numba import float64, jit, int16, boolean, int64, autojit
# from numba import jit

from mathtools import utils


def log_prob(data_scores, y, max_segs, pw=None):
    # Scores has shape (num_segs, num_samples, num_classes),
    # so marginalizing over number of segments and number of classes at the last
    # frame should give us the data likelihood... I think :/
    scores = segmental_forward(data_scores, max_segs, pw=pw, semiring='log')
    log_Z = torch.logsumexp(scores[-1, :], -1)
    if torch.isnan(log_Z).any():
        raise ValueError("log_Z contains NaN values")

    y_score = segmental_score(data_scores, y, pw=pw)
    if torch.isnan(y_score).any():
        raise ValueError("y_score contains NaN values")

    return -y_score - log_Z


def segmental_score(data_scores, y, pw=None):
    score = 0
    start_index = 0
    prev_label = None
    for seg_label, seg_len in utils.genSegments(y):
        next_start_index = start_index + seg_len + 1
        score += data_scores[start_index:next_start_index, seg_label].sum()
        if start_index > 0:
            score += pw[prev_label, seg_label]

        start_index = next_start_index
        prev_label = seg_label

    return score


def segmental_forward(x, max_dur, pw=None):
    # From S&C NIPS 2004
    T, n_classes = x.shape
    scores = torch.full([T, n_classes], -float("Inf"), dtype=x.dtype, device=x.device)
    # classes_prev = torch.ones([T, n_classes], np.int)
    if pw is None:
        pw = torch.zeros([n_classes, n_classes], dtype=x.dtype)

    # initialize first segment scores
    integral_scores = torch.cumsum(x, 0)
    scores[0] = integral_scores[0]

    def dur_score(t_end, duration, c):
        t_start = t_end - duration
        current_segment = integral_scores[t_end, c] - integral_scores[t_start, c]

        # Elementwise semiring times
        dur_scores = scores[t_start, :] + current_segment + pw[:, c]

        # Reduction: semiring plus
        # FIXME: change max to logsumexp
        return dur_scores.max()

    # Compute scores per timestep
    for t_end in range(1, T):
        # Compute scores per class
        for c in range(n_classes):
            # Compute over all durations
            best_dur_scores = torch.tensor(
                [
                    dur_score(t_end, duration, c)
                    for duration in range(1, min(t_end, max_dur) + 1)
                ]
            )

            # FIXME: change max to logsumexp
            best_score = best_dur_scores.max()

            # Add cost of curent frame to best previous cost
            scores[t_end, c] = best_score

    return scores


# @jit("float64[:,:](float64[:,:], int16, float64[:,:])")
def segmental_viterbi(x, max_dur, pw=None):
    # From S&C NIPS 2004
    T, n_classes = x.shape
    scores = torch.full([T, n_classes], -float("Inf"), dtype=x.dtype, device=x.device)
    lengths = torch.ones([T, n_classes], dtype=torch.long)
    # classes_prev = torch.ones([T, n_classes], np.int)
    if pw is None:
        pw = torch.zeros([n_classes, n_classes], dtype=x.dtype)

    # initialize first segment scores
    integral_scores = torch.cumsum(x, 0)
    scores[0] = integral_scores[0]

    # -------- Forward -----------
    # Compute scores per timestep
    for t_end in range(1, T):
        # Compute scores per class
        for c in range(n_classes):
            # Compute over all durations
            best_dur = 0
            best_score = -float("Inf")
            # best_class = -1
            for duration in range(1, min(t_end, max_dur) + 1):
                t_start = t_end - duration
                current_segment = integral_scores[t_end, c] - integral_scores[t_start, c]

                if t_start == 0 and current_segment > best_score:
                    best_dur = duration
                    best_score = current_segment
                    # best_class = -1
                    continue

                # Check if it is cheaper to create a new segment or stay in same class
                for c_prev in range(n_classes):
                    if c_prev == c:
                        continue

                    # Previous segment, other class
                    tmp = scores[t_start, c_prev] + current_segment + pw[c_prev, c]
                    if tmp > best_score:
                        best_dur = duration
                        best_score = tmp
                        # best_class = c_prev

            # Add cost of curent frame to best previous cost
            scores[t_end, c] = best_score
            lengths[t_end, c] = best_dur
            # classes_prev[t_end, c] = best_class

    # Set nonzero entries to 0 for visualization
    # scores[scores<0] = 0
    scores[torch.isinf(scores)] = 0

    # -------- Backward -----------
    classes = [scores[-1].argmax()]
    times = [T]
    t = T - lengths[-1, classes[-1]]
    while t > 0:
        class_prev = scores[t].argmax()
        length = lengths[t, class_prev]
        classes.insert(0, class_prev)
        times.insert(0, t)
        t -= length

    y_out = torch.zeros(T, torch.long)
    t = 0
    for c, l in zip(classes, times):
        y_out[t:t + l] = c
        t += l

    return scores


# @jit("float64[:,:](float64[:,:], int16, float64[:,:])")
def segmental_forward_normalized(x, max_segs, pw=None):
    """ This version maximizes!!! """
    # Assumes segment function is normalized by duration: f(x)= 1/d sum_t'=t^t+d x_t'
    T, n_classes = x.shape
    scores = torch.full([max_segs, T, n_classes], -float("Inf"), dtype=x.dtype, device=x.device)
    if pw is None:
        pw = torch.zeros([n_classes, n_classes], dtype=x.dtype)

    integral_scores = torch.cumsum(x, 0)

    # Intial scores
    scores[0] = integral_scores.copy()
    starts = torch.zeros([max_segs, n_classes], torch.long) + 1

    # Compute scores for each segment in sequence
    for m in range(1, max_segs):
        # Compute score for each class
        for c in range(n_classes):
            best_score = -float("Inf")
            for c_prev in range(n_classes):
                if c_prev == c:
                    continue

                # Compute scores for each timestep
                for t in range(1, T):
                    new_segment = integral_scores[t, c] - integral_scores[starts[m, c], c]

                    # Previous segment, other class
                    score_change = scores[m - 1, t, c_prev] + pw[c_prev, c]
                    if score_change > best_score:
                        best_score = score_change
                        starts[m, c] = t

                        # Add cost of curent frame to best previous cost
                        scores[m, t, c] = best_score + new_segment

        # Set nonzero entries to 0 for visualization
        scores[torch.isinf(scores)] = 0

        return scores


def sparsify_incoming_pw(pw):
    # Output is INCOMING transitions
    n_classes = pw.shape[0]
    valid = torch.nonzero(~torch.isinf(pw.T), as_tuple=True)  # requires pytorch 1.3
    sparse_idx = [[] for i in range(n_classes)]
    for i, j in zip(valid[0], valid[1]):
        sparse_idx[i] += [j]

    return sparse_idx


def log_prob_eccv(data_scores, y, max_segs, pw=None):
    # Scores has shape (num_segs, num_samples, num_classes),
    # so marginalizing over number of segments and number of classes at the last
    # frame should give us the data likelihood... I think :/
    scores = segmental_forward_eccv(data_scores, max_segs, pw=pw, semiring='log')
    log_Z = torch.logsumexp(torch.logsumexp(scores[:, -1, :], 0), -1)
    if torch.isnan(log_Z).any():
        raise ValueError("log_Z contains NaN values")

    y_score = segmental_score(data_scores, y, pw=pw)
    if torch.isnan(y_score).any():
        raise ValueError("y_score contains NaN values")

    return -y_score - log_Z


# @jit("float64[:,:](float64[:,:], int16, float64[:,:])")
def segmental_forward_eccv(x, max_segs, pw=None, semiring='tropical'):
    if torch.isnan(x).any():
        raise ValueError("x contains NaN values")

    if semiring == 'tropical':
        def sr_prod(x, y):
            return x + y

        def sr_sum(x):
            return x.max()
    elif semiring == 'log':
        def sr_prod(x, y):
            return x + y

        def sr_sum(x):
            return torch.logsumexp(x, 0)
    else:
        raise AssertionError()

    # Assumes segment function is additive: f(x)=sum_t'=t^t+d x_t'
    T, n_classes = x.shape
    scores = torch.full([max_segs, T, n_classes], -float("Inf"), dtype=x.dtype, device=x.device)
    if pw is None:
        pw = torch.log(1 - torch.eye(n_classes))
        # print("pw is None: Using uniform transition weights (no self-loops)")

    # initialize first segment scores
    scores[0] = torch.cumsum(x, 0)

    # Compute scores per segment
    for m in range(1, max_segs):
        # Compute scores per timestep
        for t in range(1, T):
            # Compute scores per class
            for c in range(n_classes):
                # Elementwise semiring times
                new_scores = torch.cat(
                    (scores[m, t - 1, c:c + 1], sr_prod(scores[m - 1, t - 1, :], pw[:, c]))
                )
                # Reduction: semiring plus
                best_prev = sr_sum(new_scores)

                # Add cost of curent frame to best previous cost
                scores[m, t, c] = sr_prod(best_prev, x[t, c])

    if torch.isnan(scores).any():
        raise ValueError("scores contains NaN values")

    return scores


# @jit("int16[:,:](float64[:,:], float64[:,:])")
def segmental_backward_eccv(scores, pw=None):
    n_segs, T, n_classes = scores.shape

    if pw is None:
        pw = torch.log(1 - torch.eye(n_classes))
        # print("pw is None: Using uniform transition weights (no self-loops)")

    best_scores = scores[:, -1].max(1).values
    n_segs = torch.argmax(best_scores)

    # Start at end
    seq_c = [scores[n_segs, -1].argmax()]  # Class
    seq_t = [T]  # Time
    m = n_segs

    for t in range(T, -1, -1):
        if m == 0:
            break

        # Scores of previous timestep in current segment
        score_same = scores[m, t - 1, seq_c[0]]
        score_diff = scores[m - 1, t - 1] + pw[:, seq_c[0]]

        # Check if it's better to stay or switch segments
        if any(score_diff > score_same):
            next_class = score_diff.argmax()
            score_diff = score_diff[next_class]
            seq_c.insert(0, next_class)
            seq_t.insert(0, t)
            m -= 1
        elif all(score_diff == score_same):
            m -= 1

    seq_t.insert(0, 0)

    if m != 0:
        raise AssertionError("Found " + str(m) + " segments, but expected zero!")

    y_out = torch.full((T,), -1, dtype=torch.long, device=scores.device)
    for i in range(len(seq_c)):
        y_out[seq_t[i]:seq_t[i + 1]] = seq_c[i]

    return y_out


def segmental_inference(x, max_segs, pw=None, normalized=False, verbose=False, return_scores=False):
    # Scores has shape (num_segs, num_samples, num_classes)
    scores = segmental_forward_eccv(x, max_segs, pw)
    y_out = segmental_backward_eccv(scores, pw)

    if return_scores:
        num_segs = len(utils.segment_labels(y_out))
        y_idxs = torch.arange(y_out.numel)
        y_scores = scores[num_segs, y_idxs, y_out]
        return y_out, y_scores
    return y_out


# @jit("float64[:,:](float64[:,:], int16, float64[:,:], float64[:], float64[:,:])")
def segmental_forward_oracle(x, max_segs, pw, y_oracle, oracle_valid):
    # Assumes segment function is additive: f(x)=sum_t'=t^t+d x_t'
    T, n_classes = x.shape
    scores = torch.full([max_segs, T, n_classes], -float("Inf"), dtype=x.dtype, device=x.device)
    lengths = torch.zeros([max_segs, T, n_classes], torch.long)
    if pw is None:
        pw = torch.log(1 - torch.eye(n_classes))

    # initialize first segment scores
    scores[0] = torch.cumsum(x, 0)

    # Compute scores per segment
    for m in range(1, max_segs):
        # scores[m, 0, c] = scores[m-1, 0, c]
        # Compute scores per timestep
        for t in range(1, T):
            # Compute scores per class
            for c in range(n_classes):
                # Score for staying in same segment
                best_prev = scores[m, t - 1, c]
                length = lengths[m, t - 1, c] + 1

                # Check if it is cheaper to create a new segment or stay in same class
                for c_prev in range(n_classes):
                    # Previous segment, other class
                    tmp = scores[m - 1, t - 1, c_prev] + pw[c_prev, c]
                    if tmp > best_prev:
                        best_prev = tmp
                        length = 1

                if oracle_valid[y_oracle[t], c] == 0:
                    best_prev = -float("Inf")

                # Add cost of curent frame to best previous cost
                scores[m, t, c] = best_prev + x[t, c]
                lengths[m, t, c] = length

    # Set nonzero entries to 0 for visualization
    scores[torch.isinf(scores)] = 0

    return scores


def segmental_inference_oracle(x, max_segs, pw, y_oracle, oracle_valid):
    scores = segmental_forward_oracle(x, max_segs, pw, y_oracle, oracle_valid)
    return segmental_backward_eccv(scores, pw)
