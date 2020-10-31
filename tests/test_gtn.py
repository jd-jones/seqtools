import os
import collections

import yaml
import numpy as np
import torch
import gtn

from mathtools import utils, metrics, torchutils
from seqtools import fstutils_gtn as libfst


def sampleGT(transition_probs, initial_probs):
    cur_state = np.random.choice(initial_probs.shape[0], p=initial_probs)
    gt_seq = [cur_state]
    while True:
        transitions = transition_probs[cur_state, :]
        cur_state = np.random.choice(transitions.shape[0], p=transitions)
        if cur_state == transitions.shape[0] - 1:
            return np.array(gt_seq)
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


def main(
        out_dir=None, gpu_dev_id=None,
        num_samples=10, random_seed=None,
        learning_rate=1e-3, num_epochs=500,
        dataset_kwargs={}, dataloader_kwargs={}, model_kwargs={}):

    if out_dir is None:
        out_dir = os.path.join('~', 'data', 'output', 'seqtools', 'test_gtn')

    out_dir = os.path.expanduser(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    vocabulary = ['a', 'b', 'c', 'd', 'e']

    transition = np.array(
        [[0, 1, 0, 0, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 0, 0, 1],
         [0, 1, 0, 0, 1],
         [0, 0, 0, 0, 0]], dtype=float
    )
    initial = np.array([1, 0, 1, 0, 0], dtype=float)
    final = np.array([0, 1, 0, 0, 1], dtype=float) / 10

    seq_params = (transition, initial, final)
    simulated_dataset = simulate(num_samples, *seq_params)
    label_seqs, obsv_seqs = tuple(zip(*simulated_dataset))
    seq_params = tuple(map(lambda x: -np.log(x), seq_params))

    dataset = torchutils.SequenceDataset(obsv_seqs, label_seqs, **dataset_kwargs)
    data_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    train_loader = data_loader
    val_loader = data_loader

    transition_weights = torch.tensor(transition, dtype=torch.float).log()
    initial_weights = torch.tensor(initial, dtype=torch.float).log()
    final_weights = torch.tensor(final, dtype=torch.float).log()

    model = libfst.LatticeCrf(
        vocabulary,
        transition_weights=transition_weights,
        initial_weights=initial_weights, final_weights=final_weights,
        debug_output_dir=fig_dir,
        **model_kwargs
    )

    gtn.draw(
        model._transition_fst, os.path.join(fig_dir, 'transitions-init.png'),
        isymbols=model._arc_symbols, osymbols=model._arc_symbols
    )

    # Train the model
    train_epoch_log = collections.defaultdict(list)
    val_epoch_log = collections.defaultdict(list)
    metric_dict = {
        'Avg Loss': metrics.AverageLoss(),
        'Accuracy': metrics.Accuracy()
    }

    criterion = model.nllLoss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.00)

    model, last_model_wts = torchutils.trainModel(
        model, criterion, optimizer, scheduler, train_loader,
        val_loader,
        metrics=metric_dict,
        test_metric='Avg Loss',
        train_epoch_log=train_epoch_log,
        val_epoch_log=val_epoch_log,
        num_epochs=num_epochs
    )

    model._transition_fst.set_weights(model._params['transition'].tolist())
    gtn.draw(
        model._transition_fst, os.path.join(fig_dir, 'transitions-trained.png'),
        isymbols=model._arc_symbols, osymbols=model._arc_symbols
    )

    torchutils.plotEpochLog(
        train_epoch_log, title="Train Epoch Log",
        fn=os.path.join(fig_dir, "train-log.png")
    )


if __name__ == "__main__":
    # Parse command-line args and config file
    cl_args = utils.parse_args(main)
    config, config_fn = utils.parse_config(cl_args, script_name=__file__)

    # Create output directory, instantiate log file and write config options
    out_dir = os.path.expanduser(config['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    main(**config)
