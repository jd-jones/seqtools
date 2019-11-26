import os
import collections
import argparse

import yaml
import numpy as np
import torch

from seqtools import fsm, torchutils, metrics, utils


def sampleMarkov(start_probs, end_probs, transition_probs):
    samples = []
    transition_dist = start_probs
    while True:
        new_sample = np.random.choice(len(transition_dist), p=transition_dist)
        samples.append(new_sample)
        transition_dist = transition_probs[new_sample]

        end_dist = [1 - end_probs[new_sample], end_probs[new_sample]]
        if np.random.choice(2, p=end_dist):
            return np.array(samples, dtype=int)


def sampleGHMM(start_probs, end_probs, transition_probs, means, covs):
    label_seq = sampleMarkov(start_probs, end_probs, transition_probs)
    obsv_seq = np.array(
        np.row_stack(tuple(np.random.multivariate_normal(means[l], covs[l]) for l in label_seq))
    )
    return obsv_seq, label_seq


def main(
        out_dir=None, gpu_dev_id=None,
        sample_size=10, random_seed=None,
        learning_rate=1e-3, num_epochs=500,
        dataset_kwargs={}, dataloader_kwargs={}, model_kwargs={}):

    out_dir = os.path.expanduser(out_dir)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torchutils.selectDevice(gpu_dev_id)

    start_probs = np.array([0, 0.25, 0.75])
    end_probs = np.array([0.05, 0, 0])
    transition_probs = np.array([
        [0.8, 0.2, 0.0],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ])

    means = np.array([
        [0],
        [1],
        [2]
    ])
    covs = np.array([
        [[0.01]],
        [[0.01]],
        [[0.01]]
    ])

    obsv_seqs, label_seqs = zip(
        *tuple(
            sampleGHMM(start_probs, end_probs, transition_probs, means, covs)
            for _ in range(sample_size)
        )
    )

    dataset = torchutils.SequenceDataset(obsv_seqs, label_seqs, **dataset_kwargs)
    data_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    # class CRF(fsm.FstScorer, torchutils.LinearClassifier):
    #     pass

    class CRF(torchutils.LinearChainScorer, torchutils.LinearClassifier):
        pass

    train_loader = data_loader
    val_loader = data_loader

    input_dim = dataset.num_obsv_dims
    output_dim = transition_probs.shape[0]

    transition_weights = torch.randn(transition_probs.shape).to(device)
    # transition_weights = torch.tensor(transition_probs, device=device).float().log()

    model = CRF(
        transition_weights, input_dim, output_dim,
        initial_weights=None, final_weights=None,
        **model_kwargs
    )

    # Train the model
    train_epoch_log = collections.defaultdict(list)
    val_epoch_log = collections.defaultdict(list)
    metric_dict = {
        'Avg Loss': metrics.AverageLoss(),
        'Accuracy': metrics.Accuracy()
    }

    # criterion = fsm.fstNLLLoss
    criterion = torchutils.StructuredNLLLoss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.00)

    model, last_model_wts = torchutils.trainModel(
        model, criterion, optimizer, scheduler, train_loader,
        val_loader,
        metrics=metric_dict,
        test_metric='Avg Loss',
        train_epoch_log=train_epoch_log,
        val_epoch_log=val_epoch_log,
        num_epochs=num_epochs,
        device=device,
    )

    torchutils.plotEpochLog(
        train_epoch_log, title="Train Epoch Log",
        fn=os.path.join(out_dir, "train-log.png")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.expanduser(
            os.path.join('~', 'repo', 'seqtools', 'tests', config_fn)
        )
    with open(config_file_path, 'rt') as config_file:
        config = yaml.safe_load(config_file)
    config.update(args)

    # Create output directory, instantiate log file and write config options
    out_dir = os.path.expanduser(config['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    utils.autoreload_ipython()

    main(**config)
