import logging
import time
import copy
import collections
import os

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn, cuda
from torch.utils.data import Dataset
import torch_struct


logger = logging.getLogger(__name__)


# -=( MISC )=------------------------------------------------------------------
def isscalar(x):
    return x.shape == torch.Size([])


def isreal(x):
    """ Check if a pytorch tensor has real-valued type.

    This function is used to extend the classes defined in mfst.semirings to
    wrap pytorch variables.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    is_real : bool
    """

    real_valued_types = (
        torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor,
        torch.CharTensor, torch.ShortTensor, torch.IntTensor, torch.LongTensor,
        cuda.FloatTensor, cuda.DoubleTensor, cuda.HalfTensor,
        cuda.CharTensor, cuda.ShortTensor, cuda.IntTensor, cuda.LongTensor,
    )

    return isinstance(x, real_valued_types)


def selectDevice(gpu_dev_id):
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'

    if gpu_dev_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_dev_id

    num_visible_gpus = torch.cuda.device_count()
    if num_visible_gpus > 0:
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device('cpu')
        device_name = 'CPU'

    logger.info(f'{num_visible_gpus} GPU(s) visible to script.')
    logger.info(f'Selected device: {device_name}')

    return device


# -=( TRAINING & EVALUATION )=-------------------------------------------------
def guessStructure(log_potentials):
    """ Guess the structure associated with a table of log potentials.

    This structure is used in dynamic programs for training and decoding.

    Parameters
    ----------
    log_potentials : torch.Tensor of float

    Returns
    -------
    structure : torch_struct.StructDistribution, optional
        This functions returns None of there is no structure in the output
        (i.e. outputs are independent).
    """

    # Ignore axis 0 (batch size) when examining the output shape
    table_shape = log_potentials.shape[1:]
    num_table_dims = len(table_shape)

    if num_table_dims in (1, 2):
        return None
    if num_table_dims == 3 and table_shape[-2] == table_shape[-1]:
        return torch_struct.LinearChainCRF
    elif num_table_dims == 4 and table_shape[-2] == table_shape[-1]:
        return torch_struct.SemiMarkovCRF

    err_str = f"Can't guess structure for input with shape {log_potentials.shape}"
    raise ValueError(err_str)


def StructuredNLLLoss(log_potentials, label, lengths=None, structure=None):
    """ Efficiently compute structured negative-log-likelihood loss using dynamic programming.

    Parameters
    ----------
    log_potentials : torch.Tensor of float
        Shape depends on `structure`.
    label : torch.Tensor of int, shape (batch_size, num_samples)
    structure : torch_struct.StructDistribution, optional
        If this argument is omitted, the function will try to choose a CRF-style
        structure that is compatible with the shape of the input.

    Returns
    -------
    loss : float
    """

    num_classes = log_potentials.shape[-1]

    if structure is None:
        structure = guessStructure(log_potentials)

    dist = structure(log_potentials, lengths=lengths)

    label_events = (
        structure.struct.to_parts(label, num_classes, lengths=lengths)
        .type_as(dist.log_potentials)
    )
    loss = -dist.log_prob(label_events).sum()

    return loss


def fillSegments(pred_batch, in_place=False):
    """ Fill the interior of segments decoded by torch_struct.SemiMarkovCRF

    Parameters
    ----------
    pred_batch :
    in_place : bool, optional
    """

    if not in_place:
        raise NotImplementedError()

    for sample_index in range(pred_batch.shape[0]):
        segment_bounds = (pred_batch[sample_index] != -1).nonzero().squeeze()
        for seg_start, next_seg_start in zip(segment_bounds[:-1], segment_bounds[1:]):
            seg_label = pred_batch[sample_index, seg_start]
            pred_batch[sample_index, seg_start:next_seg_start] = seg_label


def predictBatch(model, batch_input, device=None, as_numpy=False, structure=None):
    """ Use a model to make predictions from a minibatch of data.

    Parameters
    ----------
    model : torch.model
        Model to use when predicting from input. This model should return a
        torch array of scores when called, e.g. ``scores = model(input)``.
    batch_input : torch.Tensor, shape (batch_size, seq_length)
        Input to model.
    device : torch.device, optional
        Device to use when processing data with the model.
    as_numpy : bool, optional
        If True, return a numpy.array instead of torch.Tensor. False by default.
    structure : torch_struct.StructDistribution, optional
        If this argument is omitted, the function will try to choose a structure
        that is compatible with the shape of the output of ``model.forward()``.

    Returns
    -------
    preds : torch.Tensor or numpy.array of int, shape (batch_size, seq_length, num_model_states)
        Model predictions. Each entry is the index of the model output with the
        highest activation.
    outputs : torch.Tensor or numpy.array of float, shape (batch_size, seq_length)
        Model outputs. Each entry represents the model's activation for a
        particular hypothesis. Higher numbers are better.
    """

    batch_input = batch_input.to(device=device)

    outputs = model(batch_input)

    if structure is None:
        structure = guessStructure(outputs)

    if structure is None:
        __, preds = torch.max(outputs, 1)
    else:
        dist = structure(outputs)
        preds, _ = dist.struct.from_parts(dist.argmax)
        if (preds == -1).any():
            fillSegments(preds, in_place=True)
        # The first predicted label is spurious---it's the state from which the
        # initial state transitioned.
        preds = preds[:, 1:]

    if as_numpy:
        preds = preds.cpu().numpy()
        outputs = outputs.cpu().detach().numpy()

    return preds, outputs


def predictSamples(
        model, data_loader,
        criterion=None, optimizer=None, scheduler=None, data_labeled=False,
        update_model=False, device=None, update_interval=None,
        num_minibatches=None, metrics=None, return_io_history=False):
    """ Use a model to predict samples from a dataset; can also update model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to use when making predictions. If `criterion` is not None, the
        parameters of this model are also updated using back-propagation.
    data_loader : torch.utils.data.DataLoader
    criterion : torch.nn.Modules._Loss, optional
        Loss function to use when training or evaluating `model`. If
        `update_model` is True, the loss function is used to update the model.
        Otherwise the loss function is only used to evaluate the model.
    optimizer : torch.optim.Optimizer, optional
        This argument is ignored if `train_model` is False.
    scheduler : torch.optim.LR_scheduler._LRscheduler, optional
    data_labeled : bool, optional
        If True, a sample from ``data_loader`` is the tuple ``data, labels``.
        If False, a sample from ``data_loader`` is just ``data``.
        Default is False.
    update_model : bool, optional
        If False, this function only makes predictions and does not update
        parameters.
    device : torch.device, optional
        Device to perform computations on. Useful if your machine  has GPUs.
        Default is CPU.
    update_interval : int, optional
        This functions logs progress updates every `update_interval` minibatches.
        If `update_interval` is None, no updates will be logged. Default is None.
    num_minibatches : int, optional
        Maximum number of minibatches to evaluate. If `num_minibatches` is None,
        this function iterates though all the samples in `data_loader`.
    metrics : iterable( metrics.RationalPerformanceMetric ), optional
        Performance metrics to use when evaluating the model. Examples include
        accuracy, precision, recall, and F-measure.
    return_io_history : bool, optional
        If True, this function returns a list summarizing the model's input-output
        history. See return value `io_history` for further documentation.

    Returns
    -------
    io_history : iterable( (torch.Tensor, torch.Tensor, torch.Tensor) )
        `None` if ``return_io_history == False``.
    """

    if device is None:
        device = torch.device('cpu')

    if metrics is None:
        metrics = {}
    for m in metrics.values():
        m.initializeCounts()

    if update_model:
        scheduler.step()
        model.train()
    else:
        model.eval()

    io_history = None
    if return_io_history:
        io_history = []

    with torch.set_grad_enabled(update_model):
        for i, sample in enumerate(data_loader):
            if num_minibatches is not None and i > num_minibatches:
                break

            if update_interval is not None and i % update_interval == 0:
                logger.info(f'Predicted {i} minibatches...')

            if data_labeled:
                inputs, labels = sample
            else:
                inputs = sample

            preds, scores = predictBatch(model, inputs, device=device)

            if criterion is not None:
                labels = labels.to(device=device)
                loss = criterion(scores, labels)
                if update_model:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss = loss.item()
            else:
                loss = 0

            if return_io_history:
                batch_io = (preds,) + tuple(sample)
                io_history.append(batch_io)

            if metrics:
                preds = preds.cpu().numpy().squeeze()
                labels = labels.cpu().numpy().squeeze()

            for key, value in metrics.items():
                metrics[key].accumulate(preds, labels, loss)

    return io_history


def trainModel(
        model, criterion, optimizer, scheduler, train_loader, val_loader=None,
        num_epochs=25, train_epoch_log=None, val_epoch_log=None, device=None,
        update_interval=None, metrics=None, test_metric=None, improvement_thresh=0):
    """ Train a PyTorch model.

    Parameters
    ----------
    model :
    criterion :
    optimizer :
    scheduler :
    train_loader :
    val_loader :
    num_epochs :
    train_epoch_log :
    val_epoch_log :
    device :
    update_interval :
    metrics :
    test_metric :

    Returns
    -------
    model :
    last_model_wts :
    """

    logger.info('TRAINING NN MODEL')

    if metrics is None:
        metrics = {}

    if train_epoch_log is None:
        train_epoch_log = collections.defaultdict(list)
    if val_epoch_log is None:
        val_epoch_log = collections.defaultdict(list)

    if device is None:
        device = torch.device('cpu')
    model = model.to(device=device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric = 0.0

    init_time = time.time()
    for epoch in range(num_epochs):
        logger.info('EPOCH {}/{}'.format(epoch + 1, num_epochs))

        metrics = copy.deepcopy(metrics)
        _ = predictSamples(
            model, train_loader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            device=device, metrics=metrics, data_labeled=True,
            update_model=True
        )

        for metric_name, metric in metrics.items():
            train_epoch_log[metric_name].append(metric.evaluate())

        metric_str = '  '.join(str(m) for m in metrics.values())
        logger.info('[TRN]  ' + metric_str)

        if val_loader is not None:
            metrics = copy.deepcopy(metrics)
            _ = predictSamples(
                model, val_loader,
                criterion=criterion, device=device,
                metrics=metrics, data_labeled=True, update_model=False,
            )

            for metric_name, metric in metrics.items():
                val_epoch_log[metric_name].append(metric.evaluate())

            metric_str = '  '.join(str(m) for m in metrics.values())
            logger.info('[VAL]  ' + metric_str)

        if test_metric is not None:
            test_val = metrics[test_metric].evaluate()
            improvement = test_val - best_metric
            if improvement > 0:
                best_metric = test_val
                best_model_wts = copy.deepcopy(model.state_dict())
                improvement_str = f'improved {test_metric} by {improvement:.5f}'
                logger.info(f'Updated best model: {improvement_str}')

    time_str = makeTimeString(time.time() - init_time)
    logger.info(time_str)

    last_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)

    return model, last_model_wts


def plotEpochLog(epoch_log, subfig_size=None, title=''):
    num_plots = len(epoch_log)

    figsize = subfig_size[0], subfig_size[1] * num_plots
    fig, axes = plt.subplots(num_plots, figsize=figsize, sharex=True)

    for i, (name, val) in enumerate(epoch_log.items()):
        axis = axes[i]
        axis.plot(val, '-o')
        axis.set_ylabel(name)
    axes[0].set_title(title)
    axes[-1].set_xlabel('Epoch index')
    plt.tight_layout()


def makeTimeString(time_elapsed):
    mins_elapsed = time_elapsed // 60
    secs_elapsed = time_elapsed % 60
    time_str = f'Training complete in {mins_elapsed:.0f}m {secs_elapsed:.0f}s'
    return time_str


# -=( DATASETS )=--------------------------------------------------------------
class SequenceDataset(Dataset):
    """ A dataset wrapping sequences of numpy arrays stored in memory.

    Attributes
    ----------
    _data : tuple(np.ndarray, shape (num_samples, num_dims))
    _labels : tuple(np.ndarray, shape (num_samples,))
    _device : torch.Device
    """

    def __init__(self, data, labels, device=None, labels_type=None, sliding_window_args=None):
        """
        Parameters
        ----------
        data : iterable( array_like of float, shape (sequence_len, num_dims) )
        labels : iterable( array_like of int, shape (sequence_len,) )
        device :
        labels_type : string in {'float'}, optional
            If passed, labels will be converted to float type.
        sliding_window_args : tuple(int, int, int), optional
            A tuple specifying parameters for extracting sliding windows from
            the data sequences. This should be ``(dimension, size, step)``---i.e.
            the input to ``torch.unfold``. The label of each sliding window is
            taken to be the median over the labels in that window.
        """

        self.num_obsv_dims = data[0].shape[1]

        if len(labels[0].shape) == 2:
            self.num_label_types = labels[0].shape[1]
        elif len(labels[0].shape) < 2:
            self.num_label_types = np.unique(np.hstack(labels)).shape[0]
        else:
            err_str = f"Labels have a weird shape: {labels[0].shape}"
            raise ValueError(err_str)

        self.sliding_window_args = sliding_window_args

        self._device = device
        self._labels_type = labels_type

        self._data = data
        self._labels = labels

        logger.info('Initialized ArrayDataset.')
        logger.info(
            f"Data has dimension {self.num_obsv_dims}; "
            f"{self.num_label_types} unique labels"
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        data_seq = torch.tensor(self._data[i], dtype=torch.float)
        label_seq = torch.tensor(self._labels[i])

        if self._device is not None:
            data_seq = data_seq.to(device=self._device)
            label_seq = label_seq.to(device=self._device)

        if self._labels_type == 'float':
            label_seq = label_seq.float()

        # Convert input from [0, 1] --> [-1, 1]
        # FIXME: don't do this for every dataset---only JIGSAWS
        # data_seq = 2 * (data_seq - 0.5)

        # shape (sequence_len, num_dims) --> (num_dims, sequence_len)
        data_seq = data_seq.transpose(0, 1)

        if self.sliding_window_args is not None:
            # Unfold gives shape (sequence_len, window_len);
            # after transpose, data_seq has shape (window_len, sequence_len)
            data_seq = data_seq.unfold(*self.sliding_window_args).transpose(-1, -2)
            label_seq = label_seq.unfold(*self.sliding_window_args).median(dim=-1).values

        return data_seq.contiguous(), label_seq.contiguous()


class ArrayDataset(Dataset):
    """ A dataset wrapping numpy arrays stored in memory.

    Attributes
    ----------
    _data : torch.Tensor, shape (num_samples, num_dims)
    _labels : torch.Tensor, shape (num_samples,)
    _device : torch.Device
    """

    def __init__(self, data, labels, device=None, labels_type=None):
        self.num_obsv_dims = data.shape[1]

        if len(labels.shape) == 2:
            self.num_label_types = labels.shape[1]
        elif len(labels.shape) == 1:
            self.num_label_types = np.unique(labels).shape[0]
        else:
            err_str = f"Labels have a weird shape: {labels.shape}"
            raise ValueError(err_str)

        self._device = device

        data = torch.tensor(data, dtype=torch.float)
        labels = torch.tensor(labels)

        if self._device is not None:
            data = data.to(device=self._device)
            labels = labels.to(device=self._device)

        if labels_type == 'float':
            labels = labels.float()

        self._data = data
        self._labels = labels

        logger.info('Initialized ArrayDataset.')
        logger.info(
            f"Data has dimension {self.num_obsv_dims}; "
            f"{self.num_label_types} unique labels"
        )

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, i):
        data = self._data[i]
        label = self._labels[i]

        return data, label


# -=( MODELS )=----------------------------------------------------------------
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, out_set_size):
        super().__init__()
        self.input_dim = input_dim
        self.out_set_size = out_set_size
        self.linear = nn.Linear(self.input_dim, self.out_set_size)
        logger.info(
            f'Initialized linear classifier. '
            f'Input dim: {self.input_dim}, Output dim: {self.out_set_size}'
        )

    def forward(self, input_seq):
        output_seq = self.linear(input_seq)
        return output_seq


class SemiMarkovScorer(object):
    """ Pytorch mixin that creates a semi-Markov (segmental) score table.

    NOTE: Segment scores are implemented as a sum over the data scores of each
        frame in the segment. However, you can implement more complex scoring
        functions by inheriting from SemiMarkovScorer and overriding `scoreSegment`,
        or by monkey-patching it at runtime if you're a real daredevil.

    Attributes
    ----------
    transition_weights : torch.Tensor of float, shape (num_classes, num_classes)
        Element (i, j) corresponds to (cur segment == i, prev segment == j);
        i.e. the transpose of how a HMM's transition array is usually arranged.
    start_weights : torch.Tensor of float, shape (num_classes,), optional
    end_weights : torch.Tensor of float, shape (num_classes,), optional
    max_duration : int, optional
        If the duration is not bounded, the runtime complexity of most inference
        algorithms is quadratic in the sequence length instead of linear (times
        a constant factor), so it's better to bound the duration if you can.
    """

    def __init__(
            self, *super_args, transitions=None, initial_states=None, final_states=None,
            max_duration=None, update_transition_params=False,
            **super_kwargs):
        """
        Parameters
        ----------
        transitions : torch.Tensor of float, shape (num_classes, num_classes), optional
            Element (i, j) corresponds to (cur segment == i, prev segment == j);
            i.e. the transpose of how a HMM's transition array is usually arranged.
        initial_states : torch.Tensor of float, shape (num_classes,), optional
        final_states : torch.Tensor of float, shape (num_gestures,), optional
        update_transition_params : bool, optional
            If True, start, end, and transition weights will be updated during
            training.
        max_duration : int, optional
            Default is `num_samples`---note that this makes the runtime complexity of
            most inference algorithms scale quadratically with the sequence length
            instead of linearly, so it's better to bound the duration if you can
            (and you probably can).
        *super_args, **super_kwargs : optional
            Remaining arguments are passed to super's init method.
        """

        super().__init__(*super_args, **super_kwargs)

        transition_weights = torch.tensor(transitions).float().log()
        start_weights = torch.tensor(initial_states).float().log()
        end_weights = torch.tensor(final_states).float().log()

        self.transition_weights = torch.nn.Parameter(
            transition_weights, requires_grad=update_transition_params
        )
        self.start_weights = torch.nn.Parameter(
            start_weights, requires_grad=update_transition_params
        )
        self.end_weights = torch.nn.Parameter(
            end_weights, requires_grad=update_transition_params
        )
        self.max_duration = max_duration

    def forward(self, attribute_scores):
        """ Construct a table of semi-Markov (i.e. segmental) scores.

        These scores can be used to instantiate pytorch-struct's SemiMarkovCRF.

        Parameters
        ----------
        attribute_scores : torch.Tensor of float, shape (batch_size, ..., num_samples)
            NOTE: This method doesn't handle inputs with batch size > 1 (yet).

        Returns
        -------
        log_potentials : torch.Tensor of float,
                shape (batch_size, num_samples, max_duration, num_classes, num_classes)
        """

        batch_size = attribute_scores.shape[0]
        num_samples = attribute_scores.shape[-1]
        num_labels = self.start_weights.shape[0]
        max_duration = self.max_duration
        if max_duration is None:
            max_duration = num_samples

        scores_shape = (batch_size, num_samples, max_duration, num_labels, num_labels)
        log_potentials = torch.full(scores_shape, -float("Inf"), device=attribute_scores.device)

        for seg_start_index in range(num_samples):
            for seg_duration in range(1, max_duration):
                seg_end_index = seg_start_index + seg_duration
                if seg_end_index > num_samples:
                    continue

                # FIXME: Line below only looks at first sample in batch. I think
                #   pytorch's default array broadcasting should make full-batch
                #   computations work just fine, but I haven't tried it.
                segment = attribute_scores[0:1, ..., seg_start_index:seg_end_index]
                scores = self.scoreSegment(segment)

                if not seg_start_index:
                    scores += self.start_weights
                    # Arrange scores as a column vector so tensor broadcasting
                    # replicates scores across columns---scores array needs to
                    # have shape matching (cur segment) x (prev segment)
                    scores = scores[:, :, None]
                else:
                    if seg_end_index == num_samples:
                        scores += self.end_weights
                        # Arrange scores as a column vector so tensor broadcasting
                        # replicates scores across columns---scores array needs to
                        # have shape matching (cur segment) x (prev segment)
                        scores = scores[:, :, None]
                    # FIXME: Make sure transition dimensions are broadcast
                    #   correctly for batch sizes > 1
                    scores = scores + self.transition_weights

                log_potentials[0:1, seg_start_index, seg_duration, :, :] = scores

        return log_potentials

    def scoreSegment(self, segment):
        """ Score a segment.

        Parameters
        ----------
        segment : torch.Tensor of float, shape (batch_size, ..., segment_len)

        Returns
        -------
        scores : torch.Tensor of float, shape (batch_size, num_classes)
        """

        # This flattens the segment if it was a sliding window of features
        segment = segment.contiguous().view(segment.shape[0], segment.shape[1], -1)
        return super().forward(segment).sum(dim=-1)


def conv2dOutputShape(
        shape_in, out_channels, kernel_size,
        stride=None, padding=0, dilation=1):

    if stride is None:
        stride = kernel_size

    shape_in = np.array(shape_in[0:2])
    kernel_size = np.array(kernel_size)
    stride = np.array(stride)
    padding = np.array(padding)
    dilation = np.array(dilation)

    shape_out = (shape_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    shape_out = np.floor(shape_out).astype(int)

    return shape_out.tolist() + [out_channels]
