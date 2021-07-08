"""寻找最优初始学习率：以指数/线性递增的方式从一个小的学习率开始迭代计算每个批次的loss,在一定迭代次数后，
   取loss下降最快(梯度值最小，注意，梯度值区分正负)的那个点对应的学习率为最优学习率"""

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from Train.lr_scheduler import ExponentialScheduler, LinearScheduler

from common_tools.state_cache import StateCacher
from common_tools.data_loader_iter import TrainDataLoaderIter, ValDataLoaderIter


try:
    from apex import amp

    IS_AMP_AVAILABLE = True
except ImportError:
    IS_AMP_AVAILABLE = False


class LRFinder:
    """Learning rate range test.
        The learning rate range test increases the learning rate in a pre-training run
        between two boundaries in a linear or exponential manner. It provides valuable
        information on how well the network can be trained over a range of learning rates
        and what is the optimal learning rate.
        Arguments:
            model (torch.nn.Module): wrapped model.
            optimizer (torch.optim.Optimizer): wrapped optimizer where the defined learning
                is assumed to be the lower boundary of the range test.
            criterion (torch.nn.Module): wrapped loss function.
            device (str or torch.device, optional): a string ("cpu" or "cuda") with an
                optional ordinal for the device type (e.g. "cuda:X", where is the ordinal).
                Alternatively, can be an object representing the device on which the
                computation will take place. Default: None, uses the same device as `model`.
            memory_cache (boolean, optional): if this flag is set to True, `state_dict` of
                model and optimizer will be cached in memory. Otherwise, they will be saved
                to files under the `cache_dir`.
            cache_dir (string, optional): path for storing temporary files. If no path is
                specified, system-wide temporary directory is used. Notice that this
                parameter will be ignored if `memory_cache` is True.
        Example:
            >>> lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
            >>> lr_finder.range_test(dataloader, end_lr=100, num_iter=100)
            >>> lr_finder.plot() # to inspect the loss-learning rate graph
            >>> lr_finder.reset() # to reset the model and optimizer to their initial state
        Reference:
        Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
        fastai/lr_find: https://github.com/fastai/fastai
    """

    def __init__(self, model: nn.Module, optimizer: Optimizer, criterion,
                 device=None, memory_cache=True, cache_dir=None):
        self.optimizer = optimizer
        # Check if the optimizer is already attached to a scheduler
        self._check_for_scheduler()

        self.model = model
        self.criterion = criterion
        self.memory_cache = memory_cache
        self.cache_dir = cache_dir

        self.best_loss = None
        self.history = {'lr': [], 'loss': []}

        self.model_device = model.device
        self.device = device if device else self.model_device

        # Save the original state of the model and optimizer so they can be restored if needed
        self.state_cacher = StateCacher(in_memory=memory_cache, cache_dir=cache_dir)
        self._state_cache(['model', 'optimizer'], [model.state_dict(), optimizer.state_dict()])

    def _check_for_scheduler(self):
        for param_group in self.optimizer.param_groups:
            if 'initial_lr' in param_group:
                raise RuntimeError("Optimizer already has a scheduler attached to it")

    def _state_cache(self, key_list, state_dict_list):
        assert len(key_list) == len(state_dict_list), f"the length of 'key_list' {len(key_list)} " \
                                                      f"& 'state_dict_list' {len(state_dict_list)} must be the same!"

        for key, state_dict in zip(key_list, state_dict_list):
            self.state_cacher.store(key, state_dict)

    def _clear_history(self):
        """clear historical test results"""

        for key in self.history:
            self.history[key].clear()

        self.best_loss = None

    def _set_learning_rate(self, new_lrs):
        if isinstance(new_lrs, (float, int)):
            new_lrs = [new_lrs] * len(self.optimizer.param_groups)
        assert len(new_lrs) == len(self.optimizer.param_groups), f"Length of `new_lrs`: {len(new_lrs)} is not " \
                                                                 f"equal to the number of parameter groups in " \
                                                                 f"the given optimizer"

        for lr, param_group in zip(new_lrs, self.optimizer.param_groups):
            param_group['lr'] = lr

    def _move_to_device(self, inputs, labels, non_blocking=True):
        def move(obj, device, non_blocking=True):
            if hasattr(obj, "to"):
                return obj.to(device, non_blocking=non_blocking)
            elif isinstance(obj, tuple):
                return tuple(move(o, device, non_blocking=non_blocking) for o in obj)
            elif isinstance(obj, list):
                return [move(o, device, non_blocking=non_blocking) for o in obj]
            elif isinstance(obj, dict):
                return {k: move(o, device, non_blocking=non_blocking) for k, o in obj.items()}
            else:
                return obj

        inputs = move(inputs, self.device, non_blocking=non_blocking)
        labels = move(labels, self.device, non_blocking=non_blocking)

        return inputs, labels

    def _train_batch(self, train_iter, accumulation_steps, non_blocking_transfer=True):
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = torch.tensor([0.], device=self.device)
        for i in range(accumulation_steps):
            inputs, labels = next(train_iter)
            inputs, labels = self._move_to_device(inputs, labels, non_blocking=non_blocking_transfer)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss /= accumulation_steps
            # Backward pass
            if IS_AMP_AVAILABLE and hasattr(self.optimizer, "_amp_stash"):
                # For minor performance optimization, see also:
                # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
                delay_unscale = ((i + 1) % accumulation_steps) != 0

                with amp.scale_loss(
                        loss, self.optimizer, delay_unscale=delay_unscale
                ) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            total_loss += loss

        self.optimizer.step()
        return total_loss.item()

    def _validate(self, val_iter, non_blocking_transfer=True):
        # Set model to evaluation mode and disable gradient computation
        self.model.eval()

        running_loss = 0.
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_iter, start=1):
                # Move data to the correct device
                inputs, labels = self._move_to_device(inputs, labels, non_blocking=non_blocking_transfer)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

        return running_loss / i

    def reset(self):
        """Restores the model and optimizer to their initial states."""

        model_state_dict = self.state_cacher.retrieve('model')
        optimizer_state_dict = self.state_cacher.retrieve('optimizer')

        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)

        self.model.to(self.model_device)

    def range_test(self, train_loader, val_loader=None, start_lr=None, end_lr=10, num_iter=100,
                   step_mode='exp', smooth_f=5e-2, diverge_threshold=5, accumulation_steps=1,
                   non_blocking_transfer=True):
        """Performs the learning rate range test.
                Arguments:
                    train_loader (`torch.utils.data.DataLoader`
                        or child of `TrainDataLoaderIter`, optional):
                        the training set data loader.
                        If your dataset (data loader) returns a tuple (inputs, labels,*) then
                        Pytorch data loader object can be provided. However, if a dataset
                        returns different outputs e.g. dicts, then you should inherit
                        from `TrainDataLoaderIter` class and redefine `inputs_labels_from_batch`
                        method so that it outputs (inputs, labels).
                    val_loader (`torch.utils.data.DataLoader`
                        or child of `ValDataLoaderIter`, optional): if `None` the range test
                        will only use the training loss. When given a data loader, the model is
                        evaluated after each iteration on that dataset and the evaluation loss
                        is used. Note that in this mode the test takes significantly longer but
                        generally produces more precise results.
                        Similarly to `train_loader`, if your dataset outputs are not standard
                        you should inherit from `ValDataLoaderIter` class and
                        redefine method `inputs_labels_from_batch` so that
                        it outputs (inputs, labels). Default: None.
                    start_lr (float, optional): the starting learning rate for the range test.
                        Default: None (uses the learning rate from the optimizer).
                    end_lr (float, optional): the maximum learning rate to test. Default: 10.
                    num_iter (int, optional): the number of iterations over which the test
                        occurs. Default: 100.
                    step_mode (str, optional): one of the available learning rate policies,
                        linear or exponential ("linear", "exp"). Default: "exp".
                    smooth_f (float, optional): the loss smoothing factor within the [0, 1[
                        interval. Disabled if set to 0, otherwise the loss is smoothed using
                        exponential smoothing. Default: 0.05.
                    diverge_threshold (int, optional): the test is stopped when the loss surpasses the
                        threshold:  diverge_th * best_loss. Default: 5.
                    accumulation_steps (int, optional): steps for gradient accumulation. If it
                        is 1, gradients are not accumulated. Default: 1.
                    non_blocking_transfer (bool, optional): when non_blocking_transfer is set,
                        tries to convert/move data to the device asynchronously if possible,
                        e.g., moving CPU Tensors with pinned memory to CUDA devices. Default: True.
                Example (fastai approach):
                    >>> lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
                    >>> lr_finder.range_test(dataloader, end_lr=100, num_iter=100)
                Example (Leslie Smith's approach):
                    >>> lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
                    >>> lr_finder.range_test(trainloader, val_loader=val_loader, end_lr=1,
                    >>>     num_iter=100, step_mode="linear")
                Gradient accumulation is supported; example:
                    >>> train_data = ...    # prepared dataset
                    >>> desired_bs, real_bs = 32, 4         # batch size
                    >>> accumulation_steps = desired_bs // real_bs     # required steps for accumulation
                    >>> dataloader = torch.utils.data.DataLoader(train_data, batch_size=real_bs, shuffle=True)
                    >>> acc_lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
                    >>> acc_lr_finder.range_test(dataloader, end_lr=10,
                    >>>     num_iter=100, accumulation_steps=accumulation_steps)
                If your DataLoader returns e.g. dict, or other non standard output, intehit from TrainDataLoaderIter,
                redefine method `inputs_labels_from_batch` so that it outputs (inputs, lables) data:
                    >>> import torch_lr_finder
                    >>> class TrainIter(torch_lr_finder.TrainDataLoaderIter):
                    >>>     def inputs_labels_from_batch(self, batch_data):
                    >>>         return (batch_data['user_features'], batch_data['user_history']), batch_data['y_labels']
                    >>> train_data_iter = TrainIter(train_dl)
                    >>> finder = torch_lr_finder.LRFinder(model, optimizer,
                    >>>     partial(model._train_loss, need_one_hot=False))
                    >>> finder.range_test(train_data_iter, end_lr=10, num_iter=300, diverge_th=10)
                Reference:
                [Training Neural Nets on Larger Batches: Practical Tips for 1-GPU, Multi-GPU & Distributed setups](
                https://medium.com/huggingface/ec88c3e51255)
                [thomwolf/gradient_accumulation](https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3)
        """

        assert 0 <= smooth_f < 1, f"smooth_f is outside the range [0, 1]"
        assert step_mode in ('exp', 'linear'), f"'step_mode' is expected one of (exp, linear), got {step_mode}"

        assert isinstance(train_loader, (DataLoader, TrainDataLoaderIter)), f"`train_loader` has unsupported type: " \
                                                                            f"{type(train_loader)}. Expected types" \
                                                                            f" are `torch.utils.data.DataLoader` " \
                                                                            f"or child of `TrainDataLoaderIter`."
        train_iter = train_loader if isinstance(train_loader, TrainDataLoaderIter) \
            else TrainDataLoaderIter(train_loader)

        if val_loader:
            assert isinstance(val_loader, (DataLoader, ValDataLoaderIter)), f"`val_loader` has unsupported type: " \
                                                                            f"{type(val_loader)}. Expected types" \
                                                                            f" are `torch.utils.data.DataLoader` " \
                                                                            f"or child of `ValDataLoaderIter`."
            val_iter = val_loader if isinstance(val_loader, ValDataLoaderIter) \
                else ValDataLoaderIter(val_loader)

        # reset the historical test results
        self._clear_history()
        # Check if the optimizer is already attached to a scheduler
        self._check_for_scheduler()

        lr_scheduler = ExponentialScheduler(self.optimizer, end_lr, num_iter) if step_mode == 'exp' \
            else LinearScheduler(self.optimizer, end_lr, num_iter)

        # Set the starting learning rate
        if start_lr:
            self._set_learning_rate(start_lr)

        # Move the model to the proper device
        self.model.to(self.model_device)

        for iteration in tqdm(range(num_iter)):
            # Train on batch and retrieve loss
            loss = self._train_batch(train_iter, accumulation_steps, non_blocking_transfer=non_blocking_transfer)
            if val_loader:
                loss = self._validate(val_iter, non_blocking_transfer=non_blocking_transfer)

            if iteration:
                self.best_loss = loss if loss < self.best_loss else self.best_loss

                if smooth_f:
                    loss = smooth_f * loss + (1 - smooth_f) * self.history['loss'][-1]
            else:
                self.best_loss = loss

            self.history['loss'].append(loss)
            self.history['lr'].append(lr_scheduler.get_lr()[0])

            lr_scheduler.step()

            if loss > diverge_threshold * self.best_loss:
                print("Stopping early, the loss has diverged")
                break

        print(f"Learning rate search finished. See the graph with {self.__class__}().plot()")

    def plot(self, skip_start=10, skip_end=5, log_lr=True, show_lr=None, ax=None, suggest_lr=True):
        """Plots the learning rate range test.
                Arguments:
                    skip_start (int, optional): number of batches to trim from the start.
                        Default: 10.
                    skip_end (int, optional): number of batches to trim from the start.
                        Default: 5.
                    log_lr (bool, optional): True to plot the learning rate in a logarithmic
                        scale; otherwise, plotted in a linear scale. Default: True.
                    show_lr (float, optional): if set, adds a vertical line to visualize the
                        specified learning rate. Default: None.
                    ax (matplotlib.axes.Axes, optional): the plot is created in the specified
                        matplotlib axes object and the figure is not be shown. If `None`, then
                        the figure and axes object are created in this method and the figure is
                        shown . Default: None.
                    suggest_lr (bool, optional): suggest a learning rate by
                        - 'steepest': the point with steepest gradient (minimal gradient)
                        you can use that point as a first guess for an LR. Default: True.
                Returns:
                    The matplotlib.axes.Axes object that contains the plot,
                    and the suggested learning rate (if set suggest_lr=True).
        """

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")
        if show_lr is not None and not isinstance(show_lr, float):
            raise ValueError("show_lr must be float")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history['lr'][skip_start:]
        losses = self.history['loss'][skip_start:]
        if skip_end:
            lrs = lrs[:-skip_end]
            losses = losses[:-skip_end]

        # Create the figure and axes object if axes was not already given
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        # Plot loss as a function of the learning rate
        ax.plot(lrs, losses)

        if log_lr:
            ax.set_xscale('log')

        ax.set_xlabel('Learninig rate')
        ax.set_ylabel('Loss')

        if show_lr is not None:
            ax.axvline(x=show_lr, color="red")

        # Plot the suggested LR
        if suggest_lr:
            # 'steepest': the point with steepest gradient (minimal gradient)
            print("LR suggestion: steepest gradient")
            min_grad_idx = None

            try:
                min_grad_idx = (np.gradient(np.asarray(losses))).argmin()
            except ValueError:
                print("Failed to compute the gradients, there might not be enough points.")

            if min_grad_idx is not None:
                print(f"Suggested LR: {lrs[min_grad_idx]:.2E}")
                ax.scatter(lrs[min_grad_idx], losses[min_grad_idx],
                           s=75, marker="o", color="red", zorder=3, label="steepest gradient")
                ax.lengend()

        # Show only if the figure was created internally
        if fig is not None:
            plt.show()

        return ax, lrs[min_grad_idx] if suggest_lr and min_grad_idx is not None else ax
