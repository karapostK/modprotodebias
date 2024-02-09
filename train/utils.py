import abc
import inspect
import os
from pathlib import Path
from typing import Tuple, Dict, List

import torch
from torch import nn


def general_weight_init(m):
    if type(m) == nn.Linear:
        if m.weight.requires_grad:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                torch.nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Embedding:
        if m.weight.requires_grad:
            torch.nn.init.normal_(m.weight, std=.1 / m.weight.shape[
                -1])  # std suggested by https://dl.acm.org/doi/10.1145/3523227.3548486 (see Appendix)


class SaveCondition(abc.ABC):
    """
    Base class that defines the conditions on which epoch the model needs to be saved.
    It holds an internal state that is updated when the "check" function is called.
    The class manages the model_save_path which is returned when the check function is called. This allows to save the
    model multiple times during training also based on different conditions.
    """

    def __init__(self, model_base_path: str):
        """
        @param model_base_path: Where to save the model. It is used to create subfolder to save the model for different conditions.
        """
        super().__init__()
        self.model_base_path = model_base_path

        self.model_save_path = None  # Used to memorize the path where the model has to be saved. It is set by the check function
        self.metrics_to_log = {}  # Metric that will be logged every epoch.

    @abc.abstractmethod
    def check(self, metric_dict: dict, epoch: int) -> bool:
        """
        Check if the condition is satisfied.
        NB. The state of the condition is not updated. Call retrieve_state after this function.
        :param metric_dict: dictionary with the metrics
        :param epoch: current epoch
        :return: a boolean indicating if the condition is satisfied

        """

    @abc.abstractmethod
    def update_and_retrieve_state(self, metric_dict: dict, epoch: int) -> Tuple[Dict, List]:
        """
        Update and Retrieve the internal state of the condition. This is useful when the condition needs to keep track of the previous
        values of the metrics.
        :param metric_dict: dictionary with the metrics
        :param epoch: current epoch
        return
            - a dictionary with the metrics to log (can be previous values or new values)
            - a list with the paths where the model has to be saved (can be empty if the condition is not satisfied)
        """


class MaxMinMetricSaveCondition(SaveCondition):
    """
    Save the model when the metric is higher/lower than the previous value + threshold
    """

    def __init__(self, model_base_path: str, metric_name: str, max_or_min: str = 'max', threshold: float = 0):
        """
        @param metric_name: Name of the metric to keep track
        @param max_or_min: 'max' or 'min'. Default is 'max'.
        @param threshold: Model is saved only when the metric is higher/lower than the previous value + threshold
        """
        assert max_or_min in ['max', 'min'], f"max_or_min must be 'max' or 'min'. Found {max_or_min}"

        super().__init__(model_base_path)

        self.metric_name = metric_name
        self.threshold = threshold
        self.max_or_min = max_or_min

        if self.max_or_min == 'max':
            self.metrics_to_log[f"max_{self.metric_name}"] = -torch.inf
        else:
            self.metrics_to_log[f"min_{self.metric_name}"] = torch.inf
        self.metrics_to_log[f"{self.max_or_min}_{self.metric_name}_epoch"] = -1

        self.model_save_path = os.path.join(self.model_base_path,
                                            f"{self.max_or_min}_{self.metric_name}")

    def check(self, metric_dict: dict, epoch: int) -> bool:
        curr_val = metric_dict[self.metric_name]
        prev_val = self.metrics_to_log[f"{self.max_or_min}_{self.metric_name}"]

        if (
                (self.max_or_min == 'max' and curr_val > prev_val + self.threshold) or
                (self.max_or_min == 'min' and curr_val < prev_val - self.threshold)
        ):
            return True
        else:
            return False

    def update_and_retrieve_state(self, metric_dict: dict, epoch: int) -> Tuple[Dict, List]:
        if self.check(metric_dict, epoch):
            self.metrics_to_log[f"{self.max_or_min}_{self.metric_name}"] = metric_dict[self.metric_name]
            self.metrics_to_log[f"{self.max_or_min}_{self.metric_name}_epoch"] = epoch
            return self.metrics_to_log, [self.model_save_path]
        else:
            return self.metrics_to_log, []

    @staticmethod
    def build_from_conf(conf: dict, dataset):

        init_signature = inspect.signature(MaxMinMetricSaveCondition.__init__)
        def_parameters = {k: v.default for k, v in init_signature.parameters.items() if
                          v.default is not inspect.Parameter.empty}
        parameters = {**def_parameters, **conf}

        return MaxMinMetricSaveCondition(parameters['model_path'], parameters['optimizing_metric'],
                                         parameters['max_or_min'], parameters['threshold'])


class PreserveMetricCondition(SaveCondition):
    """
    Save the model when the specified metric is within the acceptable range of the initial value.
    The range is defined in % of the initial value and does not need to be symmetric. Furthermore, it is possible to
    specify only a single direction (e.g. don't want to save a model where accuracy decreases too much, but it's
    acceptable to save the model when the accuracy increases.). For example, if the metric is accuracy of 0.5 and
    decr_perc is 0.1, then the model is NOT saved when the accuracy drops below 0.495 (0.5 - 0.1 * 0.5).
    NB. The initial value is determined THE FIRST TIME check is called!
    NB. This class only works with positive metrics!
    """

    def __init__(self, model_base_path: str, metric_name: str, incr_perc: float = None, decr_perc: float = None):
        """
        @param metric_name: Name of the metric to observe
        """
        assert incr_perc is not None or decr_perc is not None, "At least one between incr_perc and decr_perc must be not None"
        if incr_perc:
            assert 0 <= incr_perc <= 1, "incr_perc must be in [0,1]"
        if decr_perc:
            assert 0 <= decr_perc <= 1, "decr_perc must be in [0,1]"
        super().__init__(model_base_path)

        self.metric_name = metric_name
        self.incr_perc = incr_perc
        self.decr_perc = decr_perc

        self.first_time = True
        self.initial_val = None

        self.metrics_to_log[f'preserve_{self.metric_name}'] = None
        self.metrics_to_log[f'preserve_{self.metric_name}_epoch'] = -1

        self.model_save_path = os.path.join(self.model_base_path,
                                            f"preserve_{self.metric_name}")

    def check(self, metric_dict: dict, epoch: int) -> bool:
        curr_val = metric_dict[self.metric_name]

        assert curr_val >= 0, f"Metric {self.metric_name} is not positive ({curr_val})"

        if self.first_time:
            self.initial_val = self.metrics_to_log[f'preserve_{self.metric_name}'] = curr_val
            self.first_time = False
            return True

        decr_cond = (
                (self.decr_perc is None) or
                (self.decr_perc is not None and curr_val >= self.initial_val * (1 - self.decr_perc))
        )
        incr_cond = (
                (self.incr_perc is None) or
                (self.incr_perc is not None and curr_val <= self.initial_val * (1 + self.incr_perc))
        )

        if decr_cond and incr_cond:
            return True
        else:
            return False

    def update_and_retrieve_state(self, metric_dict: dict, epoch: int) -> Tuple[Dict, List]:
        if self.check(metric_dict, epoch):
            self.metrics_to_log[f'preserve_{self.metric_name}'] = metric_dict[self.metric_name]
            self.metrics_to_log[f'preserve_{self.metric_name}_epoch'] = epoch
            return self.metrics_to_log, [self.model_save_path]
        else:
            return self.metrics_to_log, []

    @staticmethod
    def build_from_conf(conf: dict, dataset):

        init_signature = inspect.signature(PreserveMetricCondition.__init__)
        def_parameters = {k: v.default for k, v in init_signature.parameters.items() if
                          v.default is not inspect.Parameter.empty}
        parameters = {**def_parameters, **conf}

        return PreserveMetricCondition(parameters['model_path'], parameters['optimizing_metric'],
                                       parameters['incr_perc'], parameters['decr_perc'])


class ORSaveCondition(SaveCondition):
    """
    Performs the OR of the saving conditions. Dictionaries are added together (overwriting the keys if present), while
    lists are concatenated.
    """

    def __init__(self, model_base_path: str, conditions: List[SaveCondition]):
        super().__init__(model_base_path)
        self.conditions = conditions
        self.model_save_path = None  # this is ignored

    def check(self, metric_dict: dict, epoch: int) -> bool:
        or_satisfied = False
        for condition in self.conditions:
            satisfied = condition.check(metric_dict, epoch)
            or_satisfied = or_satisfied or satisfied
        return or_satisfied

    def update_and_retrieve_state(self, metric_dict: dict, epoch: int) -> Tuple[Dict, List]:
        model_save_paths = []
        if self.check(metric_dict, epoch):
            for condition in self.conditions:
                metrics_to_log, paths = condition.update_and_retrieve_state(metric_dict, epoch)
                self.metrics_to_log = {**self.metrics_to_log, **metrics_to_log}
                model_save_paths += paths

        return self.metrics_to_log, model_save_paths


class ANDSaveCondition(SaveCondition):
    """
    Performs the AND of the saving conditions. Dictionaries are added together (overwriting the keys if present). A key
    that indicates the last epoch in which the conditions are satisfied is added. Path names are merged together.
    """

    def __init__(self, model_base_path: str, conditions: List[SaveCondition]):
        super().__init__(model_base_path)

        self.conditions = conditions

        self.model_save_path = None  # this is ignored

    def check(self, metric_dict: dict, epoch: int) -> bool:
        and_satisfied = True
        for condition in self.conditions:
            satisfied = condition.check(metric_dict, epoch)
            and_satisfied = and_satisfied and satisfied
        return and_satisfied

    def update_and_retrieve_state(self, metric_dict: dict, epoch: int) -> Tuple[Dict, List]:
        model_save_paths = []
        if self.check(metric_dict, epoch):
            or_save_path_names = []
            cond_log_dicts = []
            for condition in self.conditions:
                metrics_to_log, paths = condition.update_and_retrieve_state(metric_dict, epoch)
                cond_log_dicts.append(metrics_to_log)
                # self.metrics_to_log = {**self.metrics_to_log, **metrics_to_log}
                # Putting together or_paths
                or_save_path_name = '_or_'.join([Path(or_path).name for or_path in paths])
                or_save_path_names.append(or_save_path_name)

            # Path names are merged together
            and_save_path_name = '_and_'.join(or_save_path_names)
            model_save_paths += [os.path.join(self.model_base_path, and_save_path_name)]
            # Dictionaries are added together + pre_key
            for cond_log_dict in cond_log_dicts:
                cond_log_dict = {f"{and_save_path_name}({k})": v for k, v in cond_log_dict.items()}
                self.metrics_to_log = {**self.metrics_to_log, **cond_log_dict}
            self.metrics_to_log[f'{and_save_path_name}_epoch'] = epoch
        return self.metrics_to_log, model_save_paths
