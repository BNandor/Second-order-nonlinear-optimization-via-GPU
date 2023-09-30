#!/usr/bin/env python
# Created by "Thieu" at 10:49, 11/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mealpy.optimizer import Optimizer
from mealpy.utils.problem import Problem
from mealpy.utils.validator import Validator
from collections import abc
from functools import partial, reduce
from itertools import product
import concurrent.futures as parallel
import operator
import os
import platform


class ParameterGrid:
    """
    Please check out this class from the scikit-learn library.

    It represents a grid of parameters with a discrete number of values for each parameter.
    This class is useful for iterating over parameter value combinations using the Python
    built-in function iter, and the generated parameter combinations' order is deterministic.

    Parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator parameters to sequences of allowed values.

        An empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is useful to avoid exploring
        parameter combinations that make no sense or have no effect. See the examples below.

    Examples
    --------
    >>> from mealpy.tuner import ParameterGrid
    >>> param_grid = {'a': [1, 2], 'b': [True, False]}
    >>> list(ParameterGrid(param_grid)) == ([{'a': 1, 'b': True}, {'a': 1, 'b': False}, {'a': 2, 'b': True}, {'a': 2, 'b': False}])
    True

    >>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
    >>> list(ParameterGrid(grid)) == [{'kernel': 'linear'}, {'kernel': 'rbf', 'gamma': 1}, {'kernel': 'rbf', 'gamma': 10}]
    True
    >>> ParameterGrid(grid)[1] == {'kernel': 'rbf', 'gamma': 1}
    True

    """

    def __init__(self, param_grid):
        if not isinstance(param_grid, (abc.Mapping, abc.Iterable)):
            raise TypeError(f"Parameter grid should be a dict or a list, got: {param_grid!r} of type {type(param_grid).__name__}")

        if isinstance(param_grid, abc.Mapping):
            # wrap dictionary in a singleton list to support either dict or list of dicts
            param_grid = [param_grid]

        # check if all entries are dictionaries of lists
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError(f"Parameter grid is not a dict ({grid!r})")
            for key, value in grid.items():
                if isinstance(value, np.ndarray) and value.ndim > 1:
                    raise ValueError(f"Parameter array for {key!r} should be one-dimensional, got: {value!r} with shape {value.shape}")
                if isinstance(value, str) or not isinstance(value, (np.ndarray, abc.Sequence)):
                    raise TypeError(
                        f"Parameter grid for parameter {key!r} needs to be a list or a"
                        f" numpy array, but got {value!r} (of type {type(value).__name__}) instead. Single values "
                        "need to be wrapped in a list with one element.")
                if len(value) == 0:
                    raise ValueError(f"Parameter grid for parameter {key!r} need to be a non-empty sequence, got: {value!r}")
        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid.

        Returns
        -------
        params : iterator over dict of str to any
            Yields dictionaries mapping each estimator parameter to one of its allowed values.
        """
        for p in self.param_grid:
            ## My version: Don't sort the key here. Keep it as it is
            if not p.items():
                yield {}
            else:
                keys, values = zip(*p.items())
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1 for p in self.param_grid)

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration

        Parameters
        ----------
        ind : int
            The iteration index

        Returns
        -------
        params : dict of str to any
            Equal to list(self)[ind]
        """
        # This is used to make discrete sampling without replacement memory efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            # keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            ## My version: Don't sort the values and don't reverse here. Keep it as it is
            keys, values_lists = zip(*sub_grid.items())
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError("ParameterGrid index out of range")


class Tuner:
    """Tuner utility class.

    This is a feature that enables the tuning of hyper-parameters for an algorithm.
    It also supports exporting results in various formats, such as Pandas DataFrame, JSON, and CSV.
    This feature provides a better option compared to using GridSearchCV or ParameterGrid from the scikit-learn library to tune hyper-parameters

    The important functions to note are 'execute()' and resolve()"

    Args:
        algorithm (Optimizer): the algorithm/optimizer to tune
        param_grid (dict, list): dict or list of dictionaries
        n_trials (int): number of repetitions
        mode (str): set the mode to run (sequential, thread, process), default="sequential"
        n_workers (int): effected only when mode is "thread" or "process".

    Examples
    --------
    >>> from opfunu.cec_based.cec2017 import F52017
    >>> from mealpy.evolutionary_based import GA
    >>> from mealpy.tuner import Tuner
    >>> f1 = F52017(30, f_bias=0)
    >>> p1 = {
    >>>     "lb": f1.lb,
    >>>     "ub": f1.ub,
    >>>     "minmax": "min",
    >>>     "fit_func": f1.evaluate,
    >>>     "name": "F5",
    >>>     "log_to": None,
    >>> }
    >>> term = {
    >>>     "max_epoch": 200,
    >>>     "max_time": 20,
    >>>     "max_fe": 10000
    >>> }
    >>> param_grid = {'epoch': [50, 100], 'pop_size': [10, 20], 'pc': [0.8, 0.85], 'pm': [0.01, 0.02]}
    >>> ga_tuner = Tuner(GA.BaseGA(), param_grid)
    >>> ga_tuner.execute(problem=p1, termination=term, n_trials=5, n_jobs=4, mode="single", n_workers=10, verbose=True)
    >>> ga_tuner.resolve(mode="thread", n_workers=10, termination=term)
    >>> ga_tuner.export_results(save_path="history/results", save_as="csv")
    """

    def __init__(self, algorithm=None, param_grid=None, **kwargs):
        self.__set_keyword_arguments(kwargs)
        self.validator = Validator(log_to="console", log_file=None)
        self.algorithm = self.validator.check_is_instance("algorithm", algorithm, Optimizer)
        self.param_grid = self.validator.check_is_instance("param_grid", param_grid, dict)
        self.results, self._best_row, self._best_params, self._best_score, self._best_algorithm = None, None, None, None, None

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def best_params(self):
        return self._best_params

    @best_params.setter
    def best_params(self, x):
        self._best_params = x

    @property
    def best_row(self):
        return self._best_row

    @property
    def best_score(self):
        return self._best_score

    @property
    def best_algorithm(self):
        self.algorithm.set_parameters(self._best_params)
        return self.algorithm

    def export_results(self, save_path=None, file_name="tuning_best_fit.csv"):
        """Export results to various file type

        Args:
            save_path (str): The path to the folder, default None
            file_name (str): The file name (with file type, e.g. dataframe, json, csv; default: "tuning_best_fit.csv") that hold results

        Raises:
            TypeError: Raises TypeError if export type is not supported

        """
        ## Check parent directories
        if save_path is None:
            save_path = f"history/{self.algorithm.get_name()}"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if type(file_name) is not str:
            raise ValueError("file_name should be a string and contains the extensions, e.g. dataframe, json, csv")
        ext = file_name.split(".")[-1]
        filename = "-".join(file_name.split(".")[:-1])
        if ext == "json":
            self.df_fit.to_json(f"{save_path}/{filename}.json")
        elif ext == "dataframe":
            self.df_fit.to_pickle(f"{save_path}/{filename}.pkl")
        else:
            self.df_fit.to_csv(f"{save_path}/{filename}.csv", header=True, index=False)

    def export_figures(self, save_path=None, file_name="tuning_epoch_fit.csv",
                       color=None, x_label=None, y_label=None, exts=(".png", ".pdf"), verbose=False):
        """Export results to various file type

        Args:
            save_path (str): The path to the folder, default None
            file_name (str): The file name (with file type, e.g. dataframe, json, csv; default: "tuning_epoch_fit.csv") that hold results

        Raises:
            TypeError: Raises TypeError if export type is not supported

        """
        ## Check parent directories
        if save_path is None:
            save_path = f"history/{self.algorithm.get_name()}"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if type(file_name) is not str:
            raise ValueError("file_name should be a string and contains the extensions, e.g. dataframe, json, csv")
        ext = file_name.split(".")[-1]
        filename = "-".join(file_name.split(".")[:-1])
        if ext == "json":
            self.df_loss.to_json(f"{save_path}/{filename}.json")
        elif ext == "dataframe":
            self.df_loss.to_pickle(f"{save_path}/{filename}.pkl")
        else:
            self.df_loss.to_csv(f"{save_path}/{filename}.csv", header=True, index=False)

        ## Draw and save convergence figures
        para_columns = list(self.param_grid.keys())
        group_trials = self.df_loss.groupby("trial")
        for trial, groups in group_trials:
            save_path_new = f"{save_path}/trial{trial}"
            Path(save_path_new).mkdir(parents=True, exist_ok=True)
            for idx_para, para in enumerate(para_columns):
                selected_paras = para_columns[:idx_para] + para_columns[idx_para + 1:]
                group_paras = groups.groupby(selected_paras)
                for idx_group, group_df in group_paras:
                    if len(group_df) <= 1:
                        continue
                    cols = list(group_df.columns.difference(['trial', ] + selected_paras, sort=False))
                    df_final = group_df[cols]
                    legends = df_final[para].values.tolist()
                    legends = [f"{para} = {item}" for item in legends]
                    # Remove the elites column if it exists
                    df_final = df_final.drop(para, axis=1)
                    # Plot a line chart for each elite parameter
                    title = f'Convergence chart for {para} parameter'
                    if x_label is None:
                        x_label = "Epoch"
                    if y_label is None:
                        y_label = "Global best fitness value"
                    if color is None:
                        df_final.T.plot(kind='line', title=title)
                    else:
                        if len(color) != df_final.values.shape[0]:
                            raise ValueError("color parameter should be a list with length equal to number of lines.")
                        else:
                            df_final.T.plot(kind='line', color=color, title=title)
                    plt.xlabel(x_label)
                    plt.ylabel(y_label)
                    plt.legend(legends)
                    fname = "-".join([f"{x}_{y}" for x, y in zip(selected_paras, idx_group)])
                    for idx, ext in enumerate(exts):
                        plt.savefig(f"{save_path_new}/{fname}{ext}", bbox_inches='tight')
                    if platform.system() != "Linux" and verbose:
                        plt.show()
                    plt.close()

    def __run__(self, id_trial, mode="single", n_workers=None, termination=None):
        _, best_fitness = self.algorithm.solve(self.problem, mode=mode, n_workers=n_workers, termination=termination)
        return id_trial, best_fitness, self.algorithm.history.list_global_best_fit

    def __generate_dict_from_list(self, my_list):
        keys = np.arange(1, len(my_list)+1)
        return dict(zip(keys, my_list))

    def __generate_dict_result(self, params, trial, loss_list):
        result_dict = dict(params)
        result_dict["trial"] = trial
        result_dict = {**result_dict, **self.__generate_dict_from_list(loss_list)}
        return result_dict

    def execute(self, problem=None, termination=None, n_trials=2, n_jobs=None, mode="single", n_workers=2, verbose=True):
        """Execute Tuner utility

        Args:
            problem (dict, Problem): An instance of Problem class or problem dictionary
            termination (None, dict, Termination): An instance of Termination class or termination dictionary
            n_trials (int): Number of trials on the Problem
            n_jobs (int, None): Speed up this task (run multiple trials at the same time) by using multiple processes. (<=1 or None: sequential, >=2: parallel)
            mode (str): Apply on current Problem ("single", "swarm", "thread", "process"), default="single".
            n_workers (int): Apply on current Problem, number of processes if mode is "thread" or "process'
            verbose (bool): Switch for verbose logging (default: False)

        Raises:
            TypeError: Raises TypeError if problem type is not dictionary or an instance Problem class

        """
        if not isinstance(problem, Problem):
            if type(problem) is dict:
                self.problem = Problem(**problem)
            else:
                raise TypeError(f"Problem is not an instance of Problem class or a Python dict.")
        self.n_trials = self.validator.check_int("n_trials", n_trials, [1, 100000])
        n_cpus = None
        if (n_jobs is not None) and (n_jobs >= 1):
            n_cpus = self.validator.check_int("n_jobs", n_jobs, [2, min(61, os.cpu_count() - 1)])

        if mode not in ("process", "thread", "single", "swarm"):
            mode = "single"

        list_params_grid = list(ParameterGrid(self.param_grid))
        trial_columns = [f"trial_{id_trial}" for id_trial in range(1, self.n_trials + 1)]
        ascending = True if self.problem.minmax == "min" else False

        best_fit_results = []
        loss_results = []
        for id_params, params in enumerate(list_params_grid):

            self.algorithm.set_parameters(params)
            best_fit_results.append({"params": params})

            trial_list = list(range(0, self.n_trials))
            if n_cpus is not None:
                with parallel.ProcessPoolExecutor(n_cpus) as executor:
                    list_results = executor.map(partial(self.__run__, n_workers=n_workers, mode=mode, termination=termination), trial_list)
                    for (idx, best_fitness, loss_epoch) in list_results:
                        best_fit_results[-1][trial_columns[idx]] = best_fitness
                        loss_results.append(self.__generate_dict_result(params, idx, loss_epoch))
                        if verbose:
                            print(f"Algorithm: {self.algorithm.get_name()}, with params: {params}, trial: {idx + 1}, best fitness: {best_fitness}")
            else:
                for idx in trial_list:
                    idx, best_fitness, loss_epoch = self.__run__(idx, mode=mode, n_workers=n_workers, termination=termination)
                    best_fit_results[-1][trial_columns[idx]] = best_fitness
                    loss_results.append(self.__generate_dict_result(params, idx, loss_epoch))
                    if verbose:
                        print(f"Algorithm: {self.algorithm.get_name()}, with params: {params}, trial: {idx+1}, best fitness: {best_fitness}")

        self.df_fit = pd.DataFrame(best_fit_results)
        self.df_fit["trial_mean"] = self.df_fit[trial_columns].mean(axis=1)
        self.df_fit["trial_std"] = self.df_fit[trial_columns].std(axis=1)
        self.df_fit["rank_mean"] = self.df_fit["trial_mean"].rank(ascending=ascending)
        self.df_fit["rank_std"] = self.df_fit["trial_std"].rank(ascending=ascending)
        self.df_fit["rank_mean_std"] = self.df_fit[["rank_mean", "rank_std"]].apply(tuple, axis=1).rank(method='dense', ascending=ascending)
        self._best_row = self.df_fit[self.df_fit["rank_mean_std"] == self.df_fit["rank_mean_std"].min()]
        self._best_params = self._best_row["params"].values[0]
        self._best_score = self._best_row["trial_mean"].values[0]
        self.df_loss = pd.DataFrame(loss_results)

    def resolve(self, mode='single', starting_positions=None, n_workers=None, termination=None):
        """
        Resolving the problem with the best parameters

        Args:
            mode (str): Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

                * 'process': The parallel mode with multiple cores run the tasks
                * 'thread': The parallel mode with multiple threads run the tasks
                * 'swarm': The sequential mode that no effect on updating phase of other agents
                * 'single': The sequential mode that effect on updating phase of other agents, default

            starting_positions(list, np.ndarray): List or 2D matrix (numpy array) of starting positions with length equal pop_size parameter
            n_workers (int): The number of workers (cores or threads) to do the tasks (effect only on parallel mode)
            termination (dict, None): The termination dictionary or an instance of Termination class

        Returns:
            list: [position, fitness value]
        """
        self.algorithm.set_parameters(self.best_params)
        return self.algorithm.solve(problem=self.problem, mode=mode, n_workers=n_workers,
                                    starting_positions=starting_positions, termination=termination)
