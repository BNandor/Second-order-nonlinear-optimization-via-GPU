#!/usr/bin/env python
# Created by "Thieu" at 19:27, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalEP(Optimizer):
    """
    The original version of: Evolutionary Programming (EP)

    Links:
        1. https://www.cleveralgorithms.com/nature-inspired/evolution/evolutionary_programming.html
        2. https://github.com/clever-algorithms/CleverAlgorithms

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + bout_size (float): [0.05, 0.2], percentage of child agents implement tournament selection

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.EP import OriginalEP
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> bout_size = 0.05
    >>> model = OriginalEP(epoch, pop_size, bout_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Yao, X., Liu, Y. and Lin, G., 1999. Evolutionary programming made faster.
    IEEE Transactions on Evolutionary computation, 3(2), pp.82-102.
    """

    ID_POS = 0
    ID_TAR = 1
    ID_STR = 2  # strategy
    ID_WIN = 3

    def __init__(self, epoch=10000, pop_size=100, bout_size=0.05, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            bout_size (float): percentage of child agents implement tournament selection
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.bout_size = self.validator.check_float("bout_size", bout_size, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "bout_size"])
        self.sort_flag = True

    def initialize_variables(self):
        self.n_bout_size = int(self.bout_size * self.pop_size)
        self.distance = 0.05 * (self.problem.ub - self.problem.lb)

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, target, strategy, times_win]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        strategy = np.random.uniform(0, self.distance, len(lb))
        times_win = 0
        return [position, target, strategy, times_win]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        child = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS] + self.pop[idx][self.ID_STR] * np.random.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            s_old = self.pop[idx][self.ID_STR] + np.random.normal(0, 1.0, self.problem.n_dims) * np.abs(self.pop[idx][self.ID_STR]) ** 0.5
            child.append([pos_new, None, s_old, 0])
            if self.mode not in self.AVAILABLE_MODES:
                child[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        child = self.update_target_wrapper_population(child)

        # Update the global best
        children, self.g_best = self.update_global_best_solution(child, save=False)
        pop = children + self.pop
        for i in range(0, len(pop)):
            ## Tournament winner (Tried with bout_size times)
            for idx in range(0, self.n_bout_size):
                rand_idx = np.random.randint(0, len(pop))
                if self.compare_agent(pop[i], pop[rand_idx]):
                    pop[i][self.ID_WIN] += 1
                else:
                    pop[rand_idx][self.ID_WIN] += 1
        pop = sorted(pop, key=lambda item: item[self.ID_WIN], reverse=True)
        self.pop = pop[:self.pop_size]


class LevyEP(OriginalEP):
    """
    The developed Levy-flight version: Evolutionary Programming (LevyEP)

    Notes
    ~~~~~
    Levy-flight is applied to EP, flow and some equations is changed.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + bout_size (float): [0.05, 0.2], percentage of child agents implement tournament selection

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.EP import LevyEP
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> bout_size = 0.05
    >>> model = LevyEP(epoch, pop_size, bout_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    ID_POS = 0
    ID_TAR = 1
    ID_STR = 2  # strategy
    ID_WIN = 3

    def __init__(self, epoch=10000, pop_size=100, bout_size=0.05, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            bout_size (float): percentage of child agents implement tournament selection
        """
        super().__init__(epoch, pop_size, bout_size, **kwargs)
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        child = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS] + self.pop[idx][self.ID_STR] * np.random.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            s_old = self.pop[idx][self.ID_STR] + np.random.normal(0, 1.0, self.problem.n_dims) * np.abs(self.pop[idx][self.ID_STR]) ** 0.5
            child.append([pos_new, None, s_old, 0])
            if self.mode not in self.AVAILABLE_MODES:
                child[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        child = self.update_target_wrapper_population(child)

        # Update the global best
        children, self.g_best = self.update_global_best_solution(child, save=False)
        pop = children + self.pop
        for i in range(0, len(pop)):
            ## Tournament winner (Tried with bout_size times)
            for idx in range(0, self.n_bout_size):
                rand_idx = np.random.randint(0, len(pop))
                if self.compare_agent(pop[i], pop[rand_idx]):
                    pop[i][self.ID_WIN] += 1
                else:
                    pop[rand_idx][self.ID_WIN] += 1

        ## Keep the top population, but 50% of left population will make a comeback an take the good position
        pop = sorted(pop, key=lambda agent: agent[self.ID_WIN], reverse=True)
        pop_new = pop[:self.pop_size]
        pop_left = pop[self.pop_size:]

        ## Choice random 50% of population left
        pop_comeback = []
        idx_list = np.random.choice(range(0, len(pop_left)), int(0.5 * len(pop_left)), replace=False)
        for idx in idx_list:
            pos_new = pop_left[idx][self.ID_POS] + self.get_levy_flight_step(multiplier=0.01, size=self.problem.n_dims, case=-1)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            strategy = self.distance = 0.05 * (self.problem.ub - self.problem.lb)
            pop_comeback.append([pos_new, None, strategy, 0])
            if self.mode not in self.AVAILABLE_MODES:
                pop_comeback[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_comeback = self.update_target_wrapper_population(pop_comeback)
        self.pop = self.get_sorted_strim_population(pop_new + pop_comeback, self.pop_size)
