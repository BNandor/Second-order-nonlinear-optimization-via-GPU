#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalAGTO(Optimizer):
    """
    The original version of: Artificial Gorilla Troops Optimization (AGTO)

    Links:
        1. https://doi.org/10.1002/int.22535
        2. https://www.mathworks.com/matlabcentral/fileexchange/95953-artificial-gorilla-troops-optimizer

    Notes (parameters):
        1. p1 (float): the probability of transition in exploration phase (p in the paper), default = 0.03
        2. p2 (float): the probability of transition in exploitation phase (w in the paper), default = 0.8
        3. beta (float): coefficient in updating equation, should be in [-5.0, 5.0], default = 3.0

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.AGTO import OriginalAGTO
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
    >>> model = OriginalAGTO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Abdollahzadeh, B., Soleimanian Gharehchopogh, F., & Mirjalili, S. (2021). Artificial gorilla troops optimizer: a new
    nature‐inspired metaheuristic algorithm for global optimization problems. International Journal of Intelligent Systems, 36(10), 5887-5958.
    """
    def __init__(self, epoch=10000, pop_size=100, p1=0.03, p2=0.8, beta=3.0, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.p1 = self.validator.check_float("p1", p1, (0, 1))      # p in the paper
        self.p2 = self.validator.check_float("p2", p2, (0, 1))      # w in the paper
        self.beta = self.validator.check_float("beta", beta, [-10.0, 10.0])
        self.set_parameters(["epoch", "pop_size", "p1", "p2", "beta"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        a = (np.cos(2*np.random.rand())+1) * (1 - (epoch+1)/self.epoch)
        c = a * (2 * np.random.rand() - 1)

        ## Exploration
        pop_new = []
        for idx in range(0, self.pop_size):
            if np.random.rand() < self.p1:
                pos_new = self.generate_position(self.problem.lb, self.problem.ub)
            else:
                if np.random.rand() >= 0.5:
                    z = np.random.uniform(-a, a, self.problem.n_dims)
                    rand_idx = np.random.randint(0, self.pop_size)
                    pos_new = (np.random.rand() - a) * self.pop[rand_idx][self.ID_POS] + c * z * self.pop[idx][self.ID_POS]
                else:
                    id1, id2 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                    pos_new = self.pop[idx][self.ID_POS] - c*(c*self.pop[idx][self.ID_POS] - self.pop[id1][self.ID_POS]) + \
                        np.random.rand() * (self.pop[idx][self.ID_POS] - self.pop[id2][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
        _, self.g_best = self.update_global_best_solution(self.pop, save=False)

        pos_list = np.array([agent[self.ID_POS] for agent in self.pop])
        ## Exploitation
        pop_new = []
        for idx in range(0, self.pop_size):
            if a >= self.p2:
                g = 2 ** c
                delta = (np.abs(np.mean(pos_list, axis=0)) ** g) ** (1.0 / g)
                pos_new = c*delta*(self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS]) + self.pop[idx][self.ID_POS]
            else:
                if np.random.rand() >= 0.5:
                    h = np.random.normal(0, 1, self.problem.n_dims)
                else:
                    h = np.random.normal(0, 1)
                r1 = np.random.rand()
                pos_new = self.g_best[self.ID_POS] - (2*r1-1)*(self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) * (self.beta * h)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)


class MGTO(Optimizer):
    """
    The original version of: Modified Gorilla Troops Optimization (mGTO)

    Notes (parameters):
        1. pp (float): the probability of transition in exploration phase (p in the paper), default = 0.03

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.AGTO import MGTO
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
    >>> model = OriginalAGTO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mostafa, R. R., Gaheen, M. A., Abd ElAziz, M., Al-Betar, M. A., & Ewees, A. A. (2023). An improved gorilla
    troops optimizer for global optimization problems and feature selection. Knowledge-Based Systems, 110462.
    """
    def __init__(self, epoch=10000, pop_size=100, pp=0.03,  **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pp (float): the probability of transition in exploration phase (p in the paper), default = 0.03
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.pp = self.validator.check_float("p1", pp, (0, 1))      # p in the paper
        self.set_parameters(["epoch", "pop_size", "pp"])
        self.sort_flag = False

    def bounded_position(self, position=None, lb=None, ub=None):
        condition = np.logical_and(lb <= position, position <= ub)
        random_pos = np.random.uniform(lb, ub)
        return np.where(condition, position, random_pos)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        F = 1 + np.cos(2 * np.random.rand())
        C = F * (1 - (epoch+1) / self.epoch)
        L = C * np.random.choice([-1, 1])

        ## Elite opposition-based learning
        pos_list = np.array([agent[self.ID_POS] for agent in self.pop])
        d_lb, d_ub = np.min(pos_list, axis=0), np.max(pos_list, axis=0)
        pos_list = d_lb + d_ub - pos_list
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.amend_position(pos_list[idx], self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = pop_new
        _, self.g_best = self.update_global_best_solution(self.pop, save=False)

        ## Exploration
        pop_new = []
        for idx in range(0, self.pop_size):
            if np.random.rand() < self.pp:
                pos_new = self.generate_position(self.problem.lb, self.problem.ub)
            else:
                if np.random.rand() >= 0.5:
                    rand_idx = np.random.randint(0, self.pop_size)
                    pos_new = (np.random.rand() - C) * self.pop[rand_idx][self.ID_POS] + L * np.random.uniform(-C, C) * self.pop[idx][self.ID_POS]
                else:
                    id1, id2 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                    pos_new = self.pop[idx][self.ID_POS] - L*(L*self.pop[idx][self.ID_POS] - self.pop[id1][self.ID_POS]) + \
                        np.random.rand() * (self.pop[idx][self.ID_POS] - self.pop[id2][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
        _, self.g_best = self.update_global_best_solution(self.pop, save=False)

        pos_list = np.array([agent[self.ID_POS] for agent in self.pop])
        ## Exploitation
        pop_new = []
        for idx in range(0, self.pop_size):
            if np.abs(C) >= 1:
                g = np.random.choice([-0.5, 2])
                M = (np.abs(np.mean(pos_list, axis=0)) ** g) ** (1.0 / g)
                p = np.random.uniform(0, 1, self.problem.n_dims)
                pos_new = L * M * (self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS]) * (0.01 * np.tan(np.pi*( p - 0.5)))
            else:
                Q = 2 * np.random.rand() - 1
                v = np.random.uniform(0, 1)
                pos_new = self.g_best[self.ID_POS] - Q * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) * np.tan(v * np.pi/2)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
