#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalGWO(Optimizer):
    """
    The original version of: Grey Wolf Optimizer (GWO)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2013.12.007
        2. https://www.mathworks.com/matlabcentral/fileexchange/44974-grey-wolf-optimizer-gwo?s_tid=FX_rc3_behav

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.GWO import OriginalGWO
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
    >>> model = OriginalGWO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., Mirjalili, S.M. and Lewis, A., 2014. Grey wolf optimizer. Advances in engineering software, 69, pp.46-61.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2 - 2 * epoch / (self.epoch - 1)
        _, list_best, _ = self.get_special_solutions(self.pop, best=3)

        pop_new = []
        for idx in range(0, self.pop_size):
            A1 = a * (2 * np.random.rand(self.problem.n_dims) - 1)
            A2 = a * (2 * np.random.rand(self.problem.n_dims) - 1)
            A3 = a * (2 * np.random.rand(self.problem.n_dims) - 1)
            C1 = 2 * np.random.rand(self.problem.n_dims)
            C2 = 2 * np.random.rand(self.problem.n_dims)
            C3 = 2 * np.random.rand(self.problem.n_dims)
            X1 = list_best[0][self.ID_POS] - A1 * np.abs(C1 * list_best[0][self.ID_POS] - self.pop[idx][self.ID_POS])
            X2 = list_best[1][self.ID_POS] - A2 * np.abs(C2 * list_best[1][self.ID_POS] - self.pop[idx][self.ID_POS])
            X3 = list_best[2][self.ID_POS] - A3 * np.abs(C3 * list_best[2][self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)


class RW_GWO(Optimizer):
    """
    The original version of: Random Walk Grey Wolf Optimizer (RW-GWO)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.GWO import RW_GWO
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
    >>> model = RW_GWO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Gupta, S. and Deep, K., 2019. A novel random walk grey wolf optimizer. Swarm and evolutionary computation, 44, pp.101-112.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0, Eq. 5
        b = 2 - 2 * epoch / (self.epoch - 1)
        # linearly decreased from 2 to 0
        a = 2 - 2 * epoch / (self.epoch - 1)
        _, leaders, _ = self.get_special_solutions(self.pop, best=3)

        ## Random walk here
        leaders_new = []
        for i in range(0, len(leaders)):
            pos_new = leaders[i][self.ID_POS] + a * np.random.standard_cauchy(self.problem.n_dims)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            leaders_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                leaders[i] = self.get_better_solution([pos_new, target], leaders[i])
        if self.mode in self.AVAILABLE_MODES:
            leaders_new = self.update_target_wrapper_population(leaders_new)
            leaders = self.greedy_selection_population(leaders, leaders_new)

        ## Update other wolfs
        pop_new = []
        for idx in range(0, self.pop_size):
            # Eq. 3 and 4
            miu1 = a * (2 * np.random.rand(self.problem.n_dims) - 1)
            miu2 = a * (2 * np.random.rand(self.problem.n_dims) - 1)
            miu3 = a * (2 * np.random.rand(self.problem.n_dims) - 1)
            c1 = 2 * np.random.rand(self.problem.n_dims)
            c2 = 2 * np.random.rand(self.problem.n_dims)
            c3 = 2 * np.random.rand(self.problem.n_dims)
            X1 = leaders[0][self.ID_POS] - miu1 * np.abs(c1 * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            X2 = leaders[1][self.ID_POS] - miu2 * np.abs(c2 * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            X3 = leaders[2][self.ID_POS] - miu3 * np.abs(c3 * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])

            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
        self.pop = self.get_sorted_strim_population(self.pop + leaders, self.pop_size)


class GWO_WOA(OriginalGWO):
    """
    The original version of: Hybrid Grey Wolf - Whale Optimization Algorithm (GWO_WOA)

    Links:
        1. https://sci-hub.se/https://doi.org/10.1177/10775463211003402

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.GWO import GWO_WOA
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
    >>> model = GWO_WOA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Obadina, O. O., Thaha, M. A., Althoefer, K., & Shaheed, M. H. (2022). Dynamic characterization of a master–slave
    robotic manipulator using a hybrid grey wolf–whale optimization algorithm. Journal of Vibration and Control, 28(15-16), 1992-2003.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.b = 1.0
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2 - (epoch + 1) / self.epoch
        _, list_best, _ = self.get_special_solutions(self.pop, best=3)

        pop_new = []
        for idx in range(0, self.pop_size):
            A1 = a * (2 * np.random.rand(self.problem.n_dims) - 1)
            A2 = a * (2 * np.random.rand(self.problem.n_dims) - 1)
            A3 = a * (2 * np.random.rand(self.problem.n_dims) - 1)
            C1 = 2 * np.random.rand(self.problem.n_dims)
            C2 = 2 * np.random.rand(self.problem.n_dims)
            C3 = 2 * np.random.rand(self.problem.n_dims)
            if np.random.random() < 0.5:
                da = np.random.random() * np.abs(C1 * list_best[0][self.ID_POS] - self.pop[idx][self.ID_POS])
            else:
                P, L = np.random.random(), np.random.uniform(-1, 1)
                da = P * np.exp(self.b * L) * np.cos(2*np.pi*L) * np.abs(C1 * list_best[0][self.ID_POS] - self.pop[idx][self.ID_POS])
            X1 = list_best[0][self.ID_POS] - A1 * da
            X2 = list_best[1][self.ID_POS] - A2 * np.abs(C2 * list_best[1][self.ID_POS] - self.pop[idx][self.ID_POS])
            X3 = list_best[2][self.ID_POS] - A3 * np.abs(C3 * list_best[2][self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)


class IGWO(OriginalGWO):
    """
    The original version of: Improved Grey Wolf Optimization (IGWO)

    Notes:
        1. Link: https://doi.org/10.1007/s00366-017-0567-1
        2. Implemented by: Mohammadtaher Abbasi (https://github.com/mtabbasi)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + a_min (float): Lower bound of a, default = 0.02
        + a_max (float): Upper bound of a, default = 2.2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.GWO import IGWO
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
    >>> a_min = 0.02
    >>> a_max = 2.2
    >>> model = IGWO(epoch, pop_size, a_min, a_max)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Kaveh, A. & Zakian, P.. (2018). Improved GWO algorithm for optimal design of truss structures.
    Engineering with Computers. 34. 10.1007/s00366-017-0567-1.
    """

    def __init__(self, epoch=10000, pop_size=100, a_min=0.02, a_max=2.2, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            a_min (float): Lower bound of a, default = 0.02
            a_max (float): Upper bound of a, default = 2.2
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.a_min = self.validator.check_float("a_min", a_min, (0.0, 1.6))
        self.a_max = self.validator.check_float("a_max", a_max, [1., 4.])
        self.set_parameters(["epoch", "pop_size", "a_min", "a_max"])
        self.growth_alpha = 2
        self.growth_delta = 3

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm.

        Args:
            epoch (int): The current iteration
        """
        _, list_best, _ = self.get_special_solutions(self.pop, best=3)

        pop_new = []
        for idx in range(0, self.pop_size):
            # IGWO functions
            a_alpha = self.a_max * np.exp(
                (epoch / self.epoch) ** self.growth_alpha
                * np.log(self.a_min / self.a_max)
            )
            a_delta = self.a_max * np.exp(
                (epoch / self.epoch) ** self.growth_delta
                * np.log(self.a_min / self.a_max)
            )
            a_beta = (a_alpha + a_delta) * 0.5
            A1 = a_alpha * (2 * np.random.rand(self.problem.n_dims) - 1)
            A2 = a_beta * (2 * np.random.rand(self.problem.n_dims) - 1)
            A3 = a_delta * (2 * np.random.rand(self.problem.n_dims) - 1)
            C1 = 2 * np.random.rand(self.problem.n_dims)
            C2 = 2 * np.random.rand(self.problem.n_dims)
            C3 = 2 * np.random.rand(self.problem.n_dims)
            X1 = list_best[0][self.ID_POS] - A1 * np.abs(
                C1 * list_best[0][self.ID_POS] - self.pop[idx][self.ID_POS]
            )
            X2 = list_best[1][self.ID_POS] - A2 * np.abs(
                C2 * list_best[1][self.ID_POS] - self.pop[idx][self.ID_POS]
            )
            X3 = list_best[2][self.ID_POS] - A3 * np.abs(
                C3 * list_best[2][self.ID_POS] - self.pop[idx][self.ID_POS]
            )
            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
