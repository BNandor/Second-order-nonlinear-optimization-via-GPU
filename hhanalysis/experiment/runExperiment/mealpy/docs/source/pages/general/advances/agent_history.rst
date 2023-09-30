Agent's History (Trajectory)
============================

**WARNING: Trajectory will cause the memory issues:**

The history of the population is not saved by default, but you can enable this feature by setting the "save_population" keyword to True in the Problem
definition. Keep in mind that enabling this option may cause memory issues if your problem is too large, as it saves the history of the population in each
generation. However, if your problem is small enough, you can turn it on and visualize the trajectory chart of search agents.

.. code-block:: python

   problem_dict1 = {
      "fit_func": F5,
      "lb": [-3, -5, 1, -10, ],
      "ub": [5, 10, 100, 30, ],
      "minmax": "min",
      "log_to": "console",
      "save_population": True,              # Default = False
   }


You can access to the history of agent/population in model.history object with variables:
	+ list_global_best: List of global best SOLUTION found so far in all previous generations
	+ list_current_best: List of current best SOLUTION in each previous generations
	+ list_global_worst: List of global worst SOLUTION found so far in all previous generations
	+ list_current_worst: List of current worst SOLUTION in each previous generations
	+ list_epoch_time: List of runtime for each generation
	+ list_global_best_fit: List of global best FITNESS found so far in all previous generations
	+ list_current_best_fit: List of current best FITNESS in each previous generations
	+ list_diversity: List of DIVERSITY of swarm in all generations
	+ list_exploitation: List of EXPLOITATION percentages for all generations
	+ list_exploration: List of EXPLORATION percentages for all generations
	+ list_population: List of POPULATION in each generations

**Note**: The last variable, 'list_population', is the one that can cause the "memory" error described above.
It is recommended to set the 'save_population' parameter to False (which is also the default) in the input problem dictionary if you do not plan to use it.



.. code-block:: python

	import numpy as np
	from mealpy.swarm_based.PSO import OriginalPSO

	def fitness_function(solution):
	    return np.sum(solution**2)

	problem_dict = {
	    "fit_func": fitness_function,
	    "lb": [-10, -15, -4, -2, -8],
	    "ub": [10, 15, 12, 8, 20],
	    "minmax": "min",
	    "verbose": True,
	    "save_population": False        # Then you can't draw the trajectory chart
	}
	model = OriginalPSO(epoch=1000, pop_size=50)
	model.solve(problem=problem_dict)

	print(model.history.list_global_best)
	print(model.history.list_current_best)
	print(model.history.list_global_worst)
	print(model.history.list_current_worst)
	print(model.history.list_epoch_time)
	print(model.history.list_global_best_fit)
	print(model.history.list_current_best_fit)
	print(model.history.list_diversity)
	print(model.history.list_exploitation)
	print(model.history.list_exploration)
	print(model.history.list_population)

	## Remember if you set "save_population" to False, then there is no variable: list_population



.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

