import random
import json
import numpy as np
import copy
import math
from scipy.optimize import minimize
import numpy as np
import GPyOpt
from GPyOpt.methods import BayesianOptimization
from runExperiment.hyperParameterTuning.pyNMHH.operators.bayesGP import bayesGP
from runExperiment.hyperParameterTuning.pyNMHH.operators.bayesTPE import bayesTPE
import multiprocessing
from mystic.solvers import fmin_powell,fmin

class OptimizationHistory:
    def __init__(self):
        self.operatorStates={}
        self.best_offspring = []
        self.function_values = []
        self.population_history = []
        self.population_values_history = []
        self.json_data_history = []
        self.operator_sequence_history = []

    def add_operator_sequence(self, sequence):
        self.operator_sequence_history.append(copy.deepcopy(sequence))

    def addBaseConfigEntry(self,config):
        self.json_data_history.append(copy.deepcopy(config))

    def add_best_offspring(self, bestGenes, bestValue):
        self.best_offspring.append(copy.deepcopy(bestGenes))
        self.function_values.append(bestValue)

    def add_population(self, population, objectives):
        self.population_history.append(copy.deepcopy(population))
        self.population_values_history.append(objectives)
        

class FunctionWrapper:
    def __init__(self, func,evallimit,inf,lowerbounds=None,upperbounds=None,xtypes=None):
        self.func = func
        self.eval_count = 0
        self.evallimit=evallimit
        self.lowerbounds=lowerbounds
        self.upperbounds=upperbounds
        self.xtypes=xtypes
        self.inf=inf

    def __call__(self, x):
        if self.eval_count < self.evallimit:
            self.eval_count += 1
            if not self.withinBounds(x):
                return self.inf
            return self.func(x)
        else:
            return self.inf
        
    def trimToBounds(self,individual):
        if individual.fitness == self.inf:
            return individual
        trimmedGenes=individual.genes
        if self.lowerbounds is not None:
            trimmedGenes = np.array([max(a,b) for a,b in zip(trimmedGenes,self.lowerbounds)])
        if self.upperbounds is not None:
            trimmedGenes = np.array([min(a,b) for a,b in zip(trimmedGenes,self.upperbounds)])
        individual.genes=trimmedGenes
        return individual
    
    def withinBounds(self,x):
        if self.lowerbounds is not None and self.upperbounds is not None:
            return all(a<=b<=c for a,b,c in zip(self.lowerbounds,x,self.upperbounds))
        return True
    def inLimits(self):
        return self.eval_count<self.evallimit
    
class Individual:
    def __init__(self, genes, fitness=None):
        self.genes = genes
        self.fitness = fitness

    def evaluate(self, func):
        if self.fitness is None:
            self.fitness = func(self.genes)
        return self.fitness

# Define operator categories and operators

def initialize_markov_chain(baseLevelConfig):
    # Category-level Markov chain
    pi_C = baseLevelConfig["CategoryTransitionMatrix"]["InitialProbabilities"]
    P_C = baseLevelConfig["CategoryTransitionMatrix"]["TransitionMatrix"]

    # Operator-level Markov chains
    pi_c = {category: data["InitialProbabilities"] for category, data in baseLevelConfig["OperatorTransitionMatrices"].items()}
    P_c = {category: data["TransitionMatrix"] for category, data in baseLevelConfig["OperatorTransitionMatrices"].items()}

    return pi_C, P_C, pi_c, P_c

def next_state(current_state, transition_matrix):
    return random.choices(range(len(transition_matrix[current_state])), 
                          weights=transition_matrix[current_state])[0]

def apply_operator(operator, params, population, previous_population, func,hist):
    if not func.inLimits():
        return population
    if operator == 'DE':
        return differential_evolution(population, params, func)
    elif operator == 'GA':
        return genetic_algorithm(population, params, func)
    elif operator == 'GD':
        return gradient_descent(population, params, func)
    elif operator == 'LBFGS':
        return lbfgs(population, params, func)
    elif operator == 'best':
        return select_best(previous_population,population , func, len(population))
    elif operator == 'BayesGP':
        newX,newY=bayesGP(hist,func)
        # oldsInds=bayes_gp(hist, func)
        return [Individual(nextOffspring,fitness) for (nextOffspring,fitness) in zip(newX,newY)]
    elif operator == 'BayesTPE':
        newX,newY=bayesTPE(hist,func)
        return [Individual(nextOffspring,fitness) for (nextOffspring,fitness) in zip(newX,newY)]
    else:
        raise ValueError("Invalid operator")
        # return population

def differential_evolution(population, params, func):
    CR = params['DE_CR']['value']
    F = params['DE_FORCE']['value']
    new_population = []
    for i in range(len(population)):
        a, b, c = random.sample(range(len(population)), 3)
        trial=[]
        R=random.randint(0,len(population[i].genes)-1)
        for j in range(len(population[i].genes)):
            if random.random() < CR or j== R:
                trial.append( population[a].genes[j] + F * (population[b].genes[j] - population[c].genes[j]))
            else:
                trial.append(population[i].genes[j])
        new_population.append(Individual(np.array(trial)))
    return new_population

def genetic_algorithm(population, params, func):
    alpha = params['GA_ALPHA']['value']
    cr = params['GA_CR']['value']
    cr_point = params['GA_CR_POINT']['value']
    mutation_rate = params['GA_MUTATION_RATE']['value']
    mutation_size = params['GA_MUTATION_SIZE']['value']
    parent_pool_ratio = params['GA_PARENTPOOL_RATIO']['value']
    
    parent_pool_size = int(len(population) * parent_pool_ratio)
    
    
    new_population = []
    # print(f'GA: in {population}')
    while len(new_population) < len(population):
        parent_pool=random.sample(population, parent_pool_size)
        parent1 = sorted(parent_pool, key=lambda i:i.evaluate(func))[0]

        parent_pool=random.sample(population, parent_pool_size)
        sortedPool = sorted(parent_pool, key=lambda i:i.evaluate(func))
        parent2=sortedPool[0]
        if parent1 == parent2:
            if len(sortedPool)>1:
                parent2=sortedPool[1]
            else:
                parent2=random.sample(population, 1)[0]
        
        # parent1, parent2 = random.sample(parent_pool, 2)
        child = []
        crossover_point = int(len(parent1.genes) * cr_point)
        
        for i in range(len(parent1.genes)):
            if random.random() < cr:
                if i < crossover_point:
                    child.append(alpha * parent1.genes[i] + (1 - alpha) * parent2.genes[i])
                else:
                    child.append(alpha * parent2.genes[i] + (1 - alpha) * parent1.genes[i])
            else:
                child.append(parent1.genes[i])
        
        if random.random() < mutation_rate:
            mutation = [np.random.normal(0, (upper-lower)/mutation_size) for (lower,upper) in zip(func.lowerbounds,func.upperbounds)]
            child = [c + m for c, m in zip(child, mutation)]
        
        new_population.append(Individual(np.array(child)))
    # print(f'GA: out {new_population}')
    return new_population

def gradient_descent(population, params, func):
    
    # alpha = params['GD_ALPHA']['value']
    # fevals = int(params['GD_FEVALS']['value'])
    
    # new_population = []
    # for individual in population:
    #     result = minimize(func, individual.genes, method='CG', options={'maxiter': fevals})
    #     new_population.append(Individual(result.x, result.fun))
    optimizer = int(params['GD_OPTIMIZER']['value'])
    eps = int(params['GD_EPS']['value'])
    
    # new_population = copy.deepcopy(population)
    new_population = []
    # best=sorted(population, key=lambda ind: ind.evaluate(func))[0]
    optimizers=['Nelder-Mead','Powell','CG','BFGS','Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    sortedPop=sorted(population, key=lambda ind: ind.evaluate(func))
    for individual in sortedPop:
        # result = minimize(func, individual.genes,bounds=[(low,up) for (low,up) in zip(func.lowerbounds,func.upperbounds)], 
        #                   method=optimizers[optimizer], options={'maxfun':len(individual.genes),
        #                                         'maxiter': 1,
        #                                         'eps':eps,
        #                                         'ftol':0.000001})
        result = fmin_powell(func, individual.genes,bounds=[(low,up) for (low,up) in zip(func.lowerbounds,func.upperbounds)], 
                             ftol=0.001,
                             maxiter=1,
                             maxfun=len(individual.genes)*2,
                             full_output=True,
                             disp=True)
        # print(result)
        # new_population[population.index(best)]=Individual(result.x, result.fun)
        new_population.append(Individual(result[0], result[1]))
    return new_population

def lbfgs(population, params, func):
    eps = int(params['LBFGS_EPS']['value'])
    
    # new_population = copy.deepcopy(population)
    # best=sorted(population, key=lambda ind: ind.evaluate(func))[0]
    new_population = []
    sortedPop=sorted(population, key=lambda ind: ind.evaluate(func))
    for individual in sortedPop:
        # result = minimize(func,  individual.genes, bounds=[(low,up)  for (low,up) in zip(func.lowerbounds,func.upperbounds)], 
        #                   method='L-BFGS-B',
        #                   options={'maxfun':len(individual.genes),
        #                            'maxiter': 1,
        #                            'maxls':3,
        #                             'maxcor': 30,
        #                             'eps':eps,
        #                             'ftol':0.0001})
        result= fmin(func, individual.genes,bounds=[(low,up) for (low,up) in zip(func.lowerbounds,func.upperbounds)], 
                             ftol=0.001,
                             maxiter=1,
                             maxfun=len(individual.genes)*2,
                             full_output=True,
                             disp=True)
        
        # new_population[population.index(best)]=Individual(result.x, result.fun)
        new_population.append(Individual(result[0], result[1]))
    # new_population.append
    return new_population

def get_operator_params(baseLevelConfig,category, operator):
    key = f"{category.capitalize()}{operator}OperatorParams"
    return baseLevelConfig['OperatorParams'].get(key, {})

def select_best(current_population,offspring_population, func, num_select):
    # sorted_current = sorted(current_population, key=lambda ind: ind.evaluate(func))
    
    # # Determine the number of elites to keep
    # num_elites = max(3, int(0.1 * num_select))  # Keep at least 1, up to 10% as elites
    
    # # Select the elites from the current population
    # elites = sorted_current[:num_elites]
    
    # # Sort the offspring population
    # sorted_offspring = sorted(offspring_population, key=lambda ind: ind.evaluate(func))
    
    # # Select the best individuals from the offspring to fill the rest of the new population
    # selected_offspring = sorted_offspring[:num_select - num_elites]
    
    # # Combine elites and selected offspring
    # new_population = elites + selected_offspring
    
    # return new_population
    # return sorted(current_population+offspring_population, key=lambda ind: ind.evaluate(func))[:num_select]
    new_population=[]
    for i in range(len(current_population)):
        if current_population[i].evaluate(func) < offspring_population[i].evaluate(func):
            new_population.append(current_population[i])
        else:
            new_population.append(offspring_population[i])
    return new_population

# Not used 
def bayes_gp(hist,func):
    histsize=len(hist.population_history)
    X = []
    Y = []
    for pop in hist.population_history[-(min(histsize,30)):]:
        for ind in pop:
            X.append(ind)
    for popvalues in hist.population_values_history[-(min(histsize,30)):]:
        for value in popvalues:
            Y.append([value])

    # Define the bounds of your search space
    bounds = [{'name': str(i), 'type': type, 'domain': (lower, upper)} for (i,(lower,upper,type))in enumerate(zip(func.lowerbounds,func.upperbounds,func.xtypes))]

    # Create the Bayesian Optimization model
    bayes_opt = BayesianOptimization(
        f=lambda x: func(x[0]),            # No function needed because we're providing initial data
        domain=bounds,
        X=np.array(X),               # Previously evaluated points
        Y=np.array(Y),               # Corresponding objective values
        acquisition_type='MPI',
        num_cores=multiprocessing.cpu_count()       # Expected Improvement acquisition function
    )
    popsize=len(hist.population_history[0])
    newpointcount=popsize
    # Run the optimization to find the next point
    bayes_opt.run_optimization(max_iter=newpointcount)
    return [Individual(nextOffspring,fitness[0]) for (nextOffspring,fitness) in zip(bayes_opt.X[-newpointcount:],bayes_opt.Y[-newpointcount:])]

def NMHHBaseOpt(wrapped_objective, initialPopulation, max_evaluations, baseLevelConfig, history):
    pi_C, P_C, pi_c, P_c = initialize_markov_chain(baseLevelConfig)
    population=[ wrapped_objective.trimToBounds(offspring) for offspring in initialPopulation]
    population_size=len(population)
    previous_population = population.copy()
    operatorSequence=[]

    category_state = random.choices(range(len(baseLevelConfig['OperatorsAndCategories']['categories'])), weights=pi_C)[0]
    operator_states = {cat: random.choices(range(len(ops)), weights=pi_c[cat])[0] 
                       for cat, ops in baseLevelConfig['OperatorsAndCategories']['operators'].items()}

    best_individual = min(population, key=lambda ind: ind.evaluate(wrapped_objective))
    best_fitness = best_individual.fitness
    if not isinstance(best_fitness, float ):
                    best_fitness=best_fitness[0]
    history.add_population([offspring.genes for offspring in population],[offspring.evaluate(wrapped_objective) for offspring in population])

    while wrapped_objective.eval_count < max_evaluations:
        current_category = baseLevelConfig['OperatorsAndCategories']['categories'][category_state]
        current_operator = baseLevelConfig['OperatorsAndCategories']['operators'][current_category][operator_states[current_category]]
        operator_params = get_operator_params(baseLevelConfig,current_category, current_operator)
        
        new_population =[wrapped_objective.trimToBounds(offspring) for offspring in apply_operator(current_operator, operator_params, population, previous_population, wrapped_objective,history)]

        previous_population = population
        population = new_population
        [offspring.evaluate(wrapped_objective) for offspring in population]
        
        if len(population) > population_size:
            population = select_best(population, population_size)

        if wrapped_objective.eval_count <= max_evaluations:
            operatorSequence.append(current_operator)
            current_best = min(population, key=lambda ind: ind.evaluate(wrapped_objective))
            if current_best.fitness < best_fitness:
                best_individual = current_best
                best_fitness = best_individual.fitness
                if not isinstance(best_fitness, float ):
                    best_fitness=best_fitness[0]
            category_state = next_state(category_state, P_C)
            operator_states[current_category] = next_state(operator_states[current_category], P_c[current_category])

            # Store population history    
            history.add_population([offspring.genes for offspring in population],[offspring.evaluate(wrapped_objective) for offspring in population])
            if  (wrapped_objective.eval_count % (max_evaluations // 10 if max_evaluations>10  else max_evaluations) == 0):         
                if not isinstance(best_fitness, float ):
                    best_fitness=best_fitness[0]
                print(f"                >>>>Evaluation count: {wrapped_objective.eval_count}, Best fitness = {best_fitness}")

    # Store best offspring history
    history.add_best_offspring(best_individual.genes, best_fitness)
    history.addBaseConfigEntry(baseLevelConfig)
    history.add_operator_sequence(operatorSequence)
    return best_individual.genes, best_fitness, wrapped_objective.eval_count

def perturb_baseLevelConfig(baseLevelConfig, temperature):
    def normalize_transition_matrix(matrix):
        """Normalize probabilities in transition matrices."""
        if isinstance(matrix, list):
            if isinstance(matrix[0], list):
                return [normalize_transition_matrix(row) for row in matrix]
            else:
                total = sum(matrix)
                return [v / total for v in matrix] if total > 0 else matrix
        return matrix
    def rebalance_transition_row(row, restricted_indices):
        # Convert row to list if it's not already
        row = list(row)
        reachableStateCount=len(row)-len(restricted_indices)
        # Calculate the sum of probabilities for non-restricted states
        compensating_prob = sum(p/reachableStateCount for i, p in enumerate(row) if i in restricted_indices)    
        # Create the new row
        new_row = []
        for i, p in enumerate(row):
            if i in restricted_indices:
                new_row.append(0)
            else:
                new_row.append(p + compensating_prob)
        
        return new_row
    def perturb_value(value, lower, upper, temp):
        max_change = (upper - lower) * temp
        return max(lower, min(upper, value + random.uniform(-max_change, max_change)))
    
    """Perturb the baseLevelConfig parameters based on the current temperature."""
    newBaseLevelConfig = copy.deepcopy(baseLevelConfig)
    for category, matrix in newBaseLevelConfig["CategoryTransitionMatrix"].items():
        if isinstance(matrix, list):
            for i in range(len(matrix)):
                if isinstance(matrix[i], list):
                    matrix[i] = [perturb_value(v, 0.01, 1, temperature) for v in matrix[i]]
                else:
                    matrix[i] = perturb_value(matrix[i], 0.01, 1, temperature)
    
    for category, data in newBaseLevelConfig["OperatorTransitionMatrices"].items():
        for key, matrix in data.items():
            if isinstance(matrix, list):
                for i in range(len(matrix)):
                    if isinstance(matrix[i], list):
                        matrix[i] = [perturb_value(v, 0.01, 1, temperature) for v in matrix[i]]
                    else:
                        matrix[i] = perturb_value(matrix[i], 0.01, 1, temperature)
    
    # Perturb other parameters in baseLevelConfig
    for key, value in newBaseLevelConfig['OperatorParams'].items():
        if isinstance(value, dict) and "lowerBound" in value and "upperBound" in value:
            newBaseLevelConfig[key]["value"] = perturb_value(value["value"], value["lowerBound"], value["upperBound"], temperature)

    newBaseLevelConfig["CategoryTransitionMatrix"]["InitialProbabilities"] = normalize_transition_matrix(newBaseLevelConfig["CategoryTransitionMatrix"]["InitialProbabilities"])
    newBaseLevelConfig["CategoryTransitionMatrix"]["TransitionMatrix"] = normalize_transition_matrix(newBaseLevelConfig["CategoryTransitionMatrix"]["TransitionMatrix"]) 
    newBaseLevelConfig["CategoryTransitionMatrix"]["TransitionMatrix"] = [rebalance_transition_row(row,restrictions) for row,restrictions in zip(newBaseLevelConfig["CategoryTransitionMatrix"]["TransitionMatrix"],newBaseLevelConfig["Restrictions"]["CategoryTransitionRestrictions"])]
    
    for category in newBaseLevelConfig["OperatorTransitionMatrices"]:
            newBaseLevelConfig["OperatorTransitionMatrices"][category]["InitialProbabilities"] = normalize_transition_matrix(newBaseLevelConfig["OperatorTransitionMatrices"][category]["InitialProbabilities"])
            newBaseLevelConfig["OperatorTransitionMatrices"][category]["TransitionMatrix"] = normalize_transition_matrix(newBaseLevelConfig["OperatorTransitionMatrices"][category]["TransitionMatrix"])
    return newBaseLevelConfig

def NMHHHyperOpt(initialBaseLevelConfig, baseLevel, iterations=1000, initialTemp=1.0, cooling_rate=0.995):
    currentBaseLevelConfig = copy.deepcopy(initialBaseLevelConfig)
    history = OptimizationHistory()
    solution=baseLevel(currentBaseLevelConfig, history)
    currentEnergy=solution['fitness'] 
    currentSolution=solution['solution']
    bestBaseLevelConfig = currentBaseLevelConfig
    bestEnergy = currentEnergy
    bestSolution=currentSolution
    bestSequence=history.operator_sequence_history[0]
    temperature = initialTemp

    for i in range(iterations-1):
        perturbedBaseLevelConfig = perturb_baseLevelConfig(currentBaseLevelConfig, temperature)
        solution=baseLevel(perturbedBaseLevelConfig, history)
        perturbedConfigEnergy= solution['fitness']
        currentSolution=solution['solution']
        
        if perturbedConfigEnergy < currentEnergy or random.random() < math.exp((currentEnergy - perturbedConfigEnergy) / temperature):
            currentBaseLevelConfig = perturbedBaseLevelConfig
            currentEnergy = perturbedConfigEnergy
            
            if currentEnergy < bestEnergy:
                bestBaseLevelConfig = currentBaseLevelConfig
                bestEnergy = currentEnergy
                bestSolution=currentSolution
                bestSequence=history.operator_sequence_history[-1]
                print(f"                >>>>Iteration {i}: New best energy = {bestEnergy}")
        
        temperature *= cooling_rate
    
    return bestBaseLevelConfig, bestEnergy, history, bestSequence,bestSolution