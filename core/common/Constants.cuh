//
// Created by spaceman on 2022. 02. 16..
//

#ifndef PARALLELLBFGS_CONSTANTS_CUH
#define PARALLELLBFGS_CONSTANTS_CUH
#ifdef PROBLEM_PLANEFITTING
#define  X_DIM 3
#define  OBSERVATION_DIM 3
#define  OBSERVATION_COUNT 20000
#endif

#ifdef PROBLEM_ROSENBROCK2D
#define  X_DIM 2
#define  OBSERVATION_DIM 2
#define  OBSERVATION_COUNT 1
#endif

#define  POPULATION_SIZE 40

#define  ITERATION_COUNT 100000
#define  ALPHA 200
#define  THREADS_PER_BLOCK 128
#define  THREADS_PER_GRID (THREADS_PER_BLOCK*POPULATION_SIZE)

// Differential Evolution Control parameters
#define CR 0.99
#define F 0.2

#endif //PARALLELLBFGS_CONSTANTS_CUH
