//
// Created by spaceman on 2022. 02. 16..
//

#ifndef PARALLELLBFGS_CONSTANTS_CUH
#define PARALLELLBFGS_CONSTANTS_CUH

#ifdef PROBLEM_PLANEFITTING
#define  X_DIM 3
#define  PARAMETER_DIM 3
#define  CONSTANT_DIM 3
#define  CONSTANT_COUNT 20000
#endif

#ifdef PROBLEM_ROSENBROCK2D
#define  X_DIM 2
#define  PARAMETER_DIM 2
#define  CONSTANT_DIM 2
#define  CONSTANT_COUNT 1
#endif

#ifdef PROBLEM_SNLP
#define  X_DIM 6
#define  PARAMETER_DIM 4
#define  CONSTANT_DIM 3
#define  CONSTANT_COUNT 3
#endif

#define  POPULATION_SIZE 1

#define DE_ITERATION_COUNT 0
#define  ITERATION_COUNT 10000
#define  ALPHA 200
#define  THREADS_PER_BLOCK 128
#define  THREADS_PER_GRID (THREADS_PER_BLOCK*POPULATION_SIZE)

// Differential Evolution Control parameters
#define CR 0.99
#define F 0.2

#endif //PARALLELLBFGS_CONSTANTS_CUH
