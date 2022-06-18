//
// Created by spaceman on 2022. 02. 16..
//

#ifndef PARALLELLBFGS_CONSTANTS_CUH
#define PARALLELLBFGS_CONSTANTS_CUH

#ifdef PROBLEM_PLANEFITTING
#define  X_DIM 3
#define  RESIDUAL_COUNT 1

#define  RESIDUAL_PARAMETERS_DIM_1 3
#define  RESIDUAL_CONSTANTS_DIM_1 3
//#define  RESIDUAL_CONSTANTS_COUNT_1 20000
#endif

#ifdef PROBLEM_ROSENBROCK2D
#define  X_DIM 2
#define  RESIDUAL_COUNT 1

#define  RESIDUAL_PARAMETERS_DIM_1 2
#define  RESIDUAL_CONSTANTS_DIM_1 2
#define  RESIDUAL_CONSTANTS_COUNT_1 1

#endif


#ifdef PROBLEM_SNLP
//#define  X_DIM 200
#define  RESIDUAL_COUNT 2

#define  RESIDUAL_PARAMETERS_DIM_1 4
#define  RESIDUAL_CONSTANTS_DIM_1 3
//#define  RESIDUAL_CONSTANTS_COUNT_1 100

#define  RESIDUAL_PARAMETERS_DIM_2 2
#define  RESIDUAL_CONSTANTS_DIM_2 4
//#define  RESIDUAL_CONSTANTS_COUNT_2 10
#endif

#ifdef PROBLEM_SNLP3D
//#define  X_DIM 300
#define  RESIDUAL_COUNT 2

#define  RESIDUAL_PARAMETERS_DIM_1 6
#define  RESIDUAL_CONSTANTS_DIM_1 3
//#define  RESIDUAL_CONSTANTS_COUNT_1 294

#define  RESIDUAL_PARAMETERS_DIM_2 3
#define  RESIDUAL_CONSTANTS_DIM_2 5
//#define  RESIDUAL_CONSTANTS_COUNT_2 1
#endif

//#define  POPULATION_SIZE 5

//#define  DE_ITERATION_COUNT 2
//#define  ITERATION_COUNT 500
#define  THREADS_PER_BLOCK 128
#define  THREADS_PER_GRID (THREADS_PER_BLOCK*POPULATION_SIZE)

// Differential Evolution Control parameters
#define CR 0.99
#define F 0.6

//#define CR 0.9
//#define F  0.9

// Stop condition parameters
#define FEPSILON 1
#define  ALPHA 1

//#define FRAMESIZE 5
#endif //PARALLELLBFGS_CONSTANTS_CUH
