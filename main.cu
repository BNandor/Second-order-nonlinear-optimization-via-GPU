#include <iostream>
#include "core/optimizer/base/BaseLevel.cuh"
#include "core/optimizer/hyper/HyperLevel.cuh"
#include "core/optimizer/hyper/RandomHyperLevel.cuh"
#include "core/optimizer/hyper/SimulatedAnnealingHyperLevel.cuh"

int main(int argc, char** argv) {
    int totalFunctionEvaluations=DE_ITERATION_COUNT*ITERATION_COUNT;
//    HyperLevel* hyperLevel=new RandomHyperLevel();
    HyperLevel* hyperLevel=new SimulatedAnnealingHyperLevel();
    hyperLevel->hyperOptimize(totalFunctionEvaluations);
    delete hyperLevel;
    return 0;
}

// TODO add Simulated Annealing to hyper level

// Offline:
// Select best parameter combinations on a training set
// Validate the best parameter combinations on test set

// Hyper opt:

// a) random:
//      - population of random params -> select min

// b) simulated annealing
//      - 1 random parameter set and multiple perturbations of epsilon size

// c) DE,GA for hyperparameter selection
//      - population of random params, evolve parameters with population based operator

// Online:
// select best parameter combinations while solving the test set

// d) adaptiv approach:
// refine: 1%
// pertub: 100%
