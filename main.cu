#include <iostream>
#include "core/optimizer/base/BaseLevel.cuh"
#include "core/optimizer/hyper/HyperLevel.cuh"
#include "core/optimizer/hyper/RandomHyperLevel.cuh"
#include "core/optimizer/hyper/SimulatedAnnealingHyperLevel.cuh"
#include "core/optimizer/hyper/SimpleLocalSearchHyperLevel.cuh"
#include "core/optimizer/hyper/SimplePerturbHyperLevel.cuh"
#include "core/common/Statistics.cuh"
#include <vector>
//#include "core/common/io/json.h"

int main(int argc, char** argv) {
//    Statistics statistics=Statistics();
//    std::vector<double> a={5,1,2,0,3,4};
//    //0,1,2,3,4,5
//    //0,1,2,3,4,5
//    std::cout<<statistics.IQR(a);
    int totalFunctionEvaluations=ITERATION_COUNT;
//    HyperLevel* hyperLevel=new RandomHyperLevel();
//    HyperLevel* hyperLevel=new SimpleLocalSearchHyperLevel();
//    HyperLevel* hyperLevel=new SimplePerturbHyperLevel();
    HyperLevel* hyperLevel=new SimulatedAnnealingHyperLevel();

    hyperLevel->hyperOptimize(totalFunctionEvaluations);
//    cudaDeviceReset();
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

// https://github.com/jcrvz/customhys TODO check bench functions and compare, look out for iteration vs evaluation
