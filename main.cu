#include <iostream>
#include "core/optimizer/base/BaseLevel.cuh"
#include "core/optimizer/hyper/HyperLevel.cuh"
#include "core/optimizer/hyper/Random/RandomHyperLevel.cuh"
#include "core/optimizer/hyper/Random/RandomGDSearchHyperLevel.cuh"
#include "core/optimizer/hyper/Random/RandomLBFGSSearchHyperLevel.cuh"
#include "core/optimizer/hyper/Random/RandomGAHyperLevel.cuh"
#include "core/optimizer/hyper/Random/RandomDEHyperLevel.cuh"
#include "core/optimizer/hyper/SA/SimulatedAnnealingHyperLevel.cuh"
#include "core/optimizer/hyper/SA/SimulatedAnnealingPerturbHyperLevel.cuh"
#include "core/optimizer/hyper/Simple/SimpleLocalSearchHyperLevel.cuh"
#include "core/optimizer/hyper/Simple/SimpleGDSearchHyperLevel.cuh"
#include "core/optimizer/hyper/Simple/SimpleLBFGSSearchHyperLevel.cuh"
#include "core/optimizer/hyper/Simple/SimplePerturbHyperLevel.cuh"
#include "core/optimizer/hyper/Simple/SimpleGAHyperLevel.cuh"
#include "core/optimizer/hyper/Simple/SimpleDEHyperLevel.cuh"
#include "core/optimizer/hyper/CMA-ES/CMAESHyperLevel.cuh"
#include "core/common/Statistics.cuh"
#include "core/optimizer/hyper/SA/SimulatedAnnealingRefineHyperLevel.cuh"
#include <vector>
#include <string.h>

#ifndef HH_METHOD
#define HH_METHOD "SA"
#endif

int main(int argc, char** argv) {
//    Statistics statistics=Statistics();
//    std::vector<double> a={5,1,2,0,3,4};
//    //0,1,2,3,4,5
//    //0,1,2,3,4,5
//    std::cout<<statistics.IQR(a);
    int totalFunctionEvaluations=ITERATION_COUNT;
    HyperLevel *hyperLevel=0;
    if (STR_EQ(HH_METHOD, "RANDOM")) {
        hyperLevel=new RandomHyperLevel();
    }
    if (STR_EQ(HH_METHOD, "REFINE")) {
     hyperLevel=new SimpleLocalSearchHyperLevel();
    }
    if (STR_EQ(HH_METHOD, "PERTURB")) {
     hyperLevel=new SimplePerturbHyperLevel();
    }
    if (STR_EQ(HH_METHOD, "SA")) {
        hyperLevel = new SimulatedAnnealingHyperLevel();
    }
    if (STR_EQ(HH_METHOD, "SA_PERTURB")) {
        hyperLevel = new SimulatedAnnealingPerturbHyperLevel();
    }

    if (STR_EQ(HH_METHOD, "SA_REFINE")) {
        hyperLevel = new SimulatedAnnealingRefineHyperLevel();
    }

    if (STR_EQ(HH_METHOD, "GD")) {
        hyperLevel = new SimpleGDSearchHyperLevel();
    }

    if (STR_EQ(HH_METHOD, "LBFGS")) {
        hyperLevel = new SimpleLBFGSSearchHyperLevel();
    }

    if (STR_EQ(HH_METHOD, "GA")) {
        hyperLevel = new SimpleGAHyperLevel();
    }

    if (STR_EQ(HH_METHOD, "DE")) {
        hyperLevel = new SimpleDEHyperLevel();
    }

    if (STR_EQ(HH_METHOD, "RANDOM-GD")) {
        hyperLevel = new RandomGDHyperLevel();
    }

    if (STR_EQ(HH_METHOD, "RANDOM-LBFGS")) {
        hyperLevel = new RandomLBFGSHyperLevel();
    }

    if (STR_EQ(HH_METHOD, "RANDOM-GA")) {
        hyperLevel = new RandomGAHyperLevel();
    }

    if (STR_EQ(HH_METHOD, "RANDOM-DE")) {
        hyperLevel = new RandomDEHyperLevel();
    }

    if (STR_EQ(HH_METHOD, "CMA-ES")) {
        hyperLevel = new CMAESHyperLevel();
    }

    if(hyperLevel == 0 ) {
        std::cerr<<"Hyperlevel not selected, please define HH_METHOD as either SA,PERTURB,REFINE or RANDOM"<<std::endl;
    }
    hyperLevel->hyperOptimize(totalFunctionEvaluations);
    hyperLevel->saveLogs();
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
