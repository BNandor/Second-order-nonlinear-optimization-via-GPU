//
// Created by spaceman on 2023. 02. 24..
//

#ifndef PARALLELLBFGS_SIMPLEGDHYPERLEVEL_CUH
#define PARALLELLBFGS_SIMPLEGDHYPERLEVEL_CUH
#include "../HyperLevel.cuh"
#include "SimplePerturbHyperLevel.cuh"
#include <limits>

class SimpleGAHyperLevel: public SimplePerturbHyperLevel {
public:
    SimpleGAHyperLevel(): SimplePerturbHyperLevel("GA") {
    }

    std::unordered_map<std::string,OperatorParameters*> createOptimizerParameters(int totalBaseLevelEvaluations) override{
        return createSimpleGAOptimizerParameters(totalBaseLevelEvaluations);
    }
};
#endif //PARALLELLBFGS_SIMPLEGDHYPERLEVEL_CUH
