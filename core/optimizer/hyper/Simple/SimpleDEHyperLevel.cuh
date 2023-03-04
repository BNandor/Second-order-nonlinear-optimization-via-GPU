//
// Created by spaceman on 2023. 02. 24..
//

#ifndef PARALLELLBFGS_SIMPLEDEHYPERLEVEL_CUH
#define PARALLELLBFGS_SIMPLEDEHYPERLEVEL_CUH
#include "../HyperLevel.cuh"
#include "SimplePerturbHyperLevel.cuh"
#include <limits>

class SimpleDEHyperLevel: public SimplePerturbHyperLevel {
public:
    SimpleDEHyperLevel():SimplePerturbHyperLevel("DE") {
    }

    std::unordered_map<std::string,OperatorParameters*> createOptimizerParameters(int totalBaseLevelEvaluations) override{
        return createSimpleDEOptimizerParameters(totalBaseLevelEvaluations);
    }
};
#endif //PARALLELLBFGS_SIMPLEDEHYPERLEVEL_CUH
