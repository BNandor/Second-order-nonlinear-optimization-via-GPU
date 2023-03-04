//
// Created by spaceman on 2023. 02. 20..
//

#ifndef PARALLELLBFGS_SIMPLEGDSEARCHHYPERLEVEL_CUH
#define PARALLELLBFGS_SIMPLEGDSEARCHHYPERLEVEL_CUH

#include "../HyperLevel.cuh"
#include "SimpleLocalSearchHyperLevel.cuh"
#include <limits>

class SimpleGDSearchHyperLevel: public SimpleLocalSearchHyperLevel {
public:
    SimpleGDSearchHyperLevel():SimpleLocalSearchHyperLevel("GD") {
    }

    std::unordered_map<std::string,OperatorParameters*> createOptimizerParameters(int totalBaseLevelEvaluations) override{
        return createSimpleGDOptimizerParameters(totalBaseLevelEvaluations);
    }
};
#endif //PARALLELLBFGS_SIMPLEGDSEARCHHYPERLEVEL_CUH
