//
// Created by spaceman on 2023. 02. 20..
//

#ifndef PARALLELLBFGS_SIMPLELBFGSSEARCHHYPERLEVEL_CUH
#define PARALLELLBFGS_SIMPLELBFGSSEARCHHYPERLEVEL_CUH
#include "../HyperLevel.cuh"
#include "SimpleLocalSearchHyperLevel.cuh"
#include <limits>

class SimpleLBFGSSearchHyperLevel: public SimpleLocalSearchHyperLevel {
public:
    SimpleLBFGSSearchHyperLevel():SimpleLocalSearchHyperLevel("LBFGS") {
    }

    std::unordered_map<std::string,OperatorParameters*> createOptimizerParameters(int totalBaseLevelEvaluations) override{
        return createSimpleLBFGSOptimizerParameters(totalBaseLevelEvaluations);
    }
};
#endif //PARALLELLBFGS_SIMPLELBFGSSEARCHHYPERLEVEL_CUH
