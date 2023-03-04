//
// Created by spaceman on 2023. 02. 20..
//

#ifndef PARALLELLBFGS_RANDOMLBFGSSEARCHHYPERLEVEL_CUH
#define PARALLELLBFGS_RANDOMLBFGSSEARCHHYPERLEVEL_CUH
#include "../HyperLevel.cuh"
#include "RandomHyperLevel.cuh"
#include <limits>

class RandomLBFGSHyperLevel: public RandomHyperLevel {
public:
    RandomLBFGSHyperLevel():RandomHyperLevel("RANDOM-LBFGS") {
    }

    std::unordered_map<std::string,OperatorParameters*> createOptimizerParameters(int totalBaseLevelEvaluations) override{
        return createSimpleLBFGSOptimizerParameters(totalBaseLevelEvaluations);
    }

    void mutateParameters(std::unordered_map<std::string,OperatorParameters*> &chainParameters) override {
        std::cout<<"mutating by selection"<<std::endl;
        setRandomUniformSelect(chainParameters,{"RefinerLBFGSOperatorParams"});
    }
};
#endif //PARALLELLBFGS_SIMPLELBFGSSEARCHHYPERLEVEL_CUH
