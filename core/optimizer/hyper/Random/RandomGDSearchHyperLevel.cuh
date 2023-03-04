//
// Created by spaceman on 2023. 02. 20..
//

#ifndef PARALLELLBFGS_RANDOMGDSEARCHHYPERLEVEL_CUH
#define PARALLELLBFGS_RANDOMGDSEARCHHYPERLEVEL_CUH

#include "../HyperLevel.cuh"
#include "RandomHyperLevel.cuh"
#include <limits>

class RandomGDHyperLevel: public RandomHyperLevel {
public:
    RandomGDHyperLevel():RandomHyperLevel("RANDOM-GD") {
    }

    std::unordered_map<std::string,OperatorParameters*> createOptimizerParameters(int totalBaseLevelEvaluations) override {
        return createSimpleGDOptimizerParameters(totalBaseLevelEvaluations);
    }

    void mutateParameters(std::unordered_map<std::string,OperatorParameters*> &chainParameters) override {
        std::cout<<"mutating by selection"<<std::endl;
        setRandomUniformSelect(chainParameters,{"RefinerGDOperatorParams"});
    }
};
#endif //PARALLELLBFGS_SIMPLEGDSEARCHHYPERLEVEL_CUH
