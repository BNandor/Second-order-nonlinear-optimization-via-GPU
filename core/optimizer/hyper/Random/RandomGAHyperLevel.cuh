//
// Created by spaceman on 2023. 02. 24..
//

#ifndef PARALLELLBFGS_RANDOMGDHYPERLEVEL_CUH
#define PARALLELLBFGS_RANDOMGDHYPERLEVEL_CUH
#include "../HyperLevel.cuh"
#include "RandomHyperLevel.cuh"
#include <limits>

class RandomGAHyperLevel: public RandomHyperLevel {
public:
    RandomGAHyperLevel():RandomHyperLevel("RANDOM-GA") {
    }

    std::unordered_map<std::string,OperatorParameters*> createOptimizerParameters(int totalBaseLevelEvaluations) override{
        return createSimpleGAOptimizerParameters(totalBaseLevelEvaluations);
    }

     void mutateParameters(std::unordered_map<std::string,OperatorParameters*> &chainParameters) override{
        std::cout<<"mutating by selection"<<std::endl;
        setRandomUniformSelect(chainParameters,{"PerturbatorGAOperatorParams"});
    }
};
#endif //PARALLELLBFGS_SIMPLEGDHYPERLEVEL_CUH
