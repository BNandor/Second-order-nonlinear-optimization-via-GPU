//
// Created by spaceman on 2023. 02. 24..
//

#ifndef PARALLELLBFGS_RANDOMDEHYPERLEVEL_CUH
#define PARALLELLBFGS_RANDOMDEHYPERLEVEL_CUH
#include "../HyperLevel.cuh"
#include "RandomHyperLevel.cuh"
#include <limits>

class RandomDEHyperLevel: public RandomHyperLevel {
public:
    RandomDEHyperLevel():RandomHyperLevel("RANDOM-DE") {
    }

    std::unordered_map<std::string,OperatorParameters*> createOptimizerParameters(int totalBaseLevelEvaluations) override{
        return createSimpleDEOptimizerParameters(totalBaseLevelEvaluations);
    }

    void mutateParameters(std::unordered_map<std::string,OperatorParameters*> &chainParameters) override{
        std::cout<<"mutating by selection"<<std::endl;
        setRandomUniformSelect(chainParameters,{"PerturbatorDEOperatorParams"});
    }
};
#endif //PARALLELLBFGS_SIMPLEDEHYPERLEVEL_CUH
