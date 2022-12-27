//
// Created by spaceman on 2022. 11. 26..
//

#ifndef PARALLELLBFGS_RANDOMHYPERLEVEL_CUH
#define PARALLELLBFGS_RANDOMHYPERLEVEL_CUH
#include "HyperLevel.cuh"
#include <limits>

class RandomHyperLevel: public HyperLevel {
public:
    RandomHyperLevel():HyperLevel("random"){
    }
    double hyperOptimize(int totalEvaluations) override {
        int trials=HH_TRIALS;
        int totalBaseLevelEvaluations=totalEvaluations;
        std::unordered_map<std::string,OperatorParameters*> defaultParameters=createDefaultOptimizerParameters(totalBaseLevelEvaluations);
        std::unordered_map<std::string,OperatorParameters*> currentParameters=std::unordered_map<std::string,OperatorParameters*>();
        cloneParameters(defaultParameters,currentParameters);
        setRandomUniform(currentParameters);
        baseLevel.init(logJson);

        for(int i=0; i < trials; i++) {
            std::cout<<"Starting trial "<<i<<std::endl;
            baseLevel.loadInitialModel();
            double currentF=baseLevel.optimize(&currentParameters,totalBaseLevelEvaluations);
            printf("f: %f trial %u \n",currentF, i);
            setRandomUniform(currentParameters);
        }
        printParameters(bestParameters);
        baseLevel.printCurrentBestGlobalModel();
        freeOperatorParamMap(defaultParameters);
        freeOperatorParamMap(currentParameters);
        freeOperatorParamMap(bestParameters);
        return 0;
    };

public:
    ~RandomHyperLevel() override = default;
};
#endif //PARALLELLBFGS_RANDOMHYPERLEVEL_CUH