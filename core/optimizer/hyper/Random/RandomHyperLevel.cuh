//
// Created by spaceman on 2022. 11. 26..
//

#ifndef PARALLELLBFGS_RANDOMHYPERLEVEL_CUH
#define PARALLELLBFGS_RANDOMHYPERLEVEL_CUH
#include "../HyperLevel.cuh"
#include <limits>

class RandomHyperLevel: public HyperLevel {
public:
    RandomHyperLevel():HyperLevel("random"){
    }

    RandomHyperLevel(std::string id):HyperLevel(std::move(id)) {
    }
    double hyperOptimize(int totalEvaluations) override {
        std::cout<<"Running "<<hyperLevelId<<" for "<<totalEvaluations<<" evaluations"<<std::endl;
        int trials=HH_TRIALS;
        int totalBaseLevelEvaluations=totalEvaluations;
        std::unordered_map<std::string,OperatorParameters*> defaultParameters=createOptimizerParameters(totalBaseLevelEvaluations);
        std::unordered_map<std::string,OperatorParameters*> currentParameters=std::unordered_map<std::string,OperatorParameters*>();
        cloneParameters(defaultParameters,currentParameters);
        baseLevel.init(logJson);

        for(int i=0; i < trials; i++) {
            std::cout<<"Starting trial "<<i<<std::endl;
            baseLevel.loadInitialModel();
            double currentF=getPerformanceSampleOfSize(baseLevelSampleSize,currentParameters,totalBaseLevelEvaluations);
            printf("f: %f trial %u \n",currentF, i);
            mutateParameters(currentParameters);
        }
        printParameters(bestParameters);
        baseLevel.printCurrentBestGlobalModel();
        freeOperatorParamMap(defaultParameters);
        freeOperatorParamMap(currentParameters);
        freeOperatorParamMap(bestParameters);
        return 0;
    };

    std::unordered_map<std::string,OperatorParameters*> createOptimizerParameters(int totalBaseLevelEvaluations) override{
        std::unordered_map<std::string,OperatorParameters*> params=createDefaultOptimizerParameters(totalBaseLevelEvaluations);
        setRandomUniform(params);
        return  params;
    }

    virtual void mutateParameters(std::unordered_map<std::string,OperatorParameters*> &chainParameters) {
        setRandomUniform(chainParameters);
    }

public:
    ~RandomHyperLevel() override = default;
};
#endif //PARALLELLBFGS_RANDOMHYPERLEVEL_CUH