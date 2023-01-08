//
// Created by spaceman on 2022. 12. 03..
//

#ifndef PARALLELLBFGS_SIMPLELOCALSEARCHHYPERLEVEL_CUH
#define PARALLELLBFGS_SIMPLELOCALSEARCHHYPERLEVEL_CUH

#include "HyperLevel.cuh"
#include <limits>

class SimpleLocalSearchHyperLevel: public HyperLevel {
public:
    SimpleLocalSearchHyperLevel():HyperLevel("LocalSearch"){
    }

    double hyperOptimize(int totalEvaluations) override {
        std::cout<<"Running "<<hyperLevelId<<" for "<<totalEvaluations<<" evaluations"<<std::endl;
        int trials=HH_TRIALS;
        int totalBaseLevelEvaluations=totalEvaluations;
        std::unordered_map<std::string,OperatorParameters*> defaultParameters=createSimpleLocalSearchOptimizerParameters(totalBaseLevelEvaluations);
        std::unordered_map<std::string,OperatorParameters*> currentParameters=std::unordered_map<std::string,OperatorParameters*>();
        std::unordered_map<std::string,OperatorParameters*> bestParameters=std::unordered_map<std::string,OperatorParameters*>();
        cloneParameters(defaultParameters,currentParameters);
        baseLevel.init(logJson);
        for(int i=0; i < trials; i++) {
            std::cout<<"Starting trial "<<i<<std::endl;
            double currentF=getPerformanceSampleOfSize(baseLevelSampleSize,currentParameters,totalBaseLevelEvaluations);
            printf("f: %f trial %u \n",currentF, i);
        }
        printParameters(bestParameters);
        baseLevel.printCurrentBestGlobalModel();
        freeOperatorParamMap(defaultParameters);
        freeOperatorParamMap(currentParameters);
        freeOperatorParamMap(bestParameters);
        return 0;
    };

public:
    ~SimpleLocalSearchHyperLevel() override = default;
};

#endif //PARALLELLBFGS_SIMPLELOCALSEARCHHYPERLEVEL_CUH
