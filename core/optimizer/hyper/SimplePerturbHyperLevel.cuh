//
// Created by spaceman on 2022. 12. 03..
//

#ifndef PARALLELLBFGS_SIMPLEPERTURBHYPERLEVEL_CUH
#define PARALLELLBFGS_SIMPLEPERTURBHYPERLEVEL_CUH


#include "HyperLevel.cuh"
#include <limits>
#include <algorithm>

class SimplePerturbHyperLevel: public HyperLevel {

    double hyperOptimize(int totalEvaluations) override {
        int trials=1;
        int totalBaseLevelEvaluations=totalEvaluations/trials;
        std::unordered_map<std::string,OperatorParameters*> defaultParameters=createSimplePerturbOptimizerParameters(totalBaseLevelEvaluations);
        std::unordered_map<std::string,OperatorParameters*> currentParameters=std::unordered_map<std::string,OperatorParameters*>();
        std::unordered_map<std::string,OperatorParameters*> bestParameters=std::unordered_map<std::string,OperatorParameters*>();
        cloneParameters(defaultParameters,currentParameters);
        baseLevel.init();

        double min=std::numeric_limits<double>::max();

        for(int i=0; i < trials; i++) {
            std::cout<<"Starting trial "<<i<<std::endl;
            baseLevel.loadInitialModel();
            double currentF=baseLevel.optimize(&currentParameters,totalBaseLevelEvaluations);
            printf("f: %f trial %u \n",currentF, i);
            if(currentF < min) {
                min=currentF;
                cloneParameters(currentParameters,bestParameters);
            }
            baseLevel.printCurrentBestGlobalModel();
        }
        printParameters(bestParameters);
        baseLevel.printCurrentBestGlobalModel();
        freeOperatorParamMap(defaultParameters);
        freeOperatorParamMap(currentParameters);
        freeOperatorParamMap(bestParameters);
        return 0;
    };

public:
    ~SimplePerturbHyperLevel() override = default;
};

#endif //PARALLELLBFGS_SIMPLEPERTURBHYPERLEVEL_CUH
