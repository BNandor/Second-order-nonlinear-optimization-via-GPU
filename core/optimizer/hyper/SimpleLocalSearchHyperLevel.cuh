//
// Created by spaceman on 2022. 12. 03..
//

#ifndef PARALLELLBFGS_SIMPLELOCALSEARCHHYPERLEVEL_CUH
#define PARALLELLBFGS_SIMPLELOCALSEARCHHYPERLEVEL_CUH

#include "HyperLevel.cuh"
#include <limits>

class SimpleLocalSearchHyperLevel: public HyperLevel {

    double hyperOptimize(int totalEvaluations) override {
        int trials=1;
        int totalBaseLevelEvaluations=totalEvaluations/trials;
        std::unordered_map<std::string,OperatorParameters*> defaultParameters=createSimpleLocalSearchOptimizerParameters(totalBaseLevelEvaluations);
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
        }
        printParameters(bestParameters);
        printf("\nfinal f: %.10f", min);
        return 0;
    };

public:
    ~SimpleLocalSearchHyperLevel() override = default;
};

#endif //PARALLELLBFGS_SIMPLELOCALSEARCHHYPERLEVEL_CUH
