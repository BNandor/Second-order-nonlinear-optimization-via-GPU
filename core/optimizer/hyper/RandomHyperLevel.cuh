//
// Created by spaceman on 2022. 11. 26..
//

#ifndef PARALLELLBFGS_RANDOMHYPERLEVEL_CUH
#define PARALLELLBFGS_RANDOMHYPERLEVEL_CUH
#include "HyperLevel.cuh"
#include <limits>

class RandomHyperLevel: public HyperLevel {

    double hyperOptimize(int totalEvaluations) override {
         std::unordered_map<std::string,OperatorParameters*> defaultParameters=createDefaultOptimizerParameters();
         std::unordered_map<std::string,OperatorParameters*> currentParameters=std::unordered_map<std::string,OperatorParameters*>();
         std::unordered_map<std::string,OperatorParameters*> bestParameters=std::unordered_map<std::string,OperatorParameters*>();
         cloneParameters(defaultParameters,currentParameters);
         setRandomUniform(currentParameters);
         baseLevel.init();
         int trials=2;
         double min=std::numeric_limits<double>::max();

         for(int i=0; i < trials; i++) {
             std::cout<<"Starting trial "<<i<<std::endl;
             baseLevel.loadInitialModel();
             double currentF=baseLevel.optimize(&currentParameters,totalEvaluations/trials);
             printf("f: %f trial %u \n",currentF, i);
             if(currentF < min) {
                 min=currentF;
                 cloneParameters(currentParameters,bestParameters);
             }
             setRandomUniform(currentParameters);
         }
         printParameters(bestParameters);
         printf("\nfinal f: %.10f", min);
         return 0;
    };

public:
    ~RandomHyperLevel() override = default;
};
#endif //PARALLELLBFGS_RANDOMHYPERLEVEL_CUH