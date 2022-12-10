//
// Created by spaceman on 2022. 12. 01..
//

#ifndef PARALLELLBFGS_SIMULATEDANNEALINGHYPERLEVEL_CUH
#define PARALLELLBFGS_SIMULATEDANNEALINGHYPERLEVEL_CUH

#include "HyperLevel.cuh"
#include <limits>
#include <math.h>

class SimulatedAnnealingHyperLevel: public HyperLevel {

    // TODO free parameters
    double hyperOptimize(int totalEvaluations) override {
        int trials=100;
        int totalBaseLevelEvaluations=totalEvaluations;
        std::mt19937 generator=std::mt19937(std::random_device()());
        std::unordered_map<std::string,OperatorParameters*> defaultParameters=createDefaultOptimizerParameters(totalBaseLevelEvaluations);
        std::unordered_map<std::string,OperatorParameters*> currentParameters=std::unordered_map<std::string,OperatorParameters*>();
        std::unordered_map<std::string,OperatorParameters*> currentMutatedByEpsilonParameters=std::unordered_map<std::string,OperatorParameters*>();
        std::unordered_map<std::string,OperatorParameters*> bestParameters=std::unordered_map<std::string,OperatorParameters*>();
        cloneParameters(defaultParameters,currentParameters);
        setRandomUniform(currentParameters);
        baseLevel.init();

        baseLevel.loadInitialModel();
        double currentF=baseLevel.optimize(&currentParameters,totalBaseLevelEvaluations);
        double min=currentF;
        cloneParameters(currentParameters,bestParameters);
        baseLevel.updateCurrentBestGlobalModel();

        double temp0=100;
        double temp=temp0;
        double alpha=0.5;
        int acceptedWorse=0;
        for(int i=0; i < trials-1 || temp < 0; i++) {
            printf("f: %f trial %u \n",currentF, i);
//            printParameters(currentParameters);
            cloneParameters(currentParameters,currentMutatedByEpsilonParameters);
            mutateByEpsilon(currentMutatedByEpsilonParameters);

            baseLevel.loadInitialModel();
            double currentFPrime=baseLevel.optimize(&currentMutatedByEpsilonParameters,totalBaseLevelEvaluations);
            printf("f': %f trial %u \n",currentFPrime, i);
//            printParameters(currentMutatedByEpsilonParameters);

            if(currentFPrime < currentF ) {
                cloneParameters(currentMutatedByEpsilonParameters,currentParameters);
                currentF=currentFPrime;
            }else{
                double r= std::uniform_real_distribution<double>(0,1)(generator);
                if(r<exp((currentF-currentFPrime)/temp)){
                    cloneParameters(currentMutatedByEpsilonParameters,currentParameters);
                    currentF=currentFPrime;
                    acceptedWorse++;
                }
            }
            if(currentF < min) {
                min=currentF;
                cloneParameters(currentParameters,bestParameters);
                baseLevel.updateCurrentBestGlobalModel();
            }
            temp=temp0/(1+alpha*i);
        }

        printParameters(bestParameters);
        baseLevel.printCurrentBestGlobalModel();
        std::cout<<"Accepted worse rate"<<(double)acceptedWorse/(double)trials<<std::endl;
        return 0;
    };

public:
    ~SimulatedAnnealingHyperLevel() override = default;
};
#endif //PARALLELLBFGS_SIMULATEDANNEALINGHYPERLEVEL_CUH
