//
// Created by spaceman on 2022. 12. 01..
//

#ifndef PARALLELLBFGS_SIMULATEDANNEALINGHYPERLEVEL_CUH
#define PARALLELLBFGS_SIMULATEDANNEALINGHYPERLEVEL_CUH

#include "HyperLevel.cuh"
#include <limits>
#include <math.h>
#include <vector>

#ifndef HH_SA_TEMP
#define HH_SA_TEMP 100
#endif

#ifndef HH_SA_ALPHA
#define HH_SA_ALPHA 0.5
#endif

class SimulatedAnnealingHyperLevel: public HyperLevel {
public:
    SimulatedAnnealingHyperLevel():HyperLevel("SA"){
    }

    double hyperOptimize(int totalEvaluations) override {
        std::cout<<"Running "<<hyperLevelId<<" for "<<totalEvaluations<<" evaluations"<<std::endl;
        int trials=HH_TRIALS;
        int totalBaseLevelEvaluations=totalEvaluations;
        std::mt19937 generator=std::mt19937(std::random_device()());
        std::unordered_map<std::string,OperatorParameters*> defaultParameters=createDefaultOptimizerParameters(totalBaseLevelEvaluations);
        std::unordered_map<std::string,OperatorParameters*> currentParameters=std::unordered_map<std::string,OperatorParameters*>();
        std::unordered_map<std::string,OperatorParameters*> currentMutatedByEpsilonParameters=std::unordered_map<std::string,OperatorParameters*>();
        std::unordered_map<std::string,OperatorParameters*> bestParameters=std::unordered_map<std::string,OperatorParameters*>();
        cloneParameters(defaultParameters,currentParameters);
        setRandomUniform(currentParameters);
        baseLevel.init(logJson);

        baseLevel.loadInitialModel();
        double currentF=getPerformanceSampleOfSize(baseLevelSampleSize,currentParameters,totalBaseLevelEvaluations);
        double min=currentF;
        cloneParameters(currentParameters,bestParameters);

        double temp0=HH_SA_TEMP;
        double temp=temp0;
        double alpha=HH_SA_ALPHA;
        int acceptedWorse=0;
        logJson["SA-temp0"]=temp0;
        logJson["SA-alpha"]=alpha;
        logJson["SA-temps"].push_back(temp);
        for(int i=0; i < trials-1 || temp < 0; i++) {

            printf("[med+iqr]f: %f trial %u \n",currentF, i);
            cloneParameters(currentParameters,currentMutatedByEpsilonParameters);
            mutateByEpsilon(currentMutatedByEpsilonParameters);

            double currentFPrime=getPerformanceSampleOfSize(baseLevelSampleSize,currentMutatedByEpsilonParameters,totalBaseLevelEvaluations);
            printf("[med+iqr]f': %f trial %u \n",currentFPrime, i);
            if(currentFPrime < currentF ) {
                cloneParameters(currentMutatedByEpsilonParameters,currentParameters);
                currentF=currentFPrime;
            }else {
                double r= std::uniform_real_distribution<double>(0,1)(generator);
                if(r<exp((currentF-currentFPrime)/temp)) {
                    cloneParameters(currentMutatedByEpsilonParameters,currentParameters);
                    currentF=currentFPrime;
                    acceptedWorse++;
                    std::cout<<"Accepted worse at "<<i<<"/"<<trials-1<<" at temp: "<<temp<<std::endl;
                }
            }
            temp=temp0/(1+alpha*i);
            logJson["SA-temps"].push_back(temp);
        }

        printParameters(bestParameters);
        baseLevel.printCurrentBestGlobalModel();
        std::cout<<"Accepted worse rate"<<(double)acceptedWorse/(double)trials<<std::endl;
        freeOperatorParamMap(defaultParameters);
        freeOperatorParamMap(currentParameters);
        freeOperatorParamMap(bestParameters);
        freeOperatorParamMap(currentMutatedByEpsilonParameters);
        return 0;
    };

public:
    ~SimulatedAnnealingHyperLevel() override = default;
};
#endif //PARALLELLBFGS_SIMULATEDANNEALINGHYPERLEVEL_CUH
