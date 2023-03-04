//
// Created by spaceman on 2022. 12. 01..
//

#ifndef PARALLELLBFGS_CMAESHYPERLEVEL_CUH
#define PARALLELLBFGS_CMAESHYPERLEVEL_CUH

#include "../HyperLevel.cuh"
#include <libcmaes/cmastrategy.h>
#include <libcmaes/esoptimizer.h>
#include <limits>
#include <math.h>
#include <utility>
#include <vector>

#ifndef HH_CMAESLambda
#define HH_CMAESLambda 5
#endif

using namespace libcmaes;



class CMAESHyperLevel: public HyperLevel {
public:
    FitFunc cigtab = [this](const double *x, const int N)
    {
        std::cout<<"Running "<<hyperLevelId<<" for "<<100<<" evaluations"<<std::endl;
        int trials=HH_TRIALS;
        int totalBaseLevelEvaluations=100;
        std::mt19937 generator=std::mt19937(std::random_device()());
        std::unordered_map<std::string,OperatorParameters*> defaultParameters=createOptimizerParameters(totalBaseLevelEvaluations);
        std::unordered_map<std::string,OperatorParameters*> currentParameters=std::unordered_map<std::string,OperatorParameters*>();
        std::unordered_map<std::string,OperatorParameters*> currentMutatedByEpsilonParameters=std::unordered_map<std::string,OperatorParameters*>();
        std::unordered_map<std::string,OperatorParameters*> bestParameters=std::unordered_map<std::string,OperatorParameters*>();
        cloneParameters(defaultParameters,currentParameters);

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
            mutateParameters(currentMutatedByEpsilonParameters);

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
    CMAESHyperLevel():HyperLevel("CMA-ES") {
    }

    CMAESHyperLevel(std::string id):HyperLevel(std::move(id)) {
    }

    double hyperOptimize(int totalEvaluations) override {

        int dim = 10;
        std::vector<double> x0 = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
        double sigma = 0.2;
        int lambda = 10;
        CMAParameters<> cmaparams(x0,sigma,lambda);
        ESOptimizer<CMAStrategy<CovarianceUpdate>,CMAParameters<>> cmaes(cigtab,cmaparams);
        cmaes.optimize();
        double edm = cmaes.edm();
        std::cout << "EDM=" << edm << " / EDM/fm=" << edm / cmaes.get_solutions().best_candidate().get_fvalue() << std::endl;
    };

    virtual std::unordered_map<std::string,OperatorParameters*>  createOptimizerParameters(int totalBaseLevelEvaluations) {
        std::unordered_map<std::string,OperatorParameters*> param=createDefaultOptimizerParameters(totalBaseLevelEvaluations);
        setRandomUniform(param);
        return param;
    }

    virtual void mutateParameters(std::unordered_map<std::string,OperatorParameters*> &chainParameters) {
        mutateAllParametersByEpsilon(chainParameters);
    }

public:
    ~CMAESHyperLevel() override = default;
};
#endif //PARALLELLBFGS_SIMULATEDANNEALINGHYPERLEVEL_CUH
