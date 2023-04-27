//
// Created by spaceman on 2023. 04. 15..
//

#ifndef PARALLELLBFGS_SA_CMAESHYPERLEVEL_CUH
#define PARALLELLBFGS_SA_CMAESHYPERLEVEL_CUH

#include "../HyperLevel.cuh"
#include <libcmaes/cmastrategy.h>
#include <libcmaes/cmaes.h>
#include <libcmaes/esoptimizer.h>
#include <limits>
#include <math.h>
#include <utility>
#include <vector>
#include <unordered_set>

#ifndef HH_CMAESLambda
#define HH_CMAESLambda 5
#endif

using namespace libcmaes;

class SA_CMAESHyperLevel: public HyperLevel {
    std::unordered_map<std::string,OperatorParameters*> parameters;
    std::unordered_set<std::string> CMAES_parameters;
    int totalBaseLevelEvaluations;
    int currentSteps=0;

public:
    FitFunc baseLevelPerformanceMedIQR = [this](const double *operatorParameters, const int N)
    {
        std::cout<<"Running "<<hyperLevelId<<" for "<<totalBaseLevelEvaluations<<" evaluations"<<std::endl;
        updateOperatorParams(parameters,operatorParameters,CMAES_parameters);
        for(int i=0;i<N;i++){
            std::cout<<operatorParameters[i]<<" ";
        }
        double result= getPerformanceSampleOfSize(baseLevelSampleSize,parameters,totalBaseLevelEvaluations);
        currentSteps+=1;
        std::cout<<"final result "<<result<<"at step"<<currentSteps<<std::endl;
        return result;
    };

    SA_CMAESHyperLevel():HyperLevel("SA-CMA-ES") {
    }

    SA_CMAESHyperLevel(std::string id):HyperLevel(std::move(id)) {
    }

    double hyperOptimize(int totalEvaluations) override {
        int totalTrials=HH_TRIALS;
        int SA_trials=totalTrials/3;
        int SA_CMA_ES_trials=totalTrials-SA_trials;
        baseLevel.init(logJson);
        simulatedAnnealingInitialization(totalEvaluations,SA_trials);
        cloneParameters(bestParameters,parameters);
        CMAES_parameters={"OptimizerChainInitializerSimplex",
                          "OptimizerChainRefinerSimplex",
                          "OptimizerChainPerturbatorSimplex",
                          "PerturbatorDESimplex",
                          "PerturbatorGASimplex",
                          "RefinerLBFGSSimplex",
                          "RefinerGDSimplex"};


        totalBaseLevelEvaluations=totalEvaluations;
        std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,std::vector<double>> serializedParameters=getParameterArrays(parameters,CMAES_parameters);
        int dim = std::get<0>(serializedParameters).size();
        double sigma = 1;
        int lambda = -1;
//        GenoPheno<pwqBoundStrategy> gp(,dim); // genotype / phenotype transform associated to bounds.
//        CMAParameters<> cmaparams(std::get<0>(serializedParameters),sigma,lambda);
        CMAParameters<GenoPheno<pwqBoundStrategy,linScalingStrategy>> cmaparams(std::get<0>(serializedParameters),std::get<3>(serializedParameters),
                                                                                lambda,
                                                                                std::get<1>(serializedParameters),
                                                                                std::get<2>(serializedParameters),0); // -1 for automatically decided lambda, 0 is for random	seeding	of the internal generator.
        cmaparams.set_quiet(false);
        cmaparams.set_restarts(0);
        cmaparams.set_max_iter(SA_CMA_ES_trials);
        cmaparams.set_max_fevals(SA_CMA_ES_trials);

        // set the maximum number of function evaluations
        cmaparams.set_algo(sepaCMAES);
//        cmaparams.set_restarts(0);
        CMASolutions cmasols = cmaes<GenoPheno<pwqBoundStrategy,linScalingStrategy>>(baseLevelPerformanceMedIQR,cmaparams);
        std::cout<<"point1"<<std::endl;
        printParameters(bestParameters);
        std::cout<<"point2"<<std::endl;
        baseLevel.printCurrentBestGlobalModel();
        return 0;
    };

    void simulatedAnnealingInitialization(int totalEvaluations,int trials ){
        std::cout<<"Running simulated annealing with "<<totalEvaluations<<" for"<<trials<<" steps"<<std::endl;
        int totalSABaseLevelEvaluations=totalEvaluations;
        std::mt19937 generator=std::mt19937(std::random_device()());
        std::unordered_map<std::string,OperatorParameters*> defaultParameters=createOptimizerParameters(totalSABaseLevelEvaluations);
        std::unordered_map<std::string,OperatorParameters*> currentParameters=std::unordered_map<std::string,OperatorParameters*>();
        std::unordered_map<std::string,OperatorParameters*> currentMutatedByEpsilonParameters=std::unordered_map<std::string,OperatorParameters*>();
        cloneParameters(defaultParameters,currentParameters);

        baseLevel.loadInitialModel();
        double currentF=getPerformanceSampleOfSize(baseLevelSampleSize,currentParameters,totalSABaseLevelEvaluations);
        currentSteps+=1;
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

            double currentFPrime=getPerformanceSampleOfSize(baseLevelSampleSize,currentMutatedByEpsilonParameters,totalSABaseLevelEvaluations);
            currentSteps+=1;
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
        freeOperatorParamMap(defaultParameters);
        freeOperatorParamMap(currentParameters);
        freeOperatorParamMap(currentMutatedByEpsilonParameters);
    }

    virtual std::unordered_map<std::string,OperatorParameters*>  createOptimizerParameters(int totalBaseLevelEvaluations) {
        std::unordered_map<std::string,OperatorParameters*> param=createDefaultOptimizerParameters(totalBaseLevelEvaluations);
//        setRandomUniform(param);
        return param;
    }

    virtual void mutateParameters(std::unordered_map<std::string,OperatorParameters*> &chainParameters) {
        mutateAllParametersByEpsilon(chainParameters);
    }

    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,std::vector<double>> getParameterArrays(
            const std::unordered_map<std::string, OperatorParameters*>& operatorParamsMap,
            const std::unordered_set<std::string>& paramsToInclude
    ) {
        // Initialize the arrays to be returned
        std::vector<double> x, l, u, sigma;

        // Iterate over the operator parameters and add their values, lower bounds, and upper bounds to the arrays
        for (const auto& paramsTuple : operatorParamsMap) {
//            if (paramsToInclude.find(paramsTuple.first) == paramsToInclude.end()) {
//                continue;
//            }
            OperatorParameters* operatorParams = paramsTuple.second;
            for (auto& paramPair : operatorParams->values) {
                BoundedParameter& param = paramPair.second;
                x.push_back(param.value);
                l.push_back(param.lowerBound);
                u.push_back(param.upperBound);
                sigma.push_back((param.upperBound-param.lowerBound));
            }
        }
        return std::make_tuple(x, l, u,sigma);
    }

    void updateOperatorParams(std::unordered_map<std::string, OperatorParameters*>& operatorParamsMap, const double* x, const std::unordered_set<std::string>& paramsToInclude) {
        size_t index = 0;
        // Iterate over the operator parameters and update the values of the parameters in the set
        for (const auto& paramsTuple : operatorParamsMap) {
//            if (paramsToInclude.find(paramsTuple.first) == paramsToInclude.end()) {
//                continue;
//            }
            OperatorParameters* operatorParams = paramsTuple.second;
            for (auto& paramPair : operatorParams->values) {
                BoundedParameter& param = paramPair.second;
                param.value=x[index];
                ++index;
            }
        }
    }


public:
    ~SA_CMAESHyperLevel() {
        freeOperatorParamMap(parameters);
    };
};


#endif //PARALLELLBFGS_SA_CMAESHYPERLEVEL_CUH
