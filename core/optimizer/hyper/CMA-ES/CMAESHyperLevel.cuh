//
// Created by spaceman on 2022. 12. 01..
//

#ifndef PARALLELLBFGS_CMAESHYPERLEVEL_CUH
#define PARALLELLBFGS_CMAESHYPERLEVEL_CUH

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

class CMAESHyperLevel: public HyperLevel {
    std::unordered_map<std::string,OperatorParameters*> parameters;
    std::unordered_set<std::string> CMAES_parameters;
    int totalBaseLevelEvaluations;

public:
    FitFunc baseLevelPerformanceMedIQR = [this](const double *operatorParameters, const int N)
    {
        std::cout<<"Running "<<hyperLevelId<<" for "<<totalBaseLevelEvaluations<<" evaluations"<<std::endl;
//        std::mt19937 generator=std::mt19937(std::random_device()());
        updateOperatorParams(parameters,operatorParameters,CMAES_parameters);
        for(int i=0;i<N;i++){
            std::cout<<operatorParameters[i]<<" ";
        }
//        std::unordered_map<std::string,OperatorParameters*> defaultParameters=parameters;
//        std::unordered_map<std::string,OperatorParameters*> currentParameters=std::unordered_map<std::string,OperatorParameters*>();
//        std::unordered_map<std::string,OperatorParameters*> currentMutatedByEpsilonParameters=std::unordered_map<std::string,OperatorParameters*>();
//        std::unordered_map<std::string,OperatorParameters*> bestParameters=std::unordered_map<std::string,OperatorParameters*>();
//        cloneParameters(defaultParameters,currentParameters);

        baseLevel.init(logJson);
        baseLevel.loadInitialModel();
        return getPerformanceSampleOfSize(baseLevelSampleSize,parameters,totalBaseLevelEvaluations);

//        double currentF=getPerformanceSampleOfSize(baseLevelSampleSize,currentParameters,totalBaseLevelEvaluations);
//        double min=currentF;
//        cloneParameters(currentParameters,bestParameters);

//        double temp0=HH_SA_TEMP;
//        double temp=temp0;
//        double alpha=HH_SA_ALPHA;
//        int acceptedWorse=0;
//        logJson["SA-temp0"]=temp0;
//        logJson["SA-alpha"]=alpha;
//        logJson["SA-temps"].push_back(temp);
//        for(int i=0; i < trials || temp < 0; i++) {
//
//            printf("[med+iqr]f: %f trial %u \n",currentF, i);
//            cloneParameters(currentParameters,currentMutatedByEpsilonParameters);
//            mutateParameters(currentMutatedByEpsilonParameters);

//            double currentFPrime=getPerformanceSampleOfSize(baseLevelSampleSize,currentMutatedByEpsilonParameters,totalBaseLevelEvaluations);
//            printf("[med+iqr]f': %f trial %u \n",currentFPrime, i);
//            if(currentFPrime < currentF ) {
//                cloneParameters(currentMutatedByEpsilonParameters,currentParameters);
//                currentF=currentFPrime;
//            }else {
//                double r= std::uniform_real_distribution<double>(0,1)(generator);
//                if(r<exp((currentF-currentFPrime)/temp)) {
//                    cloneParameters(currentMutatedByEpsilonParameters,currentParameters);
//                    currentF=currentFPrime;
//                    acceptedWorse++;
//                    std::cout<<"Accepted worse at "<<i<<"/"<<trials-1<<" at temp: "<<temp<<std::endl;
//                }
//            }
//            temp=temp0/(1+alpha*i);
//            logJson["SA-temps"].push_back(temp);
//        }

//        printParameters(bestParameters);
//        baseLevel.printCurrentBestGlobalModel();
//        std::cout<<"Accepted worse rate"<<(double)acceptedWorse/(double)trials<<std::endl;

//        freeOperatorParamMap(currentParameters);
//        freeOperatorParamMap(bestParameters);
//        freeOperatorParamMap(currentMutatedByEpsilonParameters);
//        return 0;
    };

    CMAESHyperLevel():HyperLevel("CMA-ES") {
    }

    CMAESHyperLevel(std::string id):HyperLevel(std::move(id)) {
    }

    double hyperOptimize(int totalEvaluations) override {
        parameters=createOptimizerParameters(totalEvaluations);
        CMAES_parameters={"OptimizerChainInitializerSimplex",
                          "OptimizerChainRefinerSimplex",
                          "OptimizerChainPerturbatorSimplex",
                          "PerturbatorDESimplex",
                          "PerturbatorGASimplex",
                          "RefinerLBFGSSimplex",
                          "RefinerGDSimplex"};
#ifdef BASE_PERTURB_EXTRA_OPERATORS
        std::string operators=BASE_PERTURB_EXTRA_OPERATORS;
        std::set<std::string> operatorSet=stringutil::splitString(operators,',');
        std::for_each(operatorSet.begin(),operatorSet.end(),[this](std::string op){
            CMAES_parameters.insert("Perturbator"+op+"Simplex");
            CMAES_parameters.insert("Perturbator"+op+"OperatorParams");
        });
#endif

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
        cmaparams.set_max_iter(HH_TRIALS);
        cmaparams.set_max_fevals(HH_TRIALS);  // set the maximum number of function evaluations
        cmaparams.set_algo(sepaCMAES);
//        cmaparams.set_restarts(0);
        CMASolutions cmasols = cmaes<GenoPheno<pwqBoundStrategy,linScalingStrategy>>(baseLevelPerformanceMedIQR,cmaparams);
        printParameters(bestParameters);
        baseLevel.printCurrentBestGlobalModel();
        return 0;
    };

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
    ~CMAESHyperLevel() {
        freeOperatorParamMap(parameters);
    };
};
#endif //PARALLELLBFGS_SIMULATEDANNEALINGHYPERLEVEL_CUH
