//
// Created by spaceman on 2022. 11. 26..
//

#ifndef PARALLELLBFGS_HYPERLEVEL_CUH
#define PARALLELLBFGS_HYPERLEVEL_CUH

#include "../base/BaseLevel.cuh"
#include "../../common/Statistics.cuh"
#include "../../common/io/Logs.cuh"
#include <json.hpp>
#include <utility>
#include <vector>

using json = nlohmann::json;

#ifndef HYPER_LEVEL_TRIAL_SAMPLE_SIZE
#define HYPER_LEVEL_TRIAL_SAMPLE_SIZE 30
#endif

#ifndef  LOGS_PATH
#define LOGS_PATH "hh-logs.json"
#endif

#ifndef HH_TRIALS
#define HH_TRIALS 100
#endif

#ifndef  EXPERIMENT_HASH_SHA256
#define EXPERIMENT_HASH_SHA256 "no-hash-defined"
#endif

class HyperLevel {

protected:
    BaseLevel baseLevel=BaseLevel();
    Statistics statistics = Statistics();
    double minBaseLevelStatistic=std::numeric_limits<double>::max();
    std::unordered_map<std::string,OperatorParameters*> bestParameters=std::unordered_map<std::string,OperatorParameters*>();

    json logJson;
    int baseLevelSampleSize=HYPER_LEVEL_TRIAL_SAMPLE_SIZE;
    std::string hyperLevelId;

    HyperLevel(std::string hyperLevelId):hyperLevelId(std::move(hyperLevelId)) {
        initLogJson();
    }

    void initLogJson() {
        logJson["trials"]=std::vector<double>();
        logJson["baseLevel-sampleSize"]=baseLevelSampleSize;
        logJson["hyperLevel-id"]=hyperLevelId;
        logJson["trialCount"]=HH_TRIALS;
        logJson["experimentHashSha256"]=EXPERIMENT_HASH_SHA256;
    }

    double getPerformanceSampleOfSize(int sampleSize,std::unordered_map<std::string,OperatorParameters*> & parameters,int totalBaseLevelEvaluations){
        std::vector<double> samples;
        for(int i=0;i<sampleSize;i++) {
            baseLevel.loadInitialModel();
            double sampleF=baseLevel.optimize(&parameters,totalBaseLevelEvaluations);
            std::cout<<"ran sample number "<<i<<"/"<<sampleSize<<" with minf: "<<sampleF<<std::endl;
            samples.push_back(sampleF);
        }
        double medIqr=statistics.median(samples) + statistics.IQR(samples);
        logJson["baseLevelEvals"]=totalBaseLevelEvaluations;
        logJson["trials"].push_back({{"med_+_iqr",medIqr},{"atEval",baseLevel.totalEvaluations}});
        if(medIqr < minBaseLevelStatistic) {
            minBaseLevelStatistic=medIqr;
            cloneParameters(parameters,bestParameters);
            logJson["minBaseLevelStatistic"]=minBaseLevelStatistic;
            logJson["bestParameters"]=getParametersJson(bestParameters);
        }
        return medIqr;
    }

    json getParametersJson(std::unordered_map<std::string,OperatorParameters*> & parameters){
        json parametersJson;
        std::for_each(parameters.begin(),parameters.end(),[&parametersJson](std::pair<const std::string,OperatorParameters*> & parameter){
            parametersJson[parameter.first]=parameter.second->getJson();
        });
        return parametersJson;
    }

    void setRandomUniform(std::unordered_map<std::string,OperatorParameters*> &chainParameters) {
        std::for_each(chainParameters.begin(),chainParameters.end(),[](std::pair<const std::string,OperatorParameters*>& operatorParameter){
            operatorParameter.second->setRandomUniform();
        });
    }

    void mutateByEpsilon(std::unordered_map<std::string,OperatorParameters*> &chainParameters) {
        std::for_each(chainParameters.begin(),chainParameters.end(),[](std::pair<const std::string,OperatorParameters*>& operatorParameter){
            operatorParameter.second->mutateByEpsilon();
        });
    }

    void printParameters(std::unordered_map<std::string,OperatorParameters*> &parameters) {
        std::for_each(parameters.begin(),parameters.end(),[](std::pair<const std::string,OperatorParameters*>& operatorParameter){
            std::cout<<std::get<0>(operatorParameter)<<":"<<std::endl;
            operatorParameter.second->printParameters();
        });
    }

    void cloneParameters(std::unordered_map<std::string,OperatorParameters*> &from,std::unordered_map<std::string,OperatorParameters*> &to){
       baseLevel.cloneParameters(from,to);
    }

    void freeOperatorParamMap(std::unordered_map<std::string,OperatorParameters*> &parameters){
     baseLevel.freeOperatorParamMap(parameters);
    }

    void setDefaultOptimizerChainSimplex(std::unordered_map<std::string,OperatorParameters*>&chainParameters){
// Optimizer Chain simplex
        chainParameters["OptimizerChainInitializerSimplex"]=new SimplexParameters(
                {
                        {std::string("perturbator"),BoundedParameter(1.0,0,1)},
                        {std::string("refiner"),BoundedParameter(0.0,0,1)}
                });
        chainParameters["OptimizerChainPerturbatorSimplex"]=new SimplexParameters(
                {
                        {std::string("selector"),BoundedParameter(0.5,0,1)},
                        {std::string("refiner"),BoundedParameter(0.5,0,1)}
                });
        chainParameters["OptimizerChainRefinerSimplex"]=new SimplexParameters(
                {
                        {std::string("selector"),BoundedParameter(1.0,0,1)},
                        {std::string("refiner"),BoundedParameter(0.0,0,1)}
                });
        chainParameters["OptimizerChainSelectorSimplex"]=new SimplexParameters(
                {
                        {std::string("perturbator"),BoundedParameter(1.0,0,1)}
                });
    }
    void setLocalSearchOptimizerChainSimplex(std::unordered_map<std::string,OperatorParameters*>&chainParameters){
// Optimizer Chain simplex
        chainParameters["OptimizerChainInitializerSimplex"]=new SimplexParameters(
                {
                        {std::string("perturbator"),BoundedParameter(0.0,0,1)},
                        {std::string("refiner"),BoundedParameter(1.0,0,1)}
                });
        chainParameters["OptimizerChainPerturbatorSimplex"]=new SimplexParameters(
                {
                        {std::string("selector"),BoundedParameter(0.0,0,1)},
                        {std::string("refiner"),BoundedParameter(1.0,0,1)}
                });
        chainParameters["OptimizerChainRefinerSimplex"]=new SimplexParameters(
                {
                        {std::string("selector"),BoundedParameter(0.0,0,1)},
                        {std::string("refiner"),BoundedParameter(1.0,0,1)}
                });
        chainParameters["OptimizerChainSelectorSimplex"]=new SimplexParameters(
                {
                        {std::string("perturbator"),BoundedParameter(1.0,0,1)}
                });
    }

    void setPerturbOptimizerChainSimplex(std::unordered_map<std::string,OperatorParameters*>&chainParameters){
// Optimizer Chain simplex
        chainParameters["OptimizerChainInitializerSimplex"]=new SimplexParameters(
                {
                        {std::string("perturbator"),BoundedParameter(1.0,0,1)},
                        {std::string("refiner"),BoundedParameter(0.0,0,1)}
                });
        chainParameters["OptimizerChainPerturbatorSimplex"]=new SimplexParameters(
                {
                        {std::string("selector"),BoundedParameter(1.0,0,1)},
                        {std::string("refiner"),BoundedParameter(0.0,0,1)}
                });
        chainParameters["OptimizerChainRefinerSimplex"]=new SimplexParameters(
                {
                        {std::string("selector"),BoundedParameter(1.0,0,1)},
                        {std::string("refiner"),BoundedParameter(0.0,0,1)}
                });
        chainParameters["OptimizerChainSelectorSimplex"]=new SimplexParameters(
                {
                        {std::string("perturbator"),BoundedParameter(1.0,0,1)}
                });
    }

    std::unordered_map<std::string,OperatorParameters*> createDefaultOptimizerParameters(int totalBaseLevelEvaluations) {
        auto chainParameters=std::unordered_map<std::string,OperatorParameters*>();
        setDefaultOptimizerChainSimplex(chainParameters);
        setDefaultOperatorChainSimplex(chainParameters);
        setDefaultOperatorParameters(chainParameters,totalBaseLevelEvaluations);
        return chainParameters;
    }

    std::unordered_map<std::string,OperatorParameters*> createSimpleLocalSearchOptimizerParameters(int totalBaseLevelEvaluations) {
        auto chainParameters=std::unordered_map<std::string,OperatorParameters*>();
        setLocalSearchOptimizerChainSimplex(chainParameters);
        setDefaultOperatorChainSimplex(chainParameters);
        setDefaultOperatorParameters(chainParameters,totalBaseLevelEvaluations);
        return chainParameters;
    }

    std::unordered_map<std::string,OperatorParameters*> createSimplePerturbOptimizerParameters(int totalBaseLevelEvaluations) {
        auto chainParameters=std::unordered_map<std::string,OperatorParameters*>();
        setPerturbOptimizerChainSimplex(chainParameters);
        setDefaultOperatorChainSimplex(chainParameters);
        setDefaultOperatorParameters(chainParameters,totalBaseLevelEvaluations);
        return chainParameters;
    }

    void setDefaultOperatorChainSimplex( std::unordered_map<std::string,OperatorParameters*>&chainParameters) {
        // Operator Chain simplex
        chainParameters["PerturbatorInitializerSimplex"]=new SimplexParameters(
                {
                        {std::string("DE"),BoundedParameter(0.5,0,1)},
                        {std::string("GA"),BoundedParameter(0.5,0,1)}
                });
        chainParameters["PerturbatorDESimplex"]=new SimplexParameters(
                {
                        {std::string("DE"),BoundedParameter(0.5,0,1)},
                        {std::string("GA"),BoundedParameter(0.5,0,1)}
                });
        chainParameters["PerturbatorGASimplex"]=new SimplexParameters(
                {
                        {std::string("DE"),BoundedParameter(0.5,0,1)},
                        {std::string("GA"),BoundedParameter(0.5,0,1)}
                });

        chainParameters["RefinerInitializerSimplex"]=new SimplexParameters(
                {
                        {std::string("GD"),BoundedParameter(0,0,1)},
                        {std::string("LBFGS"),BoundedParameter(1.0,0,1)}
                });
        chainParameters["RefinerGDSimplex"]=new SimplexParameters(
                {
                        {std::string("GD"),BoundedParameter(0,0,1)},
                        {std::string("LBFGS"),BoundedParameter(1.0,0,1)}
                });
        chainParameters["RefinerLBFGSSimplex"]=new SimplexParameters(
                {
                        {std::string("GD"),BoundedParameter(0,0,1)},
                        {std::string("LBFGS"),BoundedParameter(1.0,0,1)}
                });
        chainParameters["SelectorInitializerSimplex"]=new SimplexParameters(
                {
                        {std::string("best"),BoundedParameter(1.0,0,1)}
                });
        chainParameters["SelectorBestSimplex"]=new SimplexParameters(
                {
                        {std::string("best"),BoundedParameter(1.0,0,1)}
                });
    }
    void setDefaultOperatorParameters(std::unordered_map<std::string,OperatorParameters*>&chainParameters,int totalBaseLevelEvaluations){

        // Operator parameters
        // Perturbator parameters
        std::unordered_map<std::string,BoundedParameter> deParams=std::unordered_map<std::string,BoundedParameter>();
        deParams["DE_CR"]=BoundedParameter(0.99,0.9,1.0);
        deParams["DE_FORCE"]=BoundedParameter(0.6,0.4,0.7);
        chainParameters["PerturbatorDEOperatorParams"]=new OperatorParameters(deParams);

        std::unordered_map<std::string,BoundedParameter> gaParams=std::unordered_map<std::string,BoundedParameter>();
        gaParams["GA_CR"]=BoundedParameter(0.9, 0.0, 1.0);
        gaParams["GA_CR_POINT"]=BoundedParameter(0.5, 0.0, 1.0);
        gaParams["GA_MUTATION_RATE"]=BoundedParameter(0.5, 0.0, 1.0);
        gaParams["GA_MUTATION_SIZE"]=BoundedParameter(50, 0.0, 1000);
        gaParams["GA_PARENTPOOL_RATIO"]=BoundedParameter(0.3, 0.2, 1.0);
        gaParams["GA_ALPHA"]=BoundedParameter(0.2, 0.0, 1.0);
        chainParameters["PerturbatorGAOperatorParams"]=new OperatorParameters(gaParams);

        // Refiner parameters
        std::unordered_map<std::string,BoundedParameter> gdParams=std::unordered_map<std::string,BoundedParameter>();
        gdParams["GD_ALPHA"]=BoundedParameter(ALPHA, 0.5, 5);
        gdParams["GD_FEVALS"]=BoundedParameter(totalBaseLevelEvaluations/10.0, 1,totalBaseLevelEvaluations);
//        gdParams["GD_ALPHA"]=BoundedParameter(ALPHA, 0.5, 100);
//        gdParams["GD_FEVALS"]=BoundedParameter(ITERATION_COUNT, 0, optimizerContext->to
        chainParameters["RefinerGDOperatorParams"]=new OperatorParameters(gdParams);

        std::unordered_map<std::string,BoundedParameter> lbfgsParams=std::unordered_map<std::string,BoundedParameter>();
        lbfgsParams["LBFGS_ALPHA"]=BoundedParameter(ALPHA, 0.5, 5);
        lbfgsParams["LBFGS_FEVALS"]=BoundedParameter(totalBaseLevelEvaluations/10.0, 1,totalBaseLevelEvaluations);
        lbfgsParams["LBFGS_C1"]=BoundedParameter(0.0001, 0.0, 0.1);
        lbfgsParams["LBFGS_C2"]=BoundedParameter(0.9, 0.8, 1.0);

//        lbfgsParams["LBFGS_ALPHA"]=BoundedParameter(ALPHA, 0.5, 100);
//        lbfgsParams["LBFGS_FEVALS"]=BoundedParameter(ITERATION_COUNT, 0, optimizerContext->totalFunctionEvaluations/100);
//        lbfgsParams["LBFGS_C1"]=BoundedParameter(0.0001, 0.0, 1.0);
//        lbfgsParams["LBFGS_C2"]=BoundedParameter(0.9, 0.0, 1.0);
        chainParameters["RefinerLBFGSOperatorParams"]=new OperatorParameters(lbfgsParams);
    }

public:
    void saveLogs(){
        Logs::appendLogs(logJson,LOGS_PATH);
    }
    virtual double hyperOptimize(int totalEvaluations)=0;
    virtual ~HyperLevel()= default;

};

#endif //PARALLELLBFGS_HYPERLEVEL_CUH
