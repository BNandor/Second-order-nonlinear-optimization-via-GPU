//
// Created by spaceman on 2022. 11. 26..
//

#ifndef PARALLELLBFGS_HYPERLEVEL_CUH
#define PARALLELLBFGS_HYPERLEVEL_CUH

#include "../base/BaseLevel.cuh"
#include "../../common/Statistics.cuh"
#include "../../common/CommonStringUtils.h"
#include "../../common/io/JsonOperations.cuh"
#include "../../json.hpp"
#include "./sampling/SPRTTTest.h"
#include <utility>
#include <vector>

#define STR_EQ(x, y) (strcmp(x, y) == 0)

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

#ifndef  SAMPLING
#define SAMPLING "fixed-size-samples"
#endif

#ifndef  SAMPLING_SPRTT_ALPHA
#define SAMPLING_SPRTT_ALPHA 0.05
#endif

#ifndef  SAMPLING_SPRTT_BETA
#define SAMPLING_SPRTT_BETA 0.05
#endif

#ifndef  SAMPLING_SPRTT_COHENS_D
#define  SAMPLING_SPRTT_COHENS_D 1
#endif

class HyperLevel {
protected:
    BaseLevel baseLevel=BaseLevel();
    Statistics statistics = Statistics();
    double minBaseLevelStatistic=std::numeric_limits<double>::max();
    std::vector<double> bestSamples;
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
        std::cout<<"calculating medIQR "<<std::endl;
        double medIqr=statistics.median(samples) + statistics.IQR(samples);
        std::cout<<"updating json logs1 "<<std::endl;
        logJson["baseLevelEvals"]=totalBaseLevelEvaluations;
        std::cout<<"updating json logs2 "<<std::endl;
        logJson["trials"].push_back({{"med_+_iqr",medIqr},{"atEval",baseLevel.totalEvaluations},{"performanceSamples",samples}});

        if(medIqr < minBaseLevelStatistic) {
            bestSamples.assign(samples.begin(), samples.end());
            std::cout<<"updating best solution "<<std::endl;
            minBaseLevelStatistic=medIqr;
            std::cout<<"cloning best parameters "<<std::endl;
            cloneParameters(parameters,bestParameters);
            std::cout<<"setting base level statistics "<<std::endl;
            logJson["minBaseLevelStatistic"]=minBaseLevelStatistic;
            logJson["bestParameters"]=getParametersJson(bestParameters);
        }
        return medIqr;
    }

    double getPerformanceSampleOfSize(int sampleSize,std::unordered_map<std::string,OperatorParameters*> & parameters,int totalBaseLevelEvaluations,int &usedSampleCount) {
        if(sampleSize < 3 ) {
            std::cerr<<"Invalid performance sample size "<<sampleSize<<std::endl;
            return std::numeric_limits<double>::max();
        }
        double best_mean=SPRTTTest::mean(bestSamples);
        usedSampleCount=0;
        std::vector<double> samples;
        //Start with minimal number of samples
        for(int i=0;i<4&& i<sampleSize;i++) {
            baseLevel.loadInitialModel();
            double sampleF=baseLevel.optimize(&parameters,totalBaseLevelEvaluations);
            std::cout<<"ran sample number "<<i<<"/"<<sampleSize<<" with minf: "<<sampleF<<std::endl;
            samples.push_back(sampleF);
            ++usedSampleCount;
        }
        int hypothesis=-1;
        for(int i=4;i<sampleSize;i++) {
            hypothesis=SPRTTTest::checkIfNewMeanIsLessThanComparisonSequentialT(bestSamples,samples,SAMPLING_SPRTT_ALPHA,SAMPLING_SPRTT_BETA,SAMPLING_SPRTT_COHENS_D);
            if(hypothesis == 0 ){
                break;
            }
            baseLevel.loadInitialModel();
            double sampleF=baseLevel.optimize(&parameters,totalBaseLevelEvaluations);
            std::cout<<"ran sample number "<<i<<"/"<<sampleSize<<" with minf: "<<sampleF<<std::endl;
            samples.push_back(sampleF);
            ++usedSampleCount;
        }
        double currentAvg=SPRTTTest::mean(samples);
        if(hypothesis == 0 ) {
            printf("new mean (%f) >= best mean (%f)\n",currentAvg, best_mean);
        }
        if(hypothesis == 1 ) {
            printf("new mean (%f) < best mean (%f)\n",currentAvg, best_mean);
        }
        if(hypothesis == -1 ) {
            printf("inconclusive SPRTTest.\n");
        }
        printf("Sample size required:%d\n", usedSampleCount);

//        std::cout<<"calculating medIQR "<<std::endl;
        double medIqr=statistics.median(samples) + statistics.IQR(samples);
//        std::cout<<"updating json logs1 "<<std::endl;
        logJson["baseLevelEvals"]=totalBaseLevelEvaluations;
//        std::cout<<"updating json logs2 "<<std::endl;
        logJson["trials"].push_back({{"med_+_iqr",medIqr},{"atEval",baseLevel.totalEvaluations},{"performanceSamples",samples}});

        if( currentAvg < best_mean && hypothesis !=0 ) {
//            std::cout<<"updating best solution "<<std::endl;
            minBaseLevelStatistic=medIqr;
            bestSamples.assign(samples.begin(), samples.end());
//            std::cout<<"cloning best parameters "<<std::endl;
            cloneParameters(parameters,bestParameters);
//            std::cout<<"setting base level statistics "<<std::endl;
            logJson["minBaseLevelStatistic"]=minBaseLevelStatistic;
            logJson["bestParameters"]=getParametersJson(bestParameters);
        }
        return medIqr;
    }

    json getParametersJson(std::unordered_map<std::string,OperatorParameters*> & parameters){
        json parametersJson;
        std::for_each(parameters.begin(),parameters.end(),[&parametersJson](auto& parameter){
            parametersJson[std::get<0>(parameter)]=std::get<1>(parameter)->getJson();
        });
        return parametersJson;
    }

//    void setParametersJson(std::string  parametersPath,std::unordered_map<std::string,OperatorParameters*> & parameters)
//    {
//        json paramsJson=JsonOperations::loadJsonFrom(parametersPath);
//        std::for_each(paramsJson.begin(),paramsJson.end(),[&parameters](auto &param) {
//            std::for_each(param.value().begin(),param.value().end(),[&parameters,&param](auto &v) {
//                std::cout<<"Setting"<<param.key()<<"-"<<v.key()<<" to "<<v.value()<<std::endl;
//                parameters[param.key()]->setParameterValue(v.key(),v.value());
//            });
//        });
//    }

    void setRandomUniform(std::unordered_map<std::string,OperatorParameters*> &chainParameters) {
        std::for_each(chainParameters.begin(),chainParameters.end(),[](auto& operatorParameter){
            std::get<1>(operatorParameter)->setRandomUniform();
        });
    }

    void setRandomUniformSelect(std::unordered_map<std::string,OperatorParameters*> &chainParameters, std::list<std::string> parameters) {
        std::for_each(parameters.begin(),parameters.end(),[&chainParameters](auto& operatorParameter){
            std::cout<<"mutating "<<operatorParameter<<std::endl;
            chainParameters[operatorParameter]->setRandomUniform();
        });
    }

    void mutateAllParametersByEpsilon(std::unordered_map<std::string,OperatorParameters*> &chainParameters) {
        std::for_each(chainParameters.begin(),chainParameters.end(),[](auto& operatorParameter){
            std::get<1>(operatorParameter)->mutateByEpsilon();
        });
    }

    void mutateSelectParametersByEpsilon(std::unordered_map<std::string,OperatorParameters*> &chainParameters,std::list<std::string> parameters)
    {
        std::for_each(parameters.begin(),parameters.end(),[&chainParameters](auto& operatorParameter){
            std::cout<<"mutating "<<operatorParameter<<std::endl;
            chainParameters[operatorParameter]->mutateByEpsilon();
            std::cout<<"mutated "<<operatorParameter<<std::endl;
        });
    }

    void printParameters(std::unordered_map<std::string,OperatorParameters*> &parameters) {
        std::cout<<"accessing paraemters"<<std::endl;
        std::cout<<parameters.size()<<std::endl;
        std::cout<<"accessed parameters"<<std::endl;
        std::for_each(parameters.begin(),parameters.end(),[](auto& operatorParameter){
            std::cout<<std::get<0>(operatorParameter)<<":"<<std::endl;
            std::cout<<"accessing paraemter value at"<<std::get<1>(operatorParameter)<<std::endl;
            std::get<1>(operatorParameter)->printParameters();
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
                        {std::string("refiner"),BoundedParameter(0.0,0,0.9)}
                });
        chainParameters["OptimizerChainSelectorSimplex"]=new SimplexParameters(
                {
                        {std::string("perturbator"),BoundedParameter(1.0,0,1)}
                });
    }
    virtual std::unordered_map<std::string,OperatorParameters*> createOptimizerParameters(int totalBaseLevelEvaluations)=0;

    std::unordered_map<std::string,OperatorParameters*> createDefaultOptimizerParameters(int totalBaseLevelEvaluations) {
        auto chainParameters=std::unordered_map<std::string,OperatorParameters*>();
        setDefaultOptimizerChainSimplex(chainParameters);
        printf("default optimizer chain simplex\n");
        setDefaultOperatorChainSimplex(chainParameters);
        printf("default operator chain simplex\n");
        setDefaultOperatorParameters(chainParameters,totalBaseLevelEvaluations);
        printf("default operator parameters\n");
        return chainParameters;
    }

    std::unordered_map<std::string,OperatorParameters*> createSimpleLocalSearchOptimizerParameters(int totalBaseLevelEvaluations) {
        auto chainParameters=std::unordered_map<std::string,OperatorParameters*>();
        setLocalSearchOptimizerChainSimplex(chainParameters);
        setDefaultOperatorChainSimplex(chainParameters);
        setDefaultOperatorParameters(chainParameters,totalBaseLevelEvaluations);
        return chainParameters;
    }

    std::unordered_map<std::string,OperatorParameters*> createSimpleGDOptimizerParameters(int totalBaseLevelEvaluations) {
        auto chainParameters=std::unordered_map<std::string,OperatorParameters*>();
        setLocalSearchOptimizerChainSimplex(chainParameters);
        setDefaultPerturbOperatorChainSimplex(chainParameters);
        setGDOperatorChainSimplex(chainParameters);
        setSelectorOperatorChainSimplex(chainParameters);
        setDefaultOperatorParameters(chainParameters,totalBaseLevelEvaluations);
        return chainParameters;
    }

    std::unordered_map<std::string,OperatorParameters*> createSimpleGAOptimizerParameters(int totalBaseLevelEvaluations) {
        auto chainParameters=std::unordered_map<std::string,OperatorParameters*>();
        setPerturbOptimizerChainSimplex(chainParameters);
        setGAOperatorChainSimplex(chainParameters);
        setDefaultOperatorParameters(chainParameters,totalBaseLevelEvaluations);
        return chainParameters;
    }

    std::unordered_map<std::string,OperatorParameters*> createSimpleDEOptimizerParameters(int totalBaseLevelEvaluations) {
        auto chainParameters=std::unordered_map<std::string,OperatorParameters*>();
        setPerturbOptimizerChainSimplex(chainParameters);
        setDEOperatorChainSimplex(chainParameters);
        setDefaultOperatorParameters(chainParameters,totalBaseLevelEvaluations);
        return chainParameters;
    }

    std::unordered_map<std::string,OperatorParameters*> createSimpleLBFGSOptimizerParameters(int totalBaseLevelEvaluations) {
        auto chainParameters=std::unordered_map<std::string,OperatorParameters*>();
        setLocalSearchOptimizerChainSimplex(chainParameters);
        setDefaultPerturbOperatorChainSimplex(chainParameters);
        setLBFGSOperatorChainSimplex(chainParameters);
        setSelectorOperatorChainSimplex(chainParameters);
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
        setDefaultPerturbOperatorChainSimplex(chainParameters);
        setLBFGSOperatorChainSimplex(chainParameters);
        setSelectorOperatorChainSimplex(chainParameters);
    }

    void setGAOperatorChainSimplex( std::unordered_map<std::string,OperatorParameters*>&chainParameters) {
        // Operator Chain simplex
        setGAPerturbOperatorChainSimplex(chainParameters);
        setLBFGSOperatorChainSimplex(chainParameters);
        setSelectorOperatorChainSimplex(chainParameters);
    }

    void setDEOperatorChainSimplex( std::unordered_map<std::string,OperatorParameters*>&chainParameters) {
        // Operator Chain simplex
        setDEPerturbOperatorChainSimplex(chainParameters);
        setLBFGSOperatorChainSimplex(chainParameters);
        setSelectorOperatorChainSimplex(chainParameters);
    }


    void setDefaultPerturbOperatorChainSimplex(
            std::unordered_map<std::string, OperatorParameters *> &chainParameters) {
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

        addExtraOperatorSimplex(chainParameters);
    }

void addExtraOperatorSimplex(std::unordered_map<std::string, OperatorParameters *> &chainParameters) {
#ifdef BASE_PERTURB_EXTRA_OPERATORS
    std::string operators=BASE_PERTURB_EXTRA_OPERATORS;
    std::set<std::string> operatorSet=stringutil::splitString(operators,',');
    printf("set of extra operators\n");

    std::for_each(operatorSet.begin(),operatorSet.end(),[this,&chainParameters,&operatorSet](std::string op){
        chainParameters["PerturbatorInitializerSimplex"]->values.insert({op,BoundedParameter(0.0,0,1)});
        chainParameters["PerturbatorDESimplex"]->values.insert({op,BoundedParameter(0.0,0,1)});
        chainParameters["PerturbatorGASimplex"]->values.insert({op,BoundedParameter(0.0,0,1)});
        chainParameters["Perturbator"+op+"Simplex"]=new SimplexParameters({{"DE",BoundedParameter(0.5,0,1)},
                                                                           {"GA",BoundedParameter(0.5,0,1)}});
        std::for_each(operatorSet.begin(),operatorSet.end(),[&op,&chainParameters](std::string op2) {
            chainParameters["Perturbator"+op+"Simplex"]->values.insert({op2,BoundedParameter(0.0,0,1)});
        });
    });
//    std::cout<<getParametersJson(chainParameters);
#endif
    }

    void setGAPerturbOperatorChainSimplex(
            std::unordered_map<std::string, OperatorParameters *> &chainParameters) {
        chainParameters["PerturbatorInitializerSimplex"]=new SimplexParameters(
                {
                        {std::string("DE"),BoundedParameter(0.0,0,1)},
                        {std::string("GA"),BoundedParameter(1.0,0,1)}
                });
        chainParameters["PerturbatorDESimplex"]=new SimplexParameters(
                {
                        {std::string("DE"),BoundedParameter(0.0,0,1)},
                        {std::string("GA"),BoundedParameter(1.0,0,1)}
                });
        chainParameters["PerturbatorGASimplex"]=new SimplexParameters(
                {
                        {std::string("DE"),BoundedParameter(0.0,0,1)},
                        {std::string("GA"),BoundedParameter(1.0,0,1)}
                });
        addExtraOperatorSimplex(chainParameters);
    }

    void setDEPerturbOperatorChainSimplex(
            std::unordered_map<std::string, OperatorParameters *> &chainParameters) {
        chainParameters["PerturbatorInitializerSimplex"]=new SimplexParameters(
                {
                        {std::string("DE"),BoundedParameter(1.0,0,1)},
                        {std::string("GA"),BoundedParameter(0.0,0,1)},
                        {std::string("GWO"),BoundedParameter(0.0,0,1)}

                });
        chainParameters["PerturbatorDESimplex"]=new SimplexParameters(
                {
                        {std::string("DE"),BoundedParameter(1.0,0,1)},
                        {std::string("GA"),BoundedParameter(0.0,0,1)},
                        {std::string("GWO"),BoundedParameter(0.0,0,1)}

                });
        chainParameters["PerturbatorGASimplex"]=new SimplexParameters(
                {
                        {std::string("DE"),BoundedParameter(1.0,0,1)},
                        {std::string("GA"),BoundedParameter(0.0,0,1)},
                        {std::string("GWO"),BoundedParameter(0.0,0,1)}
                });
        addExtraOperatorSimplex(chainParameters);
    }

    void setSelectorOperatorChainSimplex(std::unordered_map<std::string, OperatorParameters *> &chainParameters) const {
        chainParameters["SelectorInitializerSimplex"]=new SimplexParameters(
                {
                        {std::string("best"),BoundedParameter(1.0,0,1)}
                });
        chainParameters["SelectorBestSimplex"]=new SimplexParameters(
                {
                        {std::string("best"),BoundedParameter(1.0,0,1)}
                });
    }

    void setGDOperatorChainSimplex( std::unordered_map<std::string,OperatorParameters*>&chainParameters) {
        chainParameters["RefinerInitializerSimplex"]=new SimplexParameters(
                {
                        {std::string("GD"),BoundedParameter(1.0,0,1)},
                        {std::string("LBFGS"),BoundedParameter(0.0,0,1)}
                });
        chainParameters["RefinerGDSimplex"]=new SimplexParameters(
                {
                        {std::string("GD"),BoundedParameter(1.0,0,1)},
                        {std::string("LBFGS"),BoundedParameter(0.0,0,1)}
                });
        chainParameters["RefinerLBFGSSimplex"]=new SimplexParameters(
                {
                        {std::string("GD"),BoundedParameter(1.0,0,1)},
                        {std::string("LBFGS"),BoundedParameter(0.0,0,1)}
                });
    }

    void setLBFGSOperatorChainSimplex( std::unordered_map<std::string,OperatorParameters*>&chainParameters) {
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
    }
        void setDefaultOperatorParameters(std::unordered_map<std::string,OperatorParameters*>&chainParameters,int totalBaseLevelEvaluations){

        // Operator parameters
        // Perturbator parameters
        std::unordered_map<std::string,BoundedParameter> deParams=std::unordered_map<std::string,BoundedParameter>();
        deParams["DE_CR"]=BoundedParameter(0.7,0.0,1.0);
        deParams["DE_FORCE"]=BoundedParameter(0.6,0,2);
        chainParameters["PerturbatorDEOperatorParams"]=new OperatorParameters(deParams);

        std::unordered_map<std::string,BoundedParameter> gaParams=std::unordered_map<std::string,BoundedParameter>();
        gaParams["GA_CR"]=BoundedParameter(0.9, 0.0, 1.0);
        gaParams["GA_CR_POINT"]=BoundedParameter(0.5, 0.0, 1.0);
        gaParams["GA_MUTATION_RATE"]=BoundedParameter(0.05, 0.0, 0.5);
        gaParams["GA_MUTATION_SIZE"]=BoundedParameter(50, 0.0, 100);
        gaParams["GA_PARENTPOOL_RATIO"]=BoundedParameter(0.3, 0.2, 1.0);
        gaParams["GA_ALPHA"]=BoundedParameter(0.2, 0.0, 1.0);
        chainParameters["PerturbatorGAOperatorParams"]=new OperatorParameters(gaParams);
        #ifdef BASE_PERTURB_EXTRA_OPERATORS
            std::string operators=BASE_PERTURB_EXTRA_OPERATORS;
            std::set<std::string> operatorSet=stringutil::splitString(operators,',');
            std::for_each(operatorSet.begin(),operatorSet.end(),[&chainParameters,operatorSet](std::string op) {
                if (op == "GWO") {
                    std::unordered_map<std::string,BoundedParameter> gwoParams=std::unordered_map<std::string,BoundedParameter>();
                    gwoParams["GWO_a"]=BoundedParameter(1.0, 0.0, 2.0);
                    chainParameters["PerturbatorGWOOperatorParams"] = new OperatorParameters(gwoParams);
                }
                if (op == "GWO2") {
                    std::unordered_map<std::string,BoundedParameter> gwoParams=std::unordered_map<std::string,BoundedParameter>();
                    gwoParams["GWO_a"]=BoundedParameter(1.0, 0.0, 2.0);
                    chainParameters["PerturbatorGWO2OperatorParams"] = new OperatorParameters(gwoParams);
                }
                if (op == "DE2") {
                    std::unordered_map<std::string,BoundedParameter> deParams=std::unordered_map<std::string,BoundedParameter>();
                    deParams["DE_CR"]=BoundedParameter(0.7,0.0,1.0);
                    deParams["DE_FORCE"]=BoundedParameter(0.6,0,2);
                    chainParameters["PerturbatorDE2OperatorParams"]=new OperatorParameters(deParams);
                }
                if (op == "GA2") {
                    std::unordered_map<std::string,BoundedParameter> gaParams=std::unordered_map<std::string,BoundedParameter>();
                    gaParams["GA_CR"]=BoundedParameter(0.9, 0.0, 1.0);
                    gaParams["GA_CR_POINT"]=BoundedParameter(0.5, 0.0, 1.0);
                    gaParams["GA_MUTATION_RATE"]=BoundedParameter(0.05, 0.0, 0.5);
                    gaParams["GA_MUTATION_SIZE"]=BoundedParameter(50, 0.0, 100);
                    gaParams["GA_PARENTPOOL_RATIO"]=BoundedParameter(0.3, 0.2, 1.0);
                    gaParams["GA_ALPHA"]=BoundedParameter(0.2, 0.0, 1.0);
                    chainParameters["PerturbatorGA2OperatorParams"]=new OperatorParameters(gaParams);
                }
            });
        #endif
        // Refiner parameters
        std::unordered_map<std::string,BoundedParameter> gdParams=std::unordered_map<std::string,BoundedParameter>();
        gdParams["GD_ALPHA"]=BoundedParameter(ALPHA, 0.5, 5);
        gdParams["GD_FEVALS"]=BoundedParameter(1, 1,3);
//        gdParams["GD_ALPHA"]=BoundedParameter(ALPHA, 0.5, 100);
//        gdParams["GD_FEVALS"]=BoundedParameter(ITERATION_COUNT, 0, optimizerContext->to
        chainParameters["RefinerGDOperatorParams"]=new OperatorParameters(gdParams);

        std::unordered_map<std::string,BoundedParameter> lbfgsParams=std::unordered_map<std::string,BoundedParameter>();
        lbfgsParams["LBFGS_ALPHA"]=BoundedParameter(ALPHA, 0.5, 5);
        lbfgsParams["LBFGS_FEVALS"]=BoundedParameter(6, 6,10);
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
        JsonOperations::appendLogs(logJson, LOGS_PATH);
    }
    virtual double hyperOptimize(int totalEvaluations)=0;
    virtual ~HyperLevel()= default;

};

#endif //PARALLELLBFGS_HYPERLEVEL_CUH
