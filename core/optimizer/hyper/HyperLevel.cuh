//
// Created by spaceman on 2022. 11. 26..
//

#ifndef PARALLELLBFGS_HYPERLEVEL_CUH
#define PARALLELLBFGS_HYPERLEVEL_CUH
#include "../base/BaseLevel.cuh"

class HyperLevel {
protected:
    BaseLevel baseLevel=BaseLevel();
    void setRandomUniform(std::unordered_map<std::string,OperatorParameters*> &chainParameters) {
        std::for_each(chainParameters.begin(),chainParameters.end(),[](auto& operatorParameter){
            std::get<1>(operatorParameter)->setRandomUniform();
        });
    }

    void printParameters(std::unordered_map<std::string,OperatorParameters*> &parameters) {
        std::for_each(parameters.begin(),parameters.end(),[](auto& operatorParameter){
            std::cout<<std::get<0>(operatorParameter)<<":"<<std::endl;
            std::get<1>(operatorParameter)->printParameters();
        });
    }

    void cloneParameters(std::unordered_map<std::string,OperatorParameters*> &from,std::unordered_map<std::string,OperatorParameters*> &to){
        std::for_each(from.begin(),from.end(),[&to,&from](auto& operatorParameter){
            to[std::get<0>(operatorParameter)]=std::get<1>(operatorParameter)->clone();
        });
    }

    std::unordered_map<std::string,OperatorParameters*> createDefaultOptimizerParameters() {
        auto chainParameters=std::unordered_map<std::string,OperatorParameters*>();
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
                        {std::string("GD"),BoundedParameter(0.5,0,1)},
                        {std::string("LBFGS"),BoundedParameter(0.5,0,1)}
                });
        chainParameters["RefinerGDSimplex"]=new SimplexParameters(
                {
                        {std::string("GD"),BoundedParameter(0.5,0,1)},
                        {std::string("LBFGS"),BoundedParameter(0.5,0,1)}
                });
        chainParameters["RefinerLBFGSSimplex"]=new SimplexParameters(
                {
                        {std::string("GD"),BoundedParameter(0.5,0,1)},
                        {std::string("LBFGS"),BoundedParameter(0.5,0,1)}
                });
        chainParameters["SelectorInitializerSimplex"]=new SimplexParameters(
                {
                        {std::string("best"),BoundedParameter(1.0,0,1)}
                });
        chainParameters["SelectorBestSimplex"]=new SimplexParameters(
                {
                        {std::string("best"),BoundedParameter(1.0,0,1)}
                });

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
        gaParams["GA_MUTATION_SIZE"]=BoundedParameter(50, 0.0, 100000);
        gaParams["GA_PARENTPOOL_RATIO"]=BoundedParameter(0.3, 0.2, 1.0);
        gaParams["GA_ALPHA"]=BoundedParameter(0.2, 0.0, 1.0);
        chainParameters["PerturbatorGAOperatorParams"]=new OperatorParameters(gaParams);

        // Refiner parameters
        std::unordered_map<std::string,BoundedParameter> gdParams=std::unordered_map<std::string,BoundedParameter>();
        gdParams["GD_ALPHA"]=BoundedParameter(ALPHA, 0.5, 5);
        gdParams["GD_FEVALS"]=BoundedParameter(ITERATION_COUNT, 1,ITERATION_COUNT);
//        gdParams["GD_ALPHA"]=BoundedParameter(ALPHA, 0.5, 100);
//        gdParams["GD_FEVALS"]=BoundedParameter(ITERATION_COUNT, 0, optimizerContext->to
        chainParameters["RefinerGDOperatorParams"]=new OperatorParameters(gdParams);

        std::unordered_map<std::string,BoundedParameter> lbfgsParams=std::unordered_map<std::string,BoundedParameter>();
        lbfgsParams["LBFGS_ALPHA"]=BoundedParameter(ALPHA, 0.5, 5);
        lbfgsParams["LBFGS_FEVALS"]=BoundedParameter(ITERATION_COUNT, 1,ITERATION_COUNT);
        lbfgsParams["LBFGS_C1"]=BoundedParameter(0.0001, 0.0, 0.1);
        lbfgsParams["LBFGS_C2"]=BoundedParameter(0.9, 0.8, 1.0);

//        lbfgsParams["LBFGS_ALPHA"]=BoundedParameter(ALPHA, 0.5, 100);
//        lbfgsParams["LBFGS_FEVALS"]=BoundedParameter(ITERATION_COUNT, 0, optimizerContext->totalFunctionEvaluations/100);
//        lbfgsParams["LBFGS_C1"]=BoundedParameter(0.0001, 0.0, 1.0);
//        lbfgsParams["LBFGS_C2"]=BoundedParameter(0.9, 0.0, 1.0);
        chainParameters["RefinerLBFGSOperatorParams"]=new OperatorParameters(lbfgsParams);
        return chainParameters;
    }

public:

    virtual double hyperOptimize(int totalEvaluations)=0;
    virtual ~HyperLevel()= default;

};

#endif //PARALLELLBFGS_HYPERLEVEL_CUH
