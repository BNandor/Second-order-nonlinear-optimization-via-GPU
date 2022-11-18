//
// Created by spaceman on 2022. 11. 09..
//

#ifndef PARALLELLBFGS_OPTIMIZINGMARKOVCHAIN_CUH
#define PARALLELLBFGS_OPTIMIZINGMARKOVCHAIN_CUH
#include <random>
#include "../../../common/OptimizerContext.cuh"
#include "../MarkovChain.cuh"
#include "OptimizingMarkovNode.cuh"
#include <unordered_map>
#include <algorithm>
#include "./parameters/SimplexParameters.cuh"

class OptimizingMarkovChain: public MarkovChain {
    CUDAMemoryModel* cudaMemoryModel;
    Metrics* metrics;
    std::unordered_map<std::string,OperatorParameters*> parameters;
public:

    OptimizingMarkovChain(OptimizerContext* optimizerContext, Metrics* metrics): cudaMemoryModel(&optimizerContext->cudaMemoryModel) {
        setupOptimizerParameters();
        this->metrics=metrics;
        nodes=std::unordered_map<std::string,MarkovNode*>();
        nodes["initializer"]=new OptimizingMarkovNode(buildInitializerChain(),std::string("initializer").c_str());
        nodes["perturbator"]=new OptimizingMarkovNode(buildPerturbatorChain(optimizerContext),std::string("perturbator").c_str());
        nodes["refiner"]=new OptimizingMarkovNode(buildRefinerChain(optimizerContext),std::string("refiner").c_str());
        nodes["selector"]=new OptimizingMarkovNode(buildSelectorChain(optimizerContext),std::string("selector").c_str());
        buildChain();
        currentNode=nodes["initializer"];
    }

    ~OptimizingMarkovChain() {
        std::for_each(nodes.begin(),nodes.end(),[](auto node){delete std::get<1>(node);});
    }

    void hopToNext() override {
        currentNode=currentNode->getNext(generator);
        std::cout<<"hopped to "<<currentNode->name<<std::endl;
        metrics->modelPerformanceMetrics.markovIterations++;
    }

    void operate() {
//        std::cout<<"operating: "<<currentNode->name<<std::endl;
        ((OptimizingMarkovNode*)currentNode)->operate(cudaMemoryModel);
        metrics->modelPerformanceMetrics.fEvaluations+=((OptimizingMarkovNode*)currentNode)->fEvals();
    }

    void buildChain() {
        nodes["initializer"]->addNext(nodes["perturbator"],parameters["OptimizerChainInitializerSimplex"]->values["perturbator"].value);
        nodes["initializer"]->addNext(nodes["refiner"],parameters["OptimizerChainInitializerSimplex"]->values["refiner"].value);

        nodes["perturbator"]->addNext(nodes["selector"],parameters["OptimizerChainPerturbatorSimplex"]->values["selector"].value);
        nodes["perturbator"]->addNext(nodes["refiner"],parameters["OptimizerChainPerturbatorSimplex"]->values["refiner"].value);

        nodes["refiner"]->addNext(nodes["refiner"],parameters["OptimizerChainRefinerSimplex"]->values["refiner"].value);
        nodes["refiner"]->addNext(nodes["selector"],parameters["OptimizerChainRefinerSimplex"]->values["selector"].value);

        nodes["selector"]->addNext(nodes["perturbator"], parameters["OptimizerChainSelectorSimplex"]->values["perturbator"].value);
    }

    OperatorMarkovChain *buildInitializerChain() {
        auto* initializerChain=new std::unordered_map<std::string,MarkovNode*>();
        (*initializerChain)["initializer"]=new OperatorMarkovNode(new Initializer(),std::string("initializer").c_str());
        return new OperatorMarkovChain(*initializerChain,metrics);
    }

    OperatorMarkovChain* buildPerturbatorChain(OptimizerContext* optimizerContext) {
        auto* perturbatorChain=new std::unordered_map<std::string,MarkovNode*>();
        (*perturbatorChain)["initializer"]=new OperatorMarkovNode(new Initializer(),std::string("initializer").c_str());
        (*perturbatorChain)["DE"]=new OperatorMarkovNode(&optimizerContext->differentialEvolutionContext,std::string("DE").c_str());
        (*perturbatorChain)["GA"]=new OperatorMarkovNode(&optimizerContext->geneticAlgorithmContext,std::string("GA").c_str());
        (*perturbatorChain)["initializer"]->addNext((*perturbatorChain)["DE"],parameters["PerturbatorInitializerSimplex"]->values["DE"].value);
        (*perturbatorChain)["initializer"]->addNext((*perturbatorChain)["GA"],parameters["PerturbatorInitializerSimplex"]->values["GA"].value);
        (*perturbatorChain)["DE"]->addNext((*perturbatorChain)["DE"],parameters["PerturbatorDESimplex"]->values["DE"].value);
        (*perturbatorChain)["DE"]->addNext((*perturbatorChain)["GA"],parameters["PerturbatorDESimplex"]->values["GA"].value);
        (*perturbatorChain)["GA"]->addNext((*perturbatorChain)["DE"],parameters["PerturbatorGASimplex"]->values["DE"].value);
        (*perturbatorChain)["GA"]->addNext((*perturbatorChain)["GA"],parameters["PerturbatorGASimplex"]->values["GA"].value);
        return new OperatorMarkovChain(*perturbatorChain,metrics);
    }

    OperatorMarkovChain* buildRefinerChain(OptimizerContext* optimizerContext) {
        auto* refinerChain=new std::unordered_map<std::string,MarkovNode*>();
        (*refinerChain)["initializer"]=new OperatorMarkovNode(new Initializer(), std::string("initializer").c_str());
        (*refinerChain)["GD"]=new OperatorMarkovNode(&optimizerContext->gdLocalSearch, std::string("GD").c_str());
        (*refinerChain)["LBFGS"]=new OperatorMarkovNode(&optimizerContext->lbfgsLocalSearch, std::string("LBFGS").c_str());
        (*refinerChain)["initializer"]->addNext((*refinerChain)["LBFGS"], parameters["RefinerInitializerSimplex"]->values["LBFGS"].value);
        (*refinerChain)["initializer"]->addNext((*refinerChain)["GD"], parameters["RefinerInitializerSimplex"]->values["GD"].value);
        (*refinerChain)["LBFGS"]->addNext((*refinerChain)["LBFGS"], parameters["RefinerLBFGSSimplex"]->values["LBFGS"].value);
        (*refinerChain)["LBFGS"]->addNext((*refinerChain)["GD"], parameters["RefinerLBFGSSimplex"]->values["GD"].value);
        (*refinerChain)["GD"]->addNext((*refinerChain)["LBFGS"], parameters["RefinerGDSimplex"]->values["LBFGS"].value);
        (*refinerChain)["GD"]->addNext((*refinerChain)["GD"], parameters["RefinerGDSimplex"]->values["GD"].value);
        return new OperatorMarkovChain(*refinerChain, metrics);
    }

    OperatorMarkovChain* buildSelectorChain(OptimizerContext* optimizerContext) {
        auto* selectorChain=new std::unordered_map<std::string,MarkovNode*>();
        (*selectorChain)["initializer"]=new OperatorMarkovNode(new Initializer(), std::string("initializer").c_str());
        (*selectorChain)["best"]=new OperatorMarkovNode(&optimizerContext->bestSelector, std::string("best").c_str());
        (*selectorChain)["initializer"]->addNext((*selectorChain)["best"], parameters["SelectorInitializerSimplex"]->values["best"].value);
        (*selectorChain)["best"]->addNext((*selectorChain)["best"], parameters["SelectorBestSimplex"]->values["best"].value);
        return new OperatorMarkovChain(*selectorChain, metrics);
    }

    void setupOptimizerParameters() {
        parameters=std::unordered_map<std::string,OperatorParameters*>();
        // Optimizer Chain simplex
        parameters["OptimizerChainInitializerSimplex"]=new SimplexParameters(
                {
                    {std::string("perturbator"),BoundedParameter(1.0,0,1)},
                        {std::string("refiner"),BoundedParameter(0.0,0,1)}
                });
        parameters["OptimizerChainPerturbatorSimplex"]=new SimplexParameters(
                {
                        {std::string("selector"),BoundedParameter(0.5,0,1)},
                        {std::string("refiner"),BoundedParameter(0.5,0,1)}
                });
        parameters["OptimizerChainRefinerSimplex"]=new SimplexParameters(
                {
                        {std::string("selector"),BoundedParameter(1.0,0,1)},
                        {std::string("refiner"),BoundedParameter(0.0,0,1)}
                });
        parameters["OptimizerChainSelectorSimplex"]=new SimplexParameters(
                {
                        {std::string("perturbator"),BoundedParameter(1.0,0,1)}
                });



        // Operator Chain simplex
        parameters["PerturbatorInitializerSimplex"]=new SimplexParameters(
                {
                        {std::string("DE"),BoundedParameter(0.5,0,1)},
                        {std::string("GA"),BoundedParameter(0.5,0,1)}
                });
        parameters["PerturbatorDESimplex"]=new SimplexParameters(
                {
                        {std::string("DE"),BoundedParameter(0.5,0,1)},
                        {std::string("GA"),BoundedParameter(0.5,0,1)}
                });
        parameters["PerturbatorGASimplex"]=new SimplexParameters(
                {
                        {std::string("DE"),BoundedParameter(0.5,0,1)},
                        {std::string("GA"),BoundedParameter(0.5,0,1)}
                });

        parameters["RefinerInitializerSimplex"]=new SimplexParameters(
                {
                        {std::string("GD"),BoundedParameter(0.5,0,1)},
                        {std::string("LBFGS"),BoundedParameter(0.5,0,1)}
                });
        parameters["RefinerGDSimplex"]=new SimplexParameters(
                {
                        {std::string("GD"),BoundedParameter(0.5,0,1)},
                        {std::string("LBFGS"),BoundedParameter(0.5,0,1)}
                });
        parameters["RefinerLBFGSSimplex"]=new SimplexParameters(
                {
                        {std::string("GD"),BoundedParameter(0.5,0,1)},
                        {std::string("LBFGS"),BoundedParameter(0.5,0,1)}
                });

        parameters["SelectorInitializerSimplex"]=new SimplexParameters(
                {
                        {std::string("best"),BoundedParameter(1.0,0,1)}
                });
        parameters["SelectorBestSimplex"]=new SimplexParameters(
                {
                        {std::string("best"),BoundedParameter(1.0,0,1)}
                });

    }
};

#endif //PARALLELLBFGS_OPTIMIZINGMARKOVCHAIN_CUH
