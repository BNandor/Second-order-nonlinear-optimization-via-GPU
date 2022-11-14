//
// Created by spaceman on 2022. 11. 09..
//

#ifndef PARALLELLBFGS_OPTIMIZINGMARKOVCHAIN_CUH
#define PARALLELLBFGS_OPTIMIZINGMARKOVCHAIN_CUH
#include <random>
#include "../../common/OptimizerContext.cuh"
#include "MarkovChain.cuh"
#include "OptimizingMarkovNode.cuh"
#include <unordered_map>
#include <algorithm>

class OptimizingMarkovChain: public MarkovChain {
    CUDAMemoryModel* cudaMemoryModel;
    Metrics* metrics;
public:

    OptimizingMarkovChain(OptimizerContext* optimizerContext, Metrics* metrics): cudaMemoryModel(&optimizerContext->cudaMemoryModel) {
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
        nodes["initializer"]->addNext(nodes["perturbator"],1.0);
        nodes["initializer"]->addNext(nodes["refiner"],0.0);

        nodes["perturbator"]->addNext(nodes["selector"],0.2);
        nodes["perturbator"]->addNext(nodes["refiner"],0.8);

        nodes["refiner"]->addNext(nodes["refiner"],0.0);
        nodes["refiner"]->addNext(nodes["selector"],1.0);

        nodes["selector"]->addNext(nodes["perturbator"],1.0);
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
        (*perturbatorChain)["initializer"]->addNext((*perturbatorChain)["DE"],0.5);
        (*perturbatorChain)["initializer"]->addNext((*perturbatorChain)["GA"],0.5);
        (*perturbatorChain)["DE"]->addNext((*perturbatorChain)["DE"],0.5);
        (*perturbatorChain)["DE"]->addNext((*perturbatorChain)["GA"],0.5);
        (*perturbatorChain)["GA"]->addNext((*perturbatorChain)["DE"],0.5);
        (*perturbatorChain)["GA"]->addNext((*perturbatorChain)["GA"],0.5);
        return new OperatorMarkovChain(*perturbatorChain,metrics);
    }

    OperatorMarkovChain* buildRefinerChain(OptimizerContext* optimizerContext) {
        auto* refinerChain=new std::unordered_map<std::string,MarkovNode*>();
        (*refinerChain)["initializer"]=new OperatorMarkovNode(new Initializer(), std::string("initializer").c_str());
        (*refinerChain)["GD"]=new OperatorMarkovNode(&optimizerContext->gdLocalSearch, std::string("GD").c_str());
        (*refinerChain)["LBFGS"]=new OperatorMarkovNode(&optimizerContext->lbfgsLocalSearch, std::string("LBFGS").c_str());
        (*refinerChain)["initializer"]->addNext((*refinerChain)["LBFGS"], 0.5);
        (*refinerChain)["initializer"]->addNext((*refinerChain)["GD"], 0.5);
        (*refinerChain)["LBFGS"]->addNext((*refinerChain)["LBFGS"], 0.5);
        (*refinerChain)["LBFGS"]->addNext((*refinerChain)["GD"], 0.5);
        (*refinerChain)["GD"]->addNext((*refinerChain)["LBFGS"], 0.5);
        (*refinerChain)["GD"]->addNext((*refinerChain)["GD"], 0.5);
        return new OperatorMarkovChain(*refinerChain, metrics);
    }

    OperatorMarkovChain* buildSelectorChain(OptimizerContext* optimizerContext) {
        auto* selectorChain=new std::unordered_map<std::string,MarkovNode*>();
        (*selectorChain)["initializer"]=new OperatorMarkovNode(new Initializer(), std::string("initializer").c_str());
        (*selectorChain)["best"]=new OperatorMarkovNode(&optimizerContext->bestSelector, std::string("best").c_str());
        (*selectorChain)["initializer"]->addNext((*selectorChain)["best"], 1.0);
        (*selectorChain)["best"]->addNext((*selectorChain)["best"], 1.0);
        return new OperatorMarkovChain(*selectorChain, metrics);
    }
};

#endif //PARALLELLBFGS_OPTIMIZINGMARKOVCHAIN_CUH
