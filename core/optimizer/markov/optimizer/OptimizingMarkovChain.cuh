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
#include <utility>
#include "./parameters/SimplexParameters.cuh"

class OptimizingMarkovChain: public MarkovChain {
    CUDAMemoryModel* cudaMemoryModel;
    Metrics* metrics;
    std::unordered_map<std::string,OperatorParameters*>* chainParameters;
    OperatorMarkovNode* bestSelector;

    void resetChainBasedOnParameters(OptimizerContext* optimizerContext) {
        deleteCurrentNodes();
        nodes=std::unordered_map<std::string,MarkovNode*>();
        nodes["initializer"]=new OptimizingMarkovNode(buildInitializerChain(),std::string("initializer").c_str());
        nodes["perturbator"]=new OptimizingMarkovNode(buildPerturbatorChain(optimizerContext),std::string("perturbator").c_str());
        nodes["refiner"]=new OptimizingMarkovNode(buildRefinerChain(optimizerContext),std::string("refiner").c_str());
        nodes["selector"]=new OptimizingMarkovNode(buildSelectorChain(optimizerContext),std::string("selector").c_str());
        buildOptimizerChain();
        currentNode=nodes["initializer"];
    }

    void buildOptimizerChain() {
        nodes["initializer"]->addNext(nodes["perturbator"], (*chainParameters)["OptimizerChainInitializerSimplex"]->values["perturbator"].value);
        nodes["initializer"]->addNext(nodes["refiner"], (*chainParameters)["OptimizerChainInitializerSimplex"]->values["refiner"].value);

        nodes["perturbator"]->addNext(nodes["selector"], (*chainParameters)["OptimizerChainPerturbatorSimplex"]->values["selector"].value);
        nodes["perturbator"]->addNext(nodes["refiner"], (*chainParameters)["OptimizerChainPerturbatorSimplex"]->values["refiner"].value);

        nodes["refiner"]->addNext(nodes["refiner"], (*chainParameters)["OptimizerChainRefinerSimplex"]->values["refiner"].value);
        nodes["refiner"]->addNext(nodes["selector"], (*chainParameters)["OptimizerChainRefinerSimplex"]->values["selector"].value);

        nodes["selector"]->addNext(nodes["perturbator"], (*chainParameters)["OptimizerChainSelectorSimplex"]->values["perturbator"].value);
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
        optimizerContext->differentialEvolutionContext.parameters=*(*chainParameters)["PerturbatorDEOperatorParams"];
        (*perturbatorChain)["GA"]=new OperatorMarkovNode(&optimizerContext->geneticAlgorithmContext,std::string("GA").c_str());
        optimizerContext->geneticAlgorithmContext.parameters=*(*chainParameters)["PerturbatorGAOperatorParams"];
        (*perturbatorChain)["initializer"]->addNext((*perturbatorChain)["DE"], (*chainParameters)["PerturbatorInitializerSimplex"]->values["DE"].value);
        (*perturbatorChain)["initializer"]->addNext((*perturbatorChain)["GA"], (*chainParameters)["PerturbatorInitializerSimplex"]->values["GA"].value);
        (*perturbatorChain)["DE"]->addNext((*perturbatorChain)["DE"], (*chainParameters)["PerturbatorDESimplex"]->values["DE"].value);
        (*perturbatorChain)["DE"]->addNext((*perturbatorChain)["GA"], (*chainParameters)["PerturbatorDESimplex"]->values["GA"].value);
        (*perturbatorChain)["GA"]->addNext((*perturbatorChain)["DE"], (*chainParameters)["PerturbatorGASimplex"]->values["DE"].value);
        (*perturbatorChain)["GA"]->addNext((*perturbatorChain)["GA"], (*chainParameters)["PerturbatorGASimplex"]->values["GA"].value);
        return new OperatorMarkovChain(*perturbatorChain,metrics);
    }

    OperatorMarkovChain* buildRefinerChain(OptimizerContext* optimizerContext) {
        auto* refinerChain=new std::unordered_map<std::string,MarkovNode*>();
        (*refinerChain)["initializer"]=new OperatorMarkovNode(new Initializer(), std::string("initializer").c_str());
        (*refinerChain)["GD"]=new OperatorMarkovNode(&optimizerContext->gdLocalSearch, std::string("GD").c_str());
        optimizerContext->gdLocalSearch.parameters=*(*chainParameters)["RefinerGDOperatorParams"];
        (*refinerChain)["LBFGS"]=new OperatorMarkovNode(&optimizerContext->lbfgsLocalSearch, std::string("LBFGS").c_str());
        optimizerContext->lbfgsLocalSearch.parameters=*(*chainParameters)["RefinerLBFGSOperatorParams"];
        (*refinerChain)["initializer"]->addNext((*refinerChain)["LBFGS"], (*chainParameters)["RefinerInitializerSimplex"]->values["LBFGS"].value);
        (*refinerChain)["initializer"]->addNext((*refinerChain)["GD"], (*chainParameters)["RefinerInitializerSimplex"]->values["GD"].value);
        (*refinerChain)["LBFGS"]->addNext((*refinerChain)["LBFGS"], (*chainParameters)["RefinerLBFGSSimplex"]->values["LBFGS"].value);
        (*refinerChain)["LBFGS"]->addNext((*refinerChain)["GD"], (*chainParameters)["RefinerLBFGSSimplex"]->values["GD"].value);
        (*refinerChain)["GD"]->addNext((*refinerChain)["LBFGS"], (*chainParameters)["RefinerGDSimplex"]->values["LBFGS"].value);
        (*refinerChain)["GD"]->addNext((*refinerChain)["GD"], (*chainParameters)["RefinerGDSimplex"]->values["GD"].value);
        return new OperatorMarkovChain(*refinerChain, metrics);
    }

    OperatorMarkovChain* buildSelectorChain(OptimizerContext* optimizerContext) {
        auto* selectorChain=new std::unordered_map<std::string,MarkovNode*>();
        (*selectorChain)["initializer"]=new OperatorMarkovNode(new Initializer(), std::string("initializer").c_str());
        (*selectorChain)["best"]=new OperatorMarkovNode(&optimizerContext->bestSelector, std::string("best").c_str());
        bestSelector=(OperatorMarkovNode*)(*selectorChain)["best"];
        (*selectorChain)["initializer"]->addNext((*selectorChain)["best"], (*chainParameters)["SelectorInitializerSimplex"]->values["best"].value);
        (*selectorChain)["best"]->addNext((*selectorChain)["best"], (*chainParameters)["SelectorBestSimplex"]->values["best"].value);
        return new OperatorMarkovChain(*selectorChain, metrics);
    }

    void deleteCurrentNodes(){
        std::for_each(nodes.begin(),nodes.end(),[](std::pair<const std::string, MarkovNode*> node){delete node.second;});
    }

public:
    OptimizingMarkovChain(OptimizerContext* optimizerContext, Metrics* metrics): cudaMemoryModel(&optimizerContext->cudaMemoryModel) {
        this->metrics=metrics;
    }

    ~OptimizingMarkovChain() {
        deleteCurrentNodes();
    }

    void hopToNext() override {
        currentNode=currentNode->getNext(generator);
//        std::cout<<"hopped to "<<currentNode->name<<std::endl;
        metrics->modelPerformanceMetrics.markovIterations++;
    }

    void operate(int maxEvaluations) {
        ((OptimizingMarkovNode*)currentNode)->operate(cudaMemoryModel,maxEvaluations-metrics->modelPerformanceMetrics.fEvaluations);
        metrics->modelPerformanceMetrics.fEvaluations+=((OptimizingMarkovNode*)currentNode)->fEvals();
        ((OptimizingMarkovNode*)currentNode)->hopToNext();
    }


    void selectBest(){
        bestSelector->operate(cudaMemoryModel,1);
    }

    void printParameters() {
        std::for_each(chainParameters->begin(),chainParameters->end(),[](std::pair<const std::string,OperatorParameters*>& operatorParameter){
            std::cout<<operatorParameter.first<<":"<<std::endl;
            operatorParameter.second->printParameters();
        });
    }

    void setParameters(std::unordered_map<std::string, OperatorParameters *>*parameters, OptimizerContext* optimizerContext){
        this->chainParameters=parameters;
        resetChainBasedOnParameters(optimizerContext);
    }
};

#endif //PARALLELLBFGS_OPTIMIZINGMARKOVCHAIN_CUH
