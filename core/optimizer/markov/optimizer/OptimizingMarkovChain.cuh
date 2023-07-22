//
// Created by spaceman on 2022. 11. 09..
//

#ifndef PARALLELLBFGS_OPTIMIZINGMARKOVCHAIN_CUH
#define PARALLELLBFGS_OPTIMIZINGMARKOVCHAIN_CUH
#include <random>
#include "../../../common/OptimizerContext.cuh"
#include "../../../common/CommonStringUtils.h"
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

#ifdef BASE_PERTURB_EXTRA_OPERATORS
        std::string operators=BASE_PERTURB_EXTRA_OPERATORS;
        std::set<std::string> operatorSet=stringutil::splitString(operators,',');
        std::for_each(operatorSet.begin(),operatorSet.end(),[&perturbatorChain,optimizerContext,this,operatorSet](std::string op){
            if(op == "GWO"){
                (*perturbatorChain)["GWO"]=new OperatorMarkovNode(&optimizerContext->greyWolfOptimizerContext,std::string("GWO").c_str());
                optimizerContext->greyWolfOptimizerContext.parameters=*(*chainParameters)["PerturbatorGWOOperatorParams"];
            }
            if(op == "GWO2"){
                (*perturbatorChain)["GWO2"]=new OperatorMarkovNode(&optimizerContext->greyWolfOptimizerContext2,std::string("GWO2").c_str());
                optimizerContext->greyWolfOptimizerContext2.parameters=*(*chainParameters)["PerturbatorGWO2OperatorParams"];
            }
            if(op == "DE2"){
                (*perturbatorChain)["DE2"]=new OperatorMarkovNode(&optimizerContext->differentialEvolutionContext2,std::string("DE2").c_str());
                optimizerContext->differentialEvolutionContext2.parameters=*(*chainParameters)["PerturbatorDE2OperatorParams"];
            }
            if(op == "GA2"){
                (*perturbatorChain)["GA2"]=new OperatorMarkovNode(&optimizerContext->geneticAlgorithmContext2,std::string("GA2").c_str());
                optimizerContext->geneticAlgorithmContext2.parameters=*(*chainParameters)["PerturbatorGA2OperatorParams"];
            }
            (*perturbatorChain)["initializer"]->addNext((*perturbatorChain)[op], (*chainParameters)["PerturbatorInitializerSimplex"]->values[op].value);
            (*perturbatorChain)["DE"]->addNext((*perturbatorChain)[op], (*chainParameters)["PerturbatorDESimplex"]->values[op].value);
            (*perturbatorChain)["GA"]->addNext((*perturbatorChain)[op], (*chainParameters)["PerturbatorGASimplex"]->values[op].value);
            if((*perturbatorChain).count(op)<=0){
                std::cerr<<"Invalid operator configuration, please define "<<op<<std::endl;
                exit(5);
            }
            (*perturbatorChain)[op]->addNext((*perturbatorChain)["DE"], (*chainParameters)["Perturbator"+op+"Simplex"]->values["DE"].value);
            (*perturbatorChain)[op]->addNext((*perturbatorChain)["GA"], (*chainParameters)["Perturbator"+op+"Simplex"]->values["GA"].value);
            });
        std::for_each(operatorSet.begin(),operatorSet.end(),[&perturbatorChain,optimizerContext,this,operatorSet](std::string op){
            std::for_each(operatorSet.begin(),operatorSet.end(),[&perturbatorChain,this,&op](std::string op2){
                if((*perturbatorChain).count(op2)<=0){
                    std::cerr<<"Invalid operator configuration, please define "<<op2<<std::endl;
                    exit(5);
                }
                    (*perturbatorChain)[op]->addNext((*perturbatorChain)[op2], (*chainParameters)["Perturbator"+op+"Simplex"]->values[op2].value);
            });
        });
#endif
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
        std::for_each(nodes.begin(),nodes.end(),[](auto node){delete std::get<1>(node);});
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
//        std::cout<<"Operating"<<((OptimizingMarkovNode*)currentNode)->name<<std::endl;
//        std::cout<<"remaining"<<maxEvaluations-metrics->modelPerformanceMetrics.fEvaluations<<std::endl;
        ((OptimizingMarkovNode*)currentNode)->operate(cudaMemoryModel,maxEvaluations-metrics->modelPerformanceMetrics.fEvaluations);
//        std::cout<<"Operated"<<((OptimizingMarkovNode*)currentNode)->name<<std::endl;
        metrics->modelPerformanceMetrics.fEvaluations+=((OptimizingMarkovNode*)currentNode)->fEvals();
//        std::cout<<"feval"<<((OptimizingMarkovNode*)currentNode)->name<<std::endl;
        ((OptimizingMarkovNode*)currentNode)->hopToNext();
//        std::cout<<"hopped to next"<<((OptimizingMarkovNode*)currentNode)->name<<std::endl;
    }


    void selectBest(){
        bestSelector->operate(cudaMemoryModel,1);
    }

    void printParameters() {
        std::for_each(chainParameters->begin(),chainParameters->end(),[](auto& operatorParameter){
            std::cout<<std::get<0>(operatorParameter)<<":"<<std::endl;
            std::get<1>(operatorParameter)->printParameters();
        });
    }

    void setParameters(std::unordered_map<std::string, OperatorParameters *>*parameters, OptimizerContext* optimizerContext){
        this->chainParameters=parameters;
        resetChainBasedOnParameters(optimizerContext);
    }
};

#endif //PARALLELLBFGS_OPTIMIZINGMARKOVCHAIN_CUH
