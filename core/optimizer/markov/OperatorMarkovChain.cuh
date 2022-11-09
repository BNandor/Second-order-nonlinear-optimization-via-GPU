//
// Created by spaceman on 2022. 11. 09..
//

#ifndef PARALLELLBFGS_OPERATORMARKOVCHAIN_CUH
#define PARALLELLBFGS_OPERATORMARKOVCHAIN_CUH
#include <random>
#include "../../common/OptimizerContext.cuh"
#include "MarkovNode.cuh"
#include <unordered_map>
#include <algorithm>

class OperatorMarkovChain {
    std::mt19937 generator=std::mt19937(std::random_device()());
    std::unordered_map<std::string,MarkovNode*> nodes;
    MarkovNode* currentNode;
    CUDAMemoryModel* cudaMemoryModel;
public:

    OperatorMarkovChain(OptimizerContext* optimizerContext):cudaMemoryModel(&optimizerContext->cudaMemoryModel) {
        nodes=std::unordered_map<std::string,MarkovNode*>();
        nodes["initializer"]=new MarkovNode(optimizerContext->getCurrentInitializer(),std::string("initializer").c_str());
        nodes["perturbator"]=new MarkovNode(optimizerContext->getCurrentPerturbator(),std::string("perturbator").c_str());
        nodes["refiner"]=new MarkovNode(optimizerContext->getCurrentLocalSearch(),std::string("refiner").c_str());
        nodes["selector"]=new MarkovNode(optimizerContext->getCurrentSelector(),std::string("refiner").c_str());
        buildChain();
        currentNode=nodes["initializer"];
    }
    ~OperatorMarkovChain(){
        std::for_each(nodes.begin(),nodes.end(),[](auto node){delete std::get<1>(node);});
    }

    void hopToNext() {
        currentNode=currentNode->getNext(generator);
    }

    int operate() {
//        std::cout<<"operating: "<<currentNode->name<<std::endl;
        currentNode->operate(cudaMemoryModel);
        return currentNode->fEvals();
    }

    void buildChain() {
        nodes["initializer"]->addNext(nodes["perturbator"],1.0);
        nodes["initializer"]->addNext(nodes["refiner"],0.0);

        nodes["perturbator"]->addNext(nodes["selector"],0.5);
        nodes["perturbator"]->addNext(nodes["refiner"],0.5);

        nodes["refiner"]->addNext(nodes["refiner"],0.0);
        nodes["refiner"]->addNext(nodes["selector"],1.0);

        nodes["selector"]->addNext(nodes["perturbator"],1.0);
    }
};

#endif //PARALLELLBFGS_OPERATORMARKOVCHAIN_CUH
