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
        nodes=std::unordered_map<std::string,MarkovNode*>();
        nodes["initializer"]=new OptimizingMarkovNode((Operator**)optimizerContext->getCurrentInitializerAddress(),std::string("initializer").c_str());
        nodes["perturbator"]=new OptimizingMarkovNode((Operator**)optimizerContext->getCurrentPerturbatorAddress(),std::string("perturbator").c_str());
        nodes["refiner"]=new OptimizingMarkovNode((Operator**)optimizerContext->getCurrentLocalSearchAdress(),std::string("refiner").c_str());
        nodes["selector"]=new OptimizingMarkovNode((Operator**)optimizerContext->getCurrentSelectorAddress(),std::string("refiner").c_str());
        buildChain();
        currentNode=nodes["initializer"];
        this->metrics=metrics;
    }

    ~OptimizingMarkovChain() {
        std::for_each(nodes.begin(),nodes.end(),[](auto node){delete std::get<1>(node);});
    }

    void hopToNext() {
        currentNode=currentNode->getNext(generator);
        metrics->modelPerformanceMetrics.markovIterations++;
    }

    void operate() {
//        std::cout<<"operating: "<<currentNode->name<<std::endl;
        currentNode->operate(cudaMemoryModel);
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
};

#endif //PARALLELLBFGS_OPTIMIZINGMARKOVCHAIN_CUH
