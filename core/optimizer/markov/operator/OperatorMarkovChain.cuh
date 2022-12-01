//
// Created by spaceman on 2022. 11. 14..
//

#ifndef PARALLELLBFGS_OPERATORMARKOVCHAIN_CUH
#define PARALLELLBFGS_OPERATORMARKOVCHAIN_CUH

#include "../../../common/OptimizerContext.cuh"
#include "../MarkovChain.cuh"
#include <algorithm>
#include <utility>

class OperatorMarkovChain: public MarkovChain {
    Metrics* metrics;

public:

    OperatorMarkovChain(std::unordered_map<std::string,MarkovNode*> operatorNodes, Metrics* metrics) {
        nodes=std::move(operatorNodes);
        if(nodes.count("initializer") == 0) {
            std::cerr<<"Invalid markov chain provided, please define the initializer node"<<std::endl;
            exit(1);
        }
        currentNode=nodes["initializer"];
        this->metrics=metrics;
    }

    ~OperatorMarkovChain() {
        std::for_each(nodes.begin(),nodes.end(),[](auto node){delete std::get<1>(node);});
        delete &nodes;
    }

    void hopToNext() override {
        currentNode=currentNode->getNext(generator);
//        std::cout<<"hopped to "<<currentNode->name<<std::endl;
    }
};

#endif //PARALLELLBFGS_OPERATORMARKOVCHAIN_CUH
