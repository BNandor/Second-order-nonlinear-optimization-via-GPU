//
// Created by spaceman on 2022. 11. 14..
//

#ifndef PARALLELLBFGS_OPTIMIZINGMARKOVNODE_CUH
#define PARALLELLBFGS_OPTIMIZINGMARKOVNODE_CUH
#include "../MarkovNode.cuh"
#include "../operator/OperatorMarkovChain.cuh"
#include "../operator/OperatorMarkovNode.cuh"

class OptimizingMarkovNode: public MarkovNode {
    OperatorMarkovChain* operatorChain;

public:

    OptimizingMarkovNode(OperatorMarkovChain* anOperatorMarkovChain, const char* name):MarkovNode(name) {
        operatorChain=anOperatorMarkovChain;
    }

    ~OptimizingMarkovNode(){
        std::for_each(operatorChain->nodes.begin(),operatorChain->nodes.end(),[](auto node){delete std::get<1>(node);});
        delete operatorChain;
    }

    void operate(CUDAMemoryModel* cudaMemoryModel,int remainingEvaluations)  {
        ((OperatorMarkovNode*)operatorChain->currentNode)->operate(cudaMemoryModel, remainingEvaluations);
    }
    void hopToNext(){
        operatorChain->hopToNext();
    }
    int fEvals() {
        return ((OperatorMarkovNode*)(operatorChain->currentNode))->nodeOperator->fEvaluationCount();
    }
};
#endif //PARALLELLBFGS_OPTIMIZINGMARKOVNODE_CUH
