//
// Created by spaceman on 2022. 11. 14..
//

#ifndef PARALLELLBFGS_OPTIMIZINGMARKOVNODE_CUH
#define PARALLELLBFGS_OPTIMIZINGMARKOVNODE_CUH
#include "MarkovNode.cuh"
#include "OperatorMarkovChain.cuh"
#include "OperatorMarkovNode.cuh"

class OptimizingMarkovNode: public MarkovNode {
    OperatorMarkovChain* operatorChain;

public:

    OptimizingMarkovNode(OperatorMarkovChain* anOperatorMarkovChain, const char* name):MarkovNode(name) {
        operatorChain=anOperatorMarkovChain;
    }

    void operate(CUDAMemoryModel* cudaMemoryModel)  {
        ((OperatorMarkovNode*)operatorChain->currentNode)->operate(cudaMemoryModel);
        operatorChain->hopToNext();
    }

    int fEvals() {
        return ((OperatorMarkovNode*)(operatorChain->currentNode))->nodeOperator->fEvaluationCount();
    }
};
#endif //PARALLELLBFGS_OPTIMIZINGMARKOVNODE_CUH
