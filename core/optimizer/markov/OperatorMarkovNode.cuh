//
// Created by spaceman on 2022. 11. 14..
//

#ifndef PARALLELLBFGS_OPERATORMARKOVNODE_CUH
#define PARALLELLBFGS_OPERATORMARKOVNODE_CUH
#include "MarkovNode.cuh"

class OperatorMarkovNode: public MarkovNode {
    Operator* nodeOperator;
public:
    OperatorMarkovNode(Operator* anOperator,const char* name):MarkovNode(name),nodeOperator(anOperator){

    }
    void operate(CUDAMemoryModel* cudaMemoryModel) override {
        nodeOperator->operate(cudaMemoryModel);
    }

    int fEvals() {
        return nodeOperator->fEvaluationCount();
    }
};
#endif //PARALLELLBFGS_OPERATORMARKOVNODE_CUH
