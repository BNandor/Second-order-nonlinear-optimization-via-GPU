//
// Created by spaceman on 2022. 11. 14..
//

#ifndef PARALLELLBFGS_OPTIMIZINGMARKOVNODE_CUH
#define PARALLELLBFGS_OPTIMIZINGMARKOVNODE_CUH
#include "MarkovNode.cuh"

class OptimizingMarkovNode: public MarkovNode {
    Operator** nodeOperator;
public:

    OptimizingMarkovNode(Operator** anOperator,const char* name):MarkovNode(name),nodeOperator(anOperator){
    }

    void operate(CUDAMemoryModel* cudaMemoryModel) override {
        (*nodeOperator)->operate(cudaMemoryModel);
    }

    int fEvals() {
        return (*nodeOperator)->fEvaluationCount();
    }
};
#endif //PARALLELLBFGS_OPTIMIZINGMARKOVNODE_CUH
