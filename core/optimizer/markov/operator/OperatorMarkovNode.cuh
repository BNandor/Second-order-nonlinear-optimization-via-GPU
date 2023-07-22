//
// Created by spaceman on 2022. 11. 14..
//

#ifndef PARALLELLBFGS_OPERATORMARKOVNODE_CUH
#define PARALLELLBFGS_OPERATORMARKOVNODE_CUH
#include "../MarkovNode.cuh"

class OperatorMarkovNode: public MarkovNode {
public:
    Operator* nodeOperator;
    OperatorMarkovNode(Operator* anOperator,const char* name):MarkovNode(name),nodeOperator(anOperator){
    }

    void operate(CUDAMemoryModel* cudaMemoryModel,int remainingEvaluations) const {
//        std::cout<<"limiting evaluations on "<<nodeOperator<<std::endl;
        nodeOperator->limitEvaluationsTo(remainingEvaluations);
//        std::cout<<"operation "<<std::endl;
        nodeOperator->operate(cudaMemoryModel);
    }

};
#endif //PARALLELLBFGS_OPERATORMARKOVNODE_CUH
