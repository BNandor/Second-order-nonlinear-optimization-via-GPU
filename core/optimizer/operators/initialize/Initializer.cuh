//
// Created by spaceman on 2022. 11. 09..
//

#ifndef PARALLELLBFGS_INITIALIZER_CUH
#define PARALLELLBFGS_INITIALIZER_CUH
#include "../Operator.h"
#include <unordered_map>


class Initializer: public Operator{
public:
     int fEvaluationCount(){
         return 0;
     };

     void operate(CUDAMemoryModel* cudaContext){
     }

     void limitEvaluationsTo(int remainingEvaluations) override{

     }
};
#endif //PARALLELLBFGS_INITIALIZER_CUH
