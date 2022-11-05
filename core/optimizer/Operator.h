//
// Created by spaceman on 2022. 11. 05..
//

#ifndef PARALLELLBFGS_OPERATOR_H
#define PARALLELLBFGS_OPERATOR_H

class Operator {
public:
    OperatorParameters parameters;
    virtual int fEvaluationCount()=0;
};

#endif //PARALLELLBFGS_OPERATOR_H
