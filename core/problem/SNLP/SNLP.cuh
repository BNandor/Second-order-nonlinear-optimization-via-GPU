//
// Created by spaceman on 2022. 02. 19..
//

#ifndef PARALLELLBFGS_SNLP_CUH
#define PARALLELLBFGS_SNLP_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"
#include <math.h>

class SNLP : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 15;
    static const unsigned ThisParameterSize = 4;
    static const unsigned ThisConstantSize = 3;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    SNLP() {
        operatorTreeSize = ThisOperatorTreeSize;
        parameterSize = ThisParameterSize;
        constantSize = ThisConstantSize;
        operatorTree = ThisOperatorTree;
        jacobianIndices = ThisJacobianIndices;
        initIndex();
    }

    __device__ __host__
    void setConstants(double *constants, unsigned constantsSize) {
        initConst(constants, constantsSize);
    }

    __device__ __host__
    DDouble *eval(double *x, unsigned xSize) {
        // ((x(0)-x(2))^2 + (x(1)-x(3))^2 - c(2)^2)^2
//        initOperatorTree(x, xSize);
        initOperatorTreePartially(x, trunc(operatorTree[0].value) * 2, 2, 0);
        initOperatorTreePartially(x, trunc(operatorTree[1].value) * 2, 2, 2);

        operatorTree[7] = operatorTree[3] - operatorTree[5];
        operatorTree[8] = operatorTree[7].square();
        operatorTree[9] = operatorTree[4] - operatorTree[6];
        operatorTree[10] = operatorTree[9].square();
        operatorTree[11] = operatorTree[8] + operatorTree[10];
        operatorTree[12] = operatorTree[2].square();
        operatorTree[13] = operatorTree[11] - operatorTree[12];
        operatorTree[14] = operatorTree[13].square();

        return &operatorTree[14];
    }
};

#ifdef PROBLEM_SNLP
#define COMPUTE_RESIDUALS() \
        SNLP *f1 = ((SNLP *) localContext->residualProblems[0]); \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[0].constantsCount; spanningTID += blockDim.x) { \
            f1->setConstants(&(localContext->residualConstants[0][model->residuals.residual[0].constantsDim * spanningTID]), model->residuals.residual[0].constantsDim); \
            localContext->threadF += f1->eval(sharedContext->xCurrent, X_DIM)->value; \
            f1->evalJacobian(); \
            for (unsigned j = 0; j < model->residuals.residual[0].parametersDim; j++) { \
                atomicAdd(&sharedContext->globalData->sharedDX[f1->ThisJacobianIndices[j]], f1->operatorTree[f1->constantSize + j].derivative); } \
        } \
        SNLPAnchor *f2 = ((SNLPAnchor *) localContext->residualProblems[1]); \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[1].constantsCount; spanningTID += blockDim.x) { \
            f2->setConstants(&(localContext->residualConstants[1][model->residuals.residual[1].constantsDim * spanningTID]), \
                             model->residuals.residual[1].constantsDim); \
            localContext->threadF += f2->eval(sharedContext->xCurrent,X_DIM)->value; \
            f2->evalJacobian(); \
            for (unsigned j = 0; j < model->residuals.residual[1].parametersDim; j++) { \
                atomicAdd(&sharedContext->globalData->sharedDX[f2->ThisJacobianIndices[j]], \
                          f2->operatorTree[f2->constantSize + j].derivative); \
            } \
        } \

#endif
#endif //PARALLELLBFGS_SNLP_CUH
