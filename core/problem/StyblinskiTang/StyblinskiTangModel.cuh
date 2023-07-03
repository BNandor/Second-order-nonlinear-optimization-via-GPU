//
// Created by spaceman on 2022. 12. 07..
//

#ifndef PARALLELLBFGS_STYBLINSKITANGMODEL_CUH
#define PARALLELLBFGS_STYBLINSKITANGMODEL_CUH

#include "../../optimizer/operators/perturb/Perturbator.h"
#include "../../common/Metrics.cuh"
#include "StyblinskiTang.cuh"
#include <random>

#ifdef PROBLEM_STYBLINSKITANG
class StyblinskiTangModel: public Model {
    std::mt19937 generator=std::mt19937(std::random_device()());
    Residual StyblinskiTangResidual[1]{};
public:

    StyblinskiTangModel():Model(){};
    explicit StyblinskiTangModel(Perturbator& perturbator) : Model(perturbator.populationSize,X_DIM) {
        residuals.residualCount=1;
        StyblinskiTangResidual[0].constantsCount= X_DIM;
        StyblinskiTangResidual[0].constantsDim=4;
        StyblinskiTangResidual[0].parametersDim=1;
        residuals.residual= reinterpret_cast<Residual *>(&StyblinskiTangResidual[0]);
    }

    void loadModel(void* dev_x, void* dev_xDE, void* dev_constantData, Metrics &metrics,CUDAMemoryModel* model ) override {
        const int constantDataSize=residuals.residualDataSize();
        double x[modelPopulationSize]={};
        double data[constantDataSize]={};

        for(int i=0;i<modelPopulationSize;i++) {
            x[i]=std::uniform_real_distribution<double>(-100, 100)(generator);
        }
        for(int i=0; i < StyblinskiTangResidual[0].constantsCount; i++) {
            data[4*i]=i;
            data[4*i+1]=16.0;
            data[4*i+2]=5.0;
            data[4*i+3]=0.5;
        }
        metrics.getCudaEventMetrics().recordStartCopy();
        cudaMemcpy(dev_x, &x, modelPopulationSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_xDE, &x, modelPopulationSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_constantData, &data, constantDataSize * sizeof(double), cudaMemcpyHostToDevice);
        metrics.getCudaEventMetrics().recordStopCopy();
    }

};
#define DEFINE_RESIDUAL_FUNCTIONS() \
        StyblinskiTang f1 = StyblinskiTang(); \

#define INJECT_RESIDUAL_FUNCTIONS() \
        ((Model*)localContext.modelP)->residuals.residual[0].residualProblem=&f1; \
        ((Model*)localContext.modelP)->residuals.residual[0].constants = globalData; \

#define CAST_RESIDUAL_FUNCTIONS() \
        StyblinskiTang *f1 = ((StyblinskiTang *) model->residuals.residual[0].residualProblem); \

#define COMPUTE_RESIDUALS() \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[0].constantsCount; spanningTID += blockDim.x) { \
            f1->setConstants(&(model->residuals.residual[0].constants[model->residuals.residual[0].constantsDim * spanningTID]), model->residuals.residual[0].constantsDim); \
            localContext->threadF += f1->eval(x, X_DIM)->value; \
            f1->evalJacobian(); \
            for (unsigned j = 0; j < model->residuals.residual[0].parametersDim; j++) { \
                atomicAdd(&dx[f1->ThisJacobianIndices[j]], f1->operatorTree[f1->constantSize + j].derivative); } \
        } \

#define COMPUTE_LINESEARCH() \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[0].constantsCount; spanningTID += blockDim.x) {  \
            f1->setConstants(&(model->residuals.residual[0].constants[model->residuals.residual[0].constantsDim * spanningTID]), model->residuals.residual[0].constantsDim); \
            fNext += f1->eval(sharedContext->xNext, X_DIM)->value; \
        } \

#endif

#endif //PARALLELLBFGS_STYBLINSKITANGMODEL_CUH
