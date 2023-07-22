//
// Created by spaceman on 2023. 07. 05..
//

#ifndef PARALLELLBFGS_SUMSQUARESMODEL_CUH
#define PARALLELLBFGS_SUMSQUARESMODEL_CUH

#include "../../optimizer/operators/perturb/Perturbator.h"
#include "../../common/Metrics.cuh"
#include "../../common/model/CudaMemoryModel.cuh"
#include "SumSquares.cuh"
#include <random>

#ifdef PROBLEM_SUMSQUARES
class SumSquaresModel: public Model {
    std::mt19937 generator=std::mt19937(std::random_device()());
    Residual SumSquaresResidual[1]{};
public:

    SumSquaresModel():Model(){};
    explicit SumSquaresModel(Perturbator& perturbator) : Model(perturbator.populationSize,X_DIM) {
        residuals.residualCount=1;
        SumSquaresResidual[0].constantsCount= X_DIM;
        SumSquaresResidual[0].constantsDim=2;
        SumSquaresResidual[0].parametersDim=1;
        residuals.residual= reinterpret_cast<Residual *>(&SumSquaresResidual[0]);
    }

    void loadModel(void* dev_x, void* dev_xDE, void* dev_constantData, Metrics &metrics,CUDAMemoryModel* model ) override {
        const int constantDataSize=residuals.residualDataSize();
        double x[modelPopulationSize]={};
        double lowerbounds[modelSize]={};
        double upperbounds[modelSize]={};
        double data[constantDataSize]={};

        for(int i=0;i<modelPopulationSize;i++) {
            x[i]=std::uniform_real_distribution<double>(-10, 10)(generator);
        }
        for(int i=0;i<modelSize;i++) {
            lowerbounds[i]=-10;
            upperbounds[i]=10;
        }
        for(int i=0; i < SumSquaresResidual[0].constantsCount; i++) {
            data[2*i]=i;
            data[2*i+1]=i+1;
        }

        metrics.getCudaEventMetrics().recordStartCopy();
        model->isBounded=true;
        cudaMemcpy(model->dev_lower_bounds, &lowerbounds, modelSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(model->dev_upper_bounds, &upperbounds, modelSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_x, &x, modelPopulationSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_xDE, &x, modelPopulationSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_constantData, &data, constantDataSize * sizeof(double), cudaMemcpyHostToDevice);
        metrics.getCudaEventMetrics().recordStopCopy();
    }
};

#define DEFINE_RESIDUAL_FUNCTIONS() \
        SumSquares f1 = SumSquares(); \

#define INJECT_RESIDUAL_FUNCTIONS() \
        ((Model*)localContext.modelP)->residuals.residual[0].residualProblem=&f1; \
        ((Model*)localContext.modelP)->residuals.residual[0].constants = globalData; \

#define CAST_RESIDUAL_FUNCTIONS() \
        SumSquares *f1 = ((SumSquares *) model->residuals.residual[0].residualProblem); \

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

#endif //PARALLELLBFGS_SUMSQUARESMODEL_CUH
