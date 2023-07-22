//
// Created by spaceman on 2023. 07. 04..
//

#ifndef PARALLELLBFGS_DIXONPRICEMODEL_CUH
#define PARALLELLBFGS_DIXONPRICEMODEL_CUH

#include "../../optimizer/operators/perturb/Perturbator.h"
#include "../../common/Metrics.cuh"
#include "../../common/model/CudaMemoryModel.cuh"
#include "DixonPrice1.cuh"
#include "DixonPrice2.cuh"
#include <random>

#ifdef PROBLEM_DIXONPRICE

class DixonPriceModel: public Model {
    std::mt19937 generator=std::mt19937(std::random_device()());
    Residual DixonPriceResidual[2]{};
public:

    DixonPriceModel():Model(){};
    explicit DixonPriceModel(Perturbator& perturbator) : Model(perturbator.populationSize,X_DIM) {
        residuals.residualCount=2;
        DixonPriceResidual[0].constantsCount= 1;
        DixonPriceResidual[0].constantsDim=2;
        DixonPriceResidual[0].parametersDim=1;
        DixonPriceResidual[1].constantsCount= X_DIM-1;
        DixonPriceResidual[1].constantsDim=3;
        DixonPriceResidual[1].parametersDim=2;
        residuals.residual= reinterpret_cast<Residual *>(&DixonPriceResidual[0]);
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
        for(int i=0; i < DixonPriceResidual[0].constantsCount; i++) {
            data[2*i]=i;
            data[2*i+1]=1.0;
        }
        for(int i=0; i < DixonPriceResidual[1].constantsCount; i++) {
            data[DixonPriceResidual[0].getConstantsDim()+ 3*i]=i+2;
            data[DixonPriceResidual[0].getConstantsDim()+ 3*i+1]=i;
            data[DixonPriceResidual[0].getConstantsDim()+ 3*i+2]=2.0;
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
        DixonPrice1 f1 = DixonPrice1(); \
        DixonPrice2 f2 = DixonPrice2(); \

#define INJECT_RESIDUAL_FUNCTIONS() \
        ((Model*)localContext.modelP)->residuals.residual[0].residualProblem=&f1; \
        ((Model*)localContext.modelP)->residuals.residual[0].constants = globalData; \
        ((Model*)localContext.modelP)->residuals.residual[1].residualProblem= &f2; \
        ((Model*)localContext.modelP)->residuals.residual[1].constants = ((Model*)localContext.modelP)->residuals.residual[0].constants  + ((Model*)localContext.modelP)->residuals.residual[0].getConstantsDim(); \

#define CAST_RESIDUAL_FUNCTIONS() \
        DixonPrice1 *f1 = ((DixonPrice1 *) model->residuals.residual[0].residualProblem); \
        DixonPrice2 *f2 = ((DixonPrice2 *) model->residuals.residual[1].residualProblem); \

#define COMPUTE_RESIDUALS() \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[0].constantsCount; spanningTID += blockDim.x) { \
            f1->setConstants(&(model->residuals.residual[0].constants[model->residuals.residual[0].constantsDim * spanningTID]), model->residuals.residual[0].constantsDim); \
            localContext->threadF += f1->eval(x, X_DIM)->value; \
            f1->evalJacobian(); \
            for (unsigned j = 0; j < model->residuals.residual[0].parametersDim; j++) { \
                atomicAdd(&dx[f1->ThisJacobianIndices[j]], f1->operatorTree[f1->constantSize + j].derivative); } \
        } \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[1].constantsCount; spanningTID += blockDim.x) { \
            f2->setConstants(&(model->residuals.residual[1].constants [model->residuals.residual[1].constantsDim * spanningTID]), \
                             model->residuals.residual[1].constantsDim); \
            localContext->threadF += f2->eval(x,X_DIM)->value; \
            f2->evalJacobian(); \
            for (unsigned j = 0; j < model->residuals.residual[1].parametersDim; j++) { \
                atomicAdd(&dx[f2->ThisJacobianIndices[j]], \
                          f2->operatorTree[f2->constantSize + j].derivative); \
            } \
        } \

#define COMPUTE_LINESEARCH() \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[0].constantsCount; spanningTID += blockDim.x) {  \
            f1->setConstants(&(model->residuals.residual[0].constants[model->residuals.residual[0].constantsDim * spanningTID]), model->residuals.residual[0].constantsDim); \
            fNext += f1->eval(sharedContext->xNext, X_DIM)->value; \
        } \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[1].constantsCount; spanningTID += blockDim.x) { \
            f2->setConstants(&(model->residuals.residual[1].constants[model->residuals.residual[1].constantsDim * spanningTID]), model->residuals.residual[1].constantsDim); \
            fNext += f2->eval(sharedContext->xNext,X_DIM)->value; \
        } \

#endif

#endif //PARALLELLBFGS_DIXONPRICEMODEL_CUH