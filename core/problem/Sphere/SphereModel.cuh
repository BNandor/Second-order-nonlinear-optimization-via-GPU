//
// Created by spaceman on 2023. 07. 05..
//

#ifndef PARALLELLBFGS_SPHEREMODEL_CUH
#define PARALLELLBFGS_SPHEREMODEL_CUH

#include "../../optimizer/operators/perturb/Perturbator.h"
#include "../../common/Metrics.cuh"
#include "../../common/model/CudaMemoryModel.cuh"
#include "../SumSquares/SumSquares.cuh"
#include <random>

#ifdef PROBLEM_SPHERE
class SphereModel: public Model {
    std::mt19937 generator=std::mt19937(std::random_device()());
    Residual SphereResidual[1]{};
public:

    SphereModel():Model(){};
    explicit SphereModel(Perturbator& perturbator) : Model(perturbator.populationSize,X_DIM) {
        residuals.residualCount=1;
        SphereResidual[0].constantsCount= X_DIM;
        SphereResidual[0].constantsDim=2;
        SphereResidual[0].parametersDim=1;
        residuals.residual= reinterpret_cast<Residual *>(&SphereResidual[0]);
    }

    void loadModel(void* dev_x, void* dev_xDE, void* dev_constantData, Metrics &metrics,CUDAMemoryModel* model ) override {
        const int constantDataSize=residuals.residualDataSize();
        double x[modelPopulationSize]={};
        double lowerbounds[modelSize]={};
        double upperbounds[modelSize]={};
        double data[constantDataSize]={};

        for(int i=0;i<modelPopulationSize;i++) {
            x[i]=std::uniform_real_distribution<double>(-5.12,5.12)(generator);
        }
        for(int i=0;i<modelSize;i++) {
            lowerbounds[i]=-5.12;
            upperbounds[i]=5.12;
        }
        for(int i=0; i < SphereResidual[0].constantsCount; i++) {
            data[2*i]=i;
            data[2*i+1]=1.0;
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

#endif //PARALLELLBFGS_SPHEREMODEL_CUH
