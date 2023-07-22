//
// Created by spaceman on 2023. 07. 05..
//

#ifndef PARALLELLBFGS_SCHWEFELMODEL_CUH
#define PARALLELLBFGS_SCHWEFELMODEL_CUH

#include "../../optimizer/operators/perturb/Perturbator.h"
#include "../../common/Metrics.cuh"
#include "../../common/model/CudaMemoryModel.cuh"
#include "Schwefel.cuh"
#include <random>

#ifdef PROBLEM_SCHWEFEL
class SchwefelModel: public Model {
    std::mt19937 generator=std::mt19937(std::random_device()());
    Residual SchwefelResidual[1]{};
public:

    SchwefelModel():Model(){};
    explicit SchwefelModel(Perturbator& perturbator) : Model(perturbator.populationSize,X_DIM) {
        residuals.residualCount=1;
        SchwefelResidual[0].constantsCount= X_DIM;
        SchwefelResidual[0].constantsDim=4;
        SchwefelResidual[0].parametersDim=1;
        residuals.residual= reinterpret_cast<Residual *>(&SchwefelResidual[0]);
    }

    void loadModel(void* dev_x, void* dev_xDE, void* dev_constantData, Metrics &metrics,CUDAMemoryModel* model ) override {
        const int constantDataSize=residuals.residualDataSize();
        double x[modelPopulationSize]={};
        double lowerbounds[modelSize]={};
        double upperbounds[modelSize]={};
        double data[constantDataSize]={};

        for(int i=0;i<modelPopulationSize;i++) {
            x[i]=std::uniform_real_distribution<double>(-500, 500)(generator);
        }
        for(int i=0;i<modelSize;i++) {
            lowerbounds[i]=-500;
            upperbounds[i]=500;
        }
        for(int i=0; i < SchwefelResidual[0].constantsCount; i++) {
            data[4*i]=i;
            data[4*i+1]=-1.0;
            data[4*i+2]=1.0;
            data[4*i+3]=418.9829;
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
        Schwefel f1 = Schwefel(); \

#define INJECT_RESIDUAL_FUNCTIONS() \
        ((Model*)localContext.modelP)->residuals.residual[0].residualProblem=&f1; \
        ((Model*)localContext.modelP)->residuals.residual[0].constants = globalData; \

#define CAST_RESIDUAL_FUNCTIONS() \
        Schwefel *f1 = ((Schwefel *) model->residuals.residual[0].residualProblem); \

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

#endif //PARALLELLBFGS_SCHWEFELMODEL_CUH
