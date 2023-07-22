//
// Created by spaceman on 2022. 12. 07..
//

#ifndef PARALLELLBFGS_LEVYMODEL_CUH
#define PARALLELLBFGS_LEVYMODEL_CUH

#include "../../optimizer/operators/perturb/Perturbator.h"
#include "../../common/Metrics.cuh"
#include "../../common/model/CudaMemoryModel.cuh"
#include "Levy1.cuh"
#include "Levy2.cuh"
#include "Levy3.cuh"
#include <random>

#ifdef PROBLEM_LEVY
class LevyModel: public Model {
    std::mt19937 generator=std::mt19937(std::random_device()());
    Residual LevyResidual[3]{};
public:

    LevyModel():Model(){};
    explicit LevyModel(Perturbator& perturbator) : Model(perturbator.populationSize,X_DIM) {
        residuals.residualCount=3;
        LevyResidual[0].constantsCount=1;
        LevyResidual[0].constantsDim=4;
        LevyResidual[0].parametersDim=1;

        LevyResidual[1].constantsCount= X_DIM-1;
        LevyResidual[1].constantsDim=5;
        LevyResidual[1].parametersDim=1;

        LevyResidual[2].constantsCount= 1;
        LevyResidual[2].constantsDim=4;
        LevyResidual[2].parametersDim=1;
        residuals.residual= reinterpret_cast<Residual *>(&LevyResidual[0]);
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
        //c(0) xindex, c(1) 1, c(2) 4, c(3) pi
        for(int i=0; i < LevyResidual[0].constantsCount; i++) {
            data[4*i]=i;
            data[4*i+1]=1.0;
            data[4*i+2]=4.0;
            data[4*i+3]=M_PI;
        }

        // c(0) xindex, c(1) 1, c(2) 4, c(3) pi, c(4) 10,
        for(int i=0; i < LevyResidual[1].constantsCount; i++) {
            data[LevyResidual[0].getConstantsDim()+ 5*i]=i;
            data[LevyResidual[0].getConstantsDim()+ 5*i+1]=1.0;
            data[LevyResidual[0].getConstantsDim()+ 5*i+2]=4.0;
            data[LevyResidual[0].getConstantsDim()+ 5*i+3]=M_PI;
            data[LevyResidual[0].getConstantsDim()+ 5*i+4]=10.0;
        }
        // c(0) xindex, c(1) 1, c(2) 4, c(3) 2pi
        for(int i=0; i < LevyResidual[2].constantsCount; i++) {
            data[LevyResidual[0].getConstantsDim() +LevyResidual[1].getConstantsDim()+ 4*i]=X_DIM-1;
            data[LevyResidual[0].getConstantsDim() +LevyResidual[1].getConstantsDim()+ 4*i+1]=1.0;
            data[LevyResidual[0].getConstantsDim() +LevyResidual[1].getConstantsDim()+ 4*i+2]=4.0;
            data[LevyResidual[0].getConstantsDim() +LevyResidual[1].getConstantsDim()+ 4*i+3]=((double)2)*M_PI;
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
        Levy1 f1 = Levy1();         \
        Levy2 f2 = Levy2();         \
        Levy3 f3 = Levy3();         \


#define INJECT_RESIDUAL_FUNCTIONS() \
        ((Model*)localContext.modelP)->residuals.residual[0].residualProblem=&f1; \
        ((Model*)localContext.modelP)->residuals.residual[0].constants = globalData; \
        ((Model*)localContext.modelP)->residuals.residual[1].residualProblem= &f2; \
        ((Model*)localContext.modelP)->residuals.residual[1].constants = ((Model*)localContext.modelP)->residuals.residual[0].constants  + ((Model*)localContext.modelP)->residuals.residual[0].getConstantsDim(); \
        ((Model*)localContext.modelP)->residuals.residual[2].residualProblem= &f3; \
        ((Model*)localContext.modelP)->residuals.residual[2].constants = ((Model*)localContext.modelP)->residuals.residual[0].constants  + ((Model*)localContext.modelP)->residuals.residual[0].getConstantsDim() + ((Model*)localContext.modelP)->residuals.residual[1].getConstantsDim(); \

#define CAST_RESIDUAL_FUNCTIONS() \
        Levy1 *f1 = ((Levy1 *) model->residuals.residual[0].residualProblem); \
        Levy2 *f2 = ((Levy2 *) model->residuals.residual[1].residualProblem);         \
        Levy3 *f3 = ((Levy3 *) model->residuals.residual[2].residualProblem); \

#define COMPUTE_RESIDUALS() \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[0].constantsCount; spanningTID += blockDim.x) { \
            f1->setConstants(&(model->residuals.residual[0].constants[model->residuals.residual[0].constantsDim * spanningTID]), model->residuals.residual[0].constantsDim); \
            localContext->threadF += f1->eval(x, X_DIM)->value; \
            f1->evalJacobian(); \
            for (unsigned j = 0; j < model->residuals.residual[0].parametersDim; j++) { \
                atomicAdd(&dx[f1->ThisJacobianIndices[j]], f1->operatorTree[f1->constantSize + j].derivative); } \
        }                   \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[1].constantsCount; spanningTID += blockDim.x) { \
            f2->setConstants(&(model->residuals.residual[1].constants [model->residuals.residual[1].constantsDim * spanningTID]), \
                             model->residuals.residual[1].constantsDim); \
            localContext->threadF += f2->eval(x,X_DIM)->value; \
            f2->evalJacobian(); \
            for (unsigned j = 0; j < model->residuals.residual[1].parametersDim; j++) { \
                atomicAdd(&dx[f2->ThisJacobianIndices[j]], \
                          f2->operatorTree[f2->constantSize + j].derivative); \
            } \
        }                   \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[2].constantsCount; spanningTID += blockDim.x) { \
            f3->setConstants(&(model->residuals.residual[2].constants [model->residuals.residual[2].constantsDim * spanningTID]), \
                             model->residuals.residual[2].constantsDim); \
            localContext->threadF += f3->eval(x,X_DIM)->value; \
            f3->evalJacobian(); \
            for (unsigned j = 0; j < model->residuals.residual[2].parametersDim; j++) { \
                atomicAdd(&dx[f3->ThisJacobianIndices[j]], \
                          f3->operatorTree[f3->constantSize + j].derivative); \
            } \
        }                   \

#define COMPUTE_LINESEARCH() \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[0].constantsCount; spanningTID += blockDim.x) {  \
            f1->setConstants(&(model->residuals.residual[0].constants[model->residuals.residual[0].constantsDim * spanningTID]), model->residuals.residual[0].constantsDim); \
            fNext += f1->eval(sharedContext->xNext, X_DIM)->value; \
        } \
         for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[1].constantsCount; spanningTID += blockDim.x) { \
            f2->setConstants(&(model->residuals.residual[1].constants[model->residuals.residual[1].constantsDim * spanningTID]), model->residuals.residual[1].constantsDim); \
            fNext += f2->eval(sharedContext->xNext,X_DIM)->value; \
        }                    \
         for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[2].constantsCount; spanningTID += blockDim.x) { \
            f3->setConstants(&(model->residuals.residual[2].constants[model->residuals.residual[2].constantsDim * spanningTID]), model->residuals.residual[2].constantsDim); \
            fNext += f3->eval(sharedContext->xNext,X_DIM)->value; \
        }                    \

#endif

#endif //PARALLELLBFGS_LEVYMODEL_CUH
