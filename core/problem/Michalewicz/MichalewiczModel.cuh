//
// Created by spaceman on 2022. 12. 07..
//

#ifndef PARALLELLBFGS_MICHALEWICZMODEL_CUH
#define PARALLELLBFGS_MICHALEWICZMODEL_CUH

#include "../../optimizer/operators/perturb/Perturbator.h"
#include "../../common/Metrics.cuh"
#include "Michalewicz.cuh"
#include <random>

//#ifdef PROBLEM_MICHALEWICZ
class MichalewiczModel: public Model {
    std::mt19937 generator=std::mt19937(std::random_device()());
    Residual MichalewiczResidual[1]{};
public:

    MichalewiczModel():Model(){};
    explicit MichalewiczModel(Perturbator& perturbator) : Model(perturbator.populationSize,X_DIM) {
        residuals.residualCount=1;
        MichalewiczResidual[0].constantsCount= X_DIM;
        MichalewiczResidual[0].constantsDim=3;
        MichalewiczResidual[0].parametersDim=1;
        residuals.residual= reinterpret_cast<Residual *>(&MichalewiczResidual[0]);
    }

    void loadModel(void* dev_x, void* dev_xDE, void* dev_constantData, Metrics &metrics ) override {
        const int constantDataSize=residuals.residualDataSize();
        double x[modelPopulationSize]={};
        double data[constantDataSize]={};

        for(int i=0;i<modelPopulationSize;i++) {
            x[i]=std::uniform_real_distribution<double>(0, M_PI)(generator);
        }
        for(int i=0; i < MichalewiczResidual[0].constantsCount; i++) {
            data[3*i]=i;
            data[3*i+1]=-1.0;
            data[3*i+2]=((double)i+1)/M_PI;
        }
        metrics.getCudaEventMetrics().recordStartCopy();
        cudaMemcpy(dev_x, &x, modelPopulationSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_xDE, &x, modelPopulationSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_constantData, &data, constantDataSize * sizeof(double), cudaMemcpyHostToDevice);
        metrics.getCudaEventMetrics().recordStopCopy();
    }
};
#define DEFINE_RESIDUAL_FUNCTIONS() \
        Michalewicz f1 = Michalewicz(); \

#define INJECT_RESIDUAL_FUNCTIONS() \
        ((Model*)localContext.modelP)->residuals.residual[0].residualProblem=&f1; \
        ((Model*)localContext.modelP)->residuals.residual[0].constants = globalData; \

#define CAST_RESIDUAL_FUNCTIONS() \
        Michalewicz *f1 = ((Michalewicz *) model->residuals.residual[0].residualProblem); \

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

//#endif

#endif //PARALLELLBFGS_MICHALEWICZMODEL_CUH
