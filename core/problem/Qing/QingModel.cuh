//
// Created by spaceman on 2022. 12. 12..
//

#ifndef PARALLELLBFGS_QINGMODEL_CUH
#define PARALLELLBFGS_QINGMODEL_CUH


#include "../../optimizer/operators/perturb/Perturbator.h"
#include "../../common/Metrics.cuh"
#include "Qing.cuh"
#include <math.h>

#ifdef PROBLEM_QING
class QingModel: public Model {
    std::mt19937 generator=std::mt19937(std::random_device()());
    Residual QingResidual[1]{};
public:

    QingModel():Model(){};
    explicit QingModel(Perturbator& perturbator) : Model(perturbator.populationSize,X_DIM) {
        residuals.residualCount=1;
        QingResidual[0].constantsCount=X_DIM;
        QingResidual[0].constantsDim=1;
        QingResidual[0].parametersDim=1;
        residuals.residual= reinterpret_cast<Residual *>(&QingResidual[0]);
    }

    void loadModel(void* dev_x,void* dev_xDE,void* dev_constantData, Metrics &metrics ) override {
        const int constantDataSize=residuals.residualDataSize();
        double x[modelPopulationSize]={};
        double data[constantDataSize]={};

        for(int i=0;i<modelPopulationSize;i++) {
            x[i]=std::uniform_real_distribution<double>(-500, 500)(generator);
        }
        //Set Qing residual data
        for(int i=0; i < QingResidual[0].constantsCount; i++) {
            data[i]=i+1;
        }
        metrics.getCudaEventMetrics().recordStartCopy();
        cudaMemcpy(dev_x, &x, modelPopulationSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_xDE, &x, modelPopulationSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_constantData, &data, constantDataSize * sizeof(double), cudaMemcpyHostToDevice);
        metrics.getCudaEventMetrics().recordStopCopy();
    }
};



#define DEFINE_RESIDUAL_FUNCTIONS() \
        Qing f1 = Qing();

#define INJECT_RESIDUAL_FUNCTIONS() \
        ((Model*)localContext.modelP)->residuals.residual[0].residualProblem=&f1; \
        ((Model*)localContext.modelP)->residuals.residual[0].constants = globalData;

#define CAST_RESIDUAL_FUNCTIONS() \
        Qing *f1 = ((Qing *) model->residuals.residual[0].residualProblem);

#define COMPUTE_RESIDUALS() \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[0].constantsCount; spanningTID += blockDim.x) { \
            f1->setConstants(&(model->residuals.residual[0].constants[model->residuals.residual[0].constantsDim * spanningTID]), model->residuals.residual[0].constantsDim); \
            localContext->threadF += f1->eval(x, X_DIM)->value; \
            f1->evalJacobian(); \
            for (unsigned j = 0; j < model->residuals.residual[0].parametersDim; j++) { \
                atomicAdd(&dx[f1->ThisJacobianIndices[j]], f1->operatorTree[f1->constantSize + j].derivative); } \
        }

#define COMPUTE_LINESEARCH() \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[0].constantsCount; spanningTID += blockDim.x) {  \
            f1->setConstants(&(model->residuals.residual[0].constants[model->residuals.residual[0].constantsDim * spanningTID]), model->residuals.residual[0].constantsDim); \
            fNext += f1->eval(sharedContext->xNext, X_DIM)->value; \
        }
#endif

#endif //PARALLELLBFGS_QINGMODEL_CUH
