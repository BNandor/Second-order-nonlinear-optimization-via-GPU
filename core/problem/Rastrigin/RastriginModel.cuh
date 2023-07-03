//
// Created by spaceman on 2022. 12. 10..
//

#ifndef PARALLELLBFGS_RASTRIGINMODEL_CUH
#define PARALLELLBFGS_RASTRIGINMODEL_CUH

#include "../../optimizer/operators/perturb/Perturbator.h"
#include "../../common/Metrics.cuh"
#include "Rastrigin.cuh"
#include <math.h>

#ifdef PROBLEM_RASTRIGIN
class RastriginModel: public Model {
    std::mt19937 generator=std::mt19937(std::random_device()());
    Residual RastriginResidual[1]{};
public:

    RastriginModel():Model(){};
    explicit RastriginModel(Perturbator& perturbator) : Model(perturbator.populationSize,X_DIM) {
        residuals.residualCount=1;
        RastriginResidual[0].constantsCount=X_DIM;
        RastriginResidual[0].constantsDim=3;
        RastriginResidual[0].parametersDim=1;
        residuals.residual= reinterpret_cast<Residual *>(&RastriginResidual[0]);
    }

    void loadModel(void* dev_x, void* dev_xDE, void* dev_constantData, Metrics &metrics,CUDAMemoryModel* model ) override {
        const int constantDataSize=residuals.residualDataSize();
        double x[modelPopulationSize]={};
        double data[constantDataSize]={};

        for(int i=0;i<modelPopulationSize;i++) {
            x[i]=std::uniform_real_distribution<double>(-5.12, 5.12)(generator);
        }
        //Set Rastrigin residual data
        for(int i=0; i < RastriginResidual[0].constantsCount; i++) {
            data[3*i]=i;
            data[3*i+1]=10.0;
            data[3*i+2]=2*M_PI;
        }

//        std::cout<<"X data:";
//        for(int i=0;i<modelPopulationSize;i++){
//            std::cout<<x[i]<<",";
//        }
//        std::cout<<"\nResidual1 data:";
//        for(int i=0;i<residuals.residual[0].getConstantsDim();i++){
//            std::cout<<data[i]<<",";
//        }
//        std::cout<<"\nResidual2 data:";
//        for(int i=0;i<residuals.residual[1].getConstantsDim();i++){
//            std::cout<<data[residuals.residual[0].getConstantsDim()+i]<<",";
//        }std::cout<<std::endl;
        metrics.getCudaEventMetrics().recordStartCopy();
        cudaMemcpy(dev_x, &x, modelPopulationSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_xDE, &x, modelPopulationSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_constantData, &data, constantDataSize * sizeof(double), cudaMemcpyHostToDevice);
        metrics.getCudaEventMetrics().recordStopCopy();
    }
};



#define DEFINE_RESIDUAL_FUNCTIONS() \
        Rastrigin f1 = Rastrigin();

#define INJECT_RESIDUAL_FUNCTIONS() \
        ((Model*)localContext.modelP)->residuals.residual[0].residualProblem=&f1; \
        ((Model*)localContext.modelP)->residuals.residual[0].constants = globalData;

#define CAST_RESIDUAL_FUNCTIONS() \
        Rastrigin *f1 = ((Rastrigin *) model->residuals.residual[0].residualProblem);

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
#endif //PARALLELLBFGS_RASTRIGINMODEL_CUH
