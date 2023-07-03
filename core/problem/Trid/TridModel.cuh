//
// Created by spaceman on 2022. 12. 08..
//

#ifndef PARALLELLBFGS_TRIDMODEL_CUH
#define PARALLELLBFGS_TRIDMODEL_CUH

#include "../../optimizer/operators/perturb/Perturbator.h"
#include "../../common/Metrics.cuh"
#include "Trid1.cuh"
#include "Trid2.cuh"

#ifdef PROBLEM_TRID
class TridModel: public Model {
    std::mt19937 generator=std::mt19937(std::random_device()());
    Residual TridResidual[2]{};
public:

    TridModel():Model(){};
    explicit TridModel(Perturbator& perturbator) : Model(perturbator.populationSize,X_DIM) {
        residuals.residualCount=2;
        TridResidual[0].constantsCount=X_DIM;
        TridResidual[0].constantsDim=2;
        TridResidual[0].parametersDim=1;
        TridResidual[1].constantsCount=X_DIM-1;
        TridResidual[1].constantsDim=3;
        TridResidual[1].parametersDim=2;
        residuals.residual= reinterpret_cast<Residual *>(&TridResidual[0]);
    }

    void loadModel(void* dev_x, void* dev_xDE, void* dev_constantData, Metrics &metrics,CUDAMemoryModel* model ) override {
        const int constantDataSize=residuals.residualDataSize();
        double x[modelPopulationSize]={};
        double data[constantDataSize]={};

        for(int i=0;i<modelPopulationSize;i++) {
            x[i]=std::uniform_real_distribution<double>(-100, 100)(generator);
        }
        //Set Trid1 residual data
        for(int i=0; i < TridResidual[0].constantsCount; i++) {
            data[2*i]=i;
            data[2*i+1]=1.0;
        }
        //Set Trid2 residual data
        for(int i=0; i < TridResidual[1].constantsCount; i++) {
            data[residuals.residual[0].getConstantsDim()+3*i]=i;
            data[residuals.residual[0].getConstantsDim()+3*i+1]=i+1;
            data[residuals.residual[0].getConstantsDim()+3*i+2]=-1.0;
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
        Trid1 f1 = Trid1(); \
        Trid2 f2 = Trid2(); \

#define INJECT_RESIDUAL_FUNCTIONS() \
        ((Model*)localContext.modelP)->residuals.residual[0].residualProblem=&f1; \
        ((Model*)localContext.modelP)->residuals.residual[0].constants = globalData; \
        ((Model*)localContext.modelP)->residuals.residual[1].residualProblem= &f2; \
        ((Model*)localContext.modelP)->residuals.residual[1].constants = ((Model*)localContext.modelP)->residuals.residual[0].constants  + ((Model*)localContext.modelP)->residuals.residual[0].getConstantsDim();

#define CAST_RESIDUAL_FUNCTIONS() \
        Trid1 *f1 = ((Trid1 *) model->residuals.residual[0].residualProblem); \
        Trid2 *f2 = ((Trid2 *) model->residuals.residual[1].residualProblem);

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
        }

#define COMPUTE_LINESEARCH() \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[0].constantsCount; spanningTID += blockDim.x) {  \
            f1->setConstants(&(model->residuals.residual[0].constants[model->residuals.residual[0].constantsDim * spanningTID]), model->residuals.residual[0].constantsDim); \
            fNext += f1->eval(sharedContext->xNext, X_DIM)->value; \
        }\
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[1].constantsCount; spanningTID += blockDim.x) { \
            f2->setConstants(&(model->residuals.residual[1].constants[model->residuals.residual[1].constantsDim * spanningTID]), model->residuals.residual[1].constantsDim); \
            fNext += f2->eval(sharedContext->xNext,X_DIM)->value; \
        }
#endif

#endif //PARALLELLBFGS_TRIDMODEL_CUH
