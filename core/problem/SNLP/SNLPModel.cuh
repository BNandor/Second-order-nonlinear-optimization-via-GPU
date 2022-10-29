//
// Created by spaceman on 2022. 10. 25..
//

#ifndef PARALLELLBFGS_SNLPMODEL_CUH
#define PARALLELLBFGS_SNLPMODEL_CUH

#include "../../optimizer/perturb/Perturbator.h"
#include "../../common/Metrics.cuh"

class SNLPModel: public Model {
    Residual SNLPResidual[2]{};
public:

    SNLPModel():Model(){};
    explicit SNLPModel(Perturbator& perturbator) : Model(perturbator) {
        residuals.residualCount=2;
        SNLPResidual[0].constantsCount=RESIDUAL_CONSTANTS_COUNT_1;
        SNLPResidual[0].constantsDim=RESIDUAL_CONSTANTS_DIM_1;
        SNLPResidual[0].parametersDim=RESIDUAL_PARAMETERS_DIM_1;
        SNLPResidual[1].constantsCount=RESIDUAL_CONSTANTS_COUNT_2;
        SNLPResidual[1].constantsDim=RESIDUAL_CONSTANTS_DIM_2;
        SNLPResidual[1].parametersDim=RESIDUAL_PARAMETERS_DIM_2;
        residuals.residual= reinterpret_cast<Residual *>(&SNLPResidual[0]);
    }

    void loadModel(void* dev_x,void* dev_constantData, Metrics &metrics ) override {
        const int constantDataSize=residuals.residualDataSize();
        double x[modelPopulationSize]={};
        double data[constantDataSize]={};

        readSNLPProblem(data, PROBLEM_PATH);
        readSNLPAnchors(data + residuals.residual[0].getConstantsDim(),
                        PROBLEM_ANCHOR_PATH);
        readPopulation(x, modelPopulationSize,PROBLEM_INPUT_POPULATION_PATH);

        metrics.getCudaEventMetrics().recordStartCopy();
        cudaMemcpy(dev_x, &x, modelPopulationSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_constantData, &data, constantDataSize * sizeof(double), cudaMemcpyHostToDevice);
        metrics.getCudaEventMetrics().recordStopCopy();
    }

    void readSNLPProblem(double *data, std::string filename) {
        std::fstream input;
        input.open(filename.c_str());
        if (input.is_open()) {
            unsigned cData = 0;
            while (input >> data[cData]) {
                cData++;
            }
            std::cout << "read: " << cData << " expected: " << residuals.residual[0].getConstantsDim()
                      << std::endl;
            assert(cData == residuals.residual[0].getConstantsDim());
        } else {
            std::cerr << "err: could not open " << filename << std::endl;
            exit(1);
        }
    }

    void readSNLPAnchors(double *data, std::string filename) {
        std::fstream input;
        input.open(filename.c_str());
        if (input.is_open()) {
            unsigned cData = 0;
            while (input >> data[cData]) {
                cData++;
            }
            std::cout << "read: " << cData << " expected: " << residuals.residual[1].getConstantsDim()
                      << std::endl;
            assert(cData == residuals.residual[1].getConstantsDim());
        } else {
            std::cerr << "err: could not open " << filename << std::endl;
            exit(1);
        }
    }
};

#ifdef PROBLEM_SNLP

#define DEFINE_RESIDUAL_FUNCTIONS() \
        SNLP f1 = SNLP(); \
        SNLPAnchor f2 = SNLPAnchor(); \

#define INJECT_RESIDUAL_FUNCTIONS() \
        ((Model*)localContext.modelP)->residuals.residual[0].residualProblem=&f1; \
        ((Model*)localContext.modelP)->residuals.residual[0].constants = globalData; \
        ((Model*)localContext.modelP)->residuals.residual[1].residualProblem= &f2; \
        ((Model*)localContext.modelP)->residuals.residual[1].constants = ((Model*)localContext.modelP)->residuals.residual[0].constants  + ((Model*)localContext.modelP)->residuals.residual[0].getConstantsDim();

#define CAST_RESIDUAL_FUNCTIONS() \
        SNLP *f1 = ((SNLP *) model->residuals.residual[0].residualProblem); \
        SNLPAnchor *f2 = ((SNLPAnchor *) model->residuals.residual[1].residualProblem);

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

#endif //PARALLELLBFGS_SNLPMODEL_CUH
