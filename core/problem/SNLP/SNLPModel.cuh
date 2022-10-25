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
    explicit SNLPModel(Perturbator& perturbator,int localIterations) : Model(perturbator,localIterations) {
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


#endif //PARALLELLBFGS_SNLPMODEL_CUH
