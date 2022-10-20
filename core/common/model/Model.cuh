//
// Created by spaceman on 2022. 10. 20..
//

#ifndef PARALLELLBFGS_MODEL_CUH
#define PARALLELLBFGS_MODEL_CUH

#include <fstream>
#include "../../optimizer/perturb/Perturbator.h"
#include "../Metrics.cuh"
class Residual {
public:
    int parametersDim;
    int constantsDim;
    int constantsCount;

    int getConstantsDim(){
        return constantsDim*constantsCount;
    }
};

class Residuals {
public:

    int residualCount;
    Residual *residual;

    int residualDataSize() const {
        int size=0;
#ifdef SAFE
        assert(residualCount>0);
        assert(residual!=0);
#endif
        printf("pointer %p\n",residual);
        Residual* it=residual;
        for(int i=0;i<residualCount;i++) {
            size+=(it->constantsCount)*(it->constantsDim);
            it++;
            printf("count %i\n",it->constantsCount);
            printf("dim %i\n",it->constantsDim);
        }

#ifdef SAFE
        assert(size>0);
#endif
        return size;
    }
};

class Model {
public:

    int modelSize;
    int modelPopulationSize;
    int populationSize;
    Residuals residuals;
    int localIterations;

    Model()=default;
    Model(Perturbator& perturbator,int localIterations) {
        modelSize=X_DIM;
        modelPopulationSize=perturbator.populationSize*modelSize;
        populationSize=perturbator.populationSize;
        this->localIterations=localIterations;
    }

    virtual void loadModel(void* dev_x,void* dev_constantData,Metrics &metrics)=0;

    void readPopulation(double *x, unsigned xSize, std::string filename) {
        std::fstream input;
        input.open(filename.c_str());
        if (input.is_open()) {
            unsigned cData = 0;
            while (input >> x[cData]) {
                cData++;
            }
            std::cout << "read: " << cData << " expected: " << xSize
                      << std::endl;
            assert(cData == xSize);
        } else {
            std::cerr << "err: could not open " << filename << std::endl;
            exit(1);
        }
    }

};

#endif //PARALLELLBFGS_MODEL_CUH
