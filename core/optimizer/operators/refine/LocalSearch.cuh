//
// Created by spaceman on 2022. 10. 17..
//

#ifndef PARALLELLBFGS_LOCALSEARCH_CUH
#define PARALLELLBFGS_LOCALSEARCH_CUH

#include "../../../common/config/CUDAConfig.cuh"
#include "../../../common/model/BoundedParameter.cuh"
#include "GradientDescent.cuh"
#include "LBFGS.cuh"
#include "../Operator.h"

class LocalSearch : public Operator{
protected:
    void* dev_globalContext;
public:

    LocalSearch() {
    }

    virtual  void
    optimize(double *globalX, double *globalData,
             double *globalF
            , void *globalSharedContext, void * model,
             CUDAConfig cudaConfig
    )=0;

    virtual int fEvaluationCount() =0;

    ~LocalSearch() {
        if(dev_globalContext!= nullptr) {
            cudaFree(dev_globalContext);
        }
    }

    virtual void setupGlobalData(int populationSize) =0;

    void *getDevGlobalContext() const {
        return dev_globalContext;
    }
};

class GDLocalSearch: public LocalSearch {
public:
    GDLocalSearch(){}

    void operate(CUDAMemoryModel* cudaMemoryModel) override {
        optimize(cudaMemoryModel->dev_x2,
                 cudaMemoryModel->dev_data,
                 cudaMemoryModel->dev_F2,
                 dev_globalContext,
                 cudaMemoryModel->dev_Model,
                 cudaMemoryModel->cudaConfig);
        cudaMemoryModel->swapModels();
    }

    void
    optimize(double *globalX, double *globalData,
             double *globalF
            , void *globalSharedContext,void* model,
            CUDAConfig cudaConfig
    ) override{
        GD::optimize<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(globalX,globalData,globalF,(GD::GlobalData*)globalSharedContext,model,
                                                                               truncf(parameters.values["GD_FEVALS"].value),
                                                                               parameters.values["GD_ALPHA"].value);
    };

    int fEvaluationCount() override {
        return parameters.values["GD_FEVALS"].value;
    }

     void setupGlobalData(int populationSize) override {
        if(dev_globalContext!= nullptr) {
            cudaFree(dev_globalContext);
        }
        cudaMalloc(&dev_globalContext, sizeof(GD::GlobalData)*populationSize);
        printf("Allocating %lu global memory for GD\n",sizeof(GD::GlobalData)*populationSize);
    }
};

class LBFGSLocalSearch: public LocalSearch {
private:
    void setParameterInvariance() {
        if(parameters.values["LBFGS_C1"].value > parameters.values["LBFGS_C2"].value){
            std::swap(parameters.values["LBFGS_C1"].value,parameters.values["LBFGS_C2"].value);
        }
    }

public:
    LBFGSLocalSearch(){}

    void operate(CUDAMemoryModel* cudaMemoryModel) override {
        // TODO add switch for dev_x1
        optimize(cudaMemoryModel->dev_x2,
                 cudaMemoryModel->dev_data,
                 cudaMemoryModel->dev_F2,
                 dev_globalContext,
                 cudaMemoryModel->dev_Model,
                 cudaMemoryModel->cudaConfig);
        cudaMemoryModel->swapModels();
    }

    void optimize(double *globalX, double *globalData,
             double *globalF
            , void *globalSharedContext,void* model,
             CUDAConfig cudaConfig
    ) override {

        LBFGS::optimize<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(globalX,globalData,globalF,(LBFGS::GlobalData*)globalSharedContext, model,
                                                                                    truncf(parameters.values["LBFGS_FEVALS"].value),
                                                                                    parameters.values["LBFGS_ALPHA"].value,
                                                                                    parameters.values["LBFGS_C1"].value,
                                                                                    parameters.values["LBFGS_C2"].value);
    };

    int fEvaluationCount() override {
        return parameters.values["LBFGS_FEVALS"].value + LBFGS_M;
    }

    void setupGlobalData(int populationSize) override {
        if(dev_globalContext!= nullptr) {
            cudaFree(dev_globalContext);
        }
        cudaMalloc(&dev_globalContext, sizeof(LBFGS::GlobalData)*populationSize);
        printf("Allocating %lu global memory for LBFGS\n",sizeof(LBFGS::GlobalData)*populationSize);
    }
};

#endif //PARALLELLBFGS_LOCALSEARCH_CUH
