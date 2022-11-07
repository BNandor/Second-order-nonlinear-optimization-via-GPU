//
// Created by spaceman on 2022. 10. 17..
//

#ifndef PARALLELLBFGS_LOCALSEARCH_CUH
#define PARALLELLBFGS_LOCALSEARCH_CUH

#include "../../common/config/CUDAConfig.cuh"
#include "../../common/model/BoundedParameter.cuh"
#include "GradientDescent.cuh"
#include "LBFGS.cuh"
#include "../Operator.h"

class LocalSearch : public Operator{
protected:
    void* dev_globalContext;
public:
    int functionEvaluations;

    LocalSearch() {
    }

    LocalSearch(int iterations): functionEvaluations(iterations) {

    }
    virtual  void
    optimize(double *globalX, double *globalData,
             double *globalF
            , void *globalSharedContext, void * model,
             CUDAConfig cudaConfig
    )=0;

    int fEvaluationCount() {
        return functionEvaluations;
    }

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
    GDLocalSearch(double alpha,int fevaluations):LocalSearch(fevaluations) {
        std::unordered_map<std::string,BoundedParameter> gdParams=std::unordered_map<std::string,BoundedParameter>();
        gdParams["GD_ALPHA"]=BoundedParameter(alpha, 0.5, 100);
        gdParams["GD_ITERATIONS"]=BoundedParameter(fevaluations, 0, 10000);
        parameters=OperatorParameters(gdParams);
    }

    void
    optimize(double *globalX, double *globalData,
             double *globalF
            , void *globalSharedContext,void* model,
            CUDAConfig cudaConfig
    ) override{
        GD::optimize<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(globalX,globalData,globalF,(GD::GlobalData*)globalSharedContext,model,
                                                                               parameters.values["GD_ITERATIONS"].value,
                                                                               parameters.values["GD_ALPHA"].value);
    };

     void setupGlobalData(int populationSize) override {
        if(dev_globalContext!= nullptr) {
            cudaFree(dev_globalContext);
        }
        cudaMalloc(&dev_globalContext, sizeof(GD::GlobalData)*populationSize);
        printf("Allocating %lu global memory\n",sizeof(GD::GlobalData)*populationSize);
    }
};

class LBFGSLocalSearch: public LocalSearch {
public:
    LBFGSLocalSearch(){}
    LBFGSLocalSearch(double alpha,int fevaluations):LocalSearch(fevaluations) {
        std::unordered_map<std::string,BoundedParameter> lbfgsParams=std::unordered_map<std::string,BoundedParameter>();
        lbfgsParams["LBFGS_ALPHA"]=BoundedParameter(alpha, 0.5, 100);
        lbfgsParams["LBFGS_ITERATIONS"]=BoundedParameter(fevaluations, 0, 10000);
        lbfgsParams["LBFGS_C1"]=BoundedParameter(0.0001, 0.0, 1.0);
        lbfgsParams["LBFGS_C2"]=BoundedParameter(0.9, 0.0, 1.0);
        parameters=OperatorParameters(lbfgsParams);
    }
    void
    optimize(double *globalX, double *globalData,
             double *globalF
            , void *globalSharedContext,void* model,
             CUDAConfig cudaConfig
    ) override {
        LBFGS::optimize<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(globalX,globalData,globalF,(LBFGS::GlobalData*)globalSharedContext, model,
                                                                                    parameters.values["LBFGS_ITERATIONS"].value,
                                                                                    parameters.values["LBFGS_ALPHA"].value,
                                                                                    parameters.values["LBFGS_C1"].value,
                                                                                    parameters.values["LBFGS_C2"].value);
    };

    void setupGlobalData(int populationSize) override {
        if(dev_globalContext!= nullptr) {
            cudaFree(dev_globalContext);
        }
        cudaMalloc(&dev_globalContext, sizeof(LBFGS::GlobalData)*populationSize);
        printf("Allocating %lu global memory\n",sizeof(LBFGS::GlobalData)*populationSize);
    }
};

#endif //PARALLELLBFGS_LOCALSEARCH_CUH
