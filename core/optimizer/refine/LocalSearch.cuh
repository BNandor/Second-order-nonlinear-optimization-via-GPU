//
// Created by spaceman on 2022. 10. 17..
//

#ifndef PARALLELLBFGS_LOCALSEARCH_CUH
#define PARALLELLBFGS_LOCALSEARCH_CUH

#include "../../common/config/CUDAConfig.cuh"
#include "GradientDescent.cuh"
#include "LBFGS.cuh"

class LocalSearch {
protected:
    void* dev_globalContext;
    int iterations;

public:
    LocalSearch(){

    }
    LocalSearch(int iterations):iterations(iterations) {

    }
    virtual  void
    optimize(double *globalX, double *globalData,
             double *globalF
            , void *globalSharedContext, void * model,
             CUDAConfig cudaConfig
    )=0;

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
    double alpha;
    GDLocalSearch(){}
    GDLocalSearch(double alpha,int iterations):LocalSearch(iterations){
        this->alpha=alpha;
    }

    void
    optimize(double *globalX, double *globalData,
             double *globalF
            , void *globalSharedContext,void* model,
            CUDAConfig cudaConfig
    ) override{
        GD::optimize<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(globalX,globalData,globalF,(GD::GlobalData*)globalSharedContext,model,iterations,alpha);
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
    double alpha;
    LBFGSLocalSearch(){}
    LBFGSLocalSearch(double alpha,int iterations):LocalSearch(iterations){
        this->alpha=alpha;
    }
    void
    optimize(double *globalX, double *globalData,
             double *globalF
            , void *globalSharedContext,void* model,
             CUDAConfig cudaConfig
    ) override {
        LBFGS::optimize<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(globalX,globalData,globalF,(LBFGS::GlobalData*)globalSharedContext,model,iterations);
    };

    void setupGlobalData(int populationSize) override{
        if(dev_globalContext!= nullptr) {
            cudaFree(dev_globalContext);
        }
        cudaMalloc(&dev_globalContext, sizeof(LBFGS::GlobalData)*populationSize);
        printf("Allocating %lu global memory\n",sizeof(LBFGS::GlobalData)*populationSize);
    }
};

#endif //PARALLELLBFGS_LOCALSEARCH_CUH
