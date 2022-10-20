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
public:
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
    void
    optimize(double *globalX, double *globalData,
             double *globalF
            , void *globalSharedContext,void* model,
            CUDAConfig cudaConfig
    ) override{
        GD::optimize<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(globalX,globalData,globalF,(GD::GlobalData*)globalSharedContext,model);
    };

     void setupGlobalData(int populationSize) {
        if(dev_globalContext!= nullptr) {
            cudaFree(dev_globalContext);
        }
        cudaMalloc(&dev_globalContext, sizeof(GD::GlobalData)*populationSize);
        printf("Allocating %lu global memory\n",sizeof(GD::GlobalData)*populationSize);
    }
};

class LBFGSLocalSearch: public LocalSearch {
    void
    optimize(double *globalX, double *globalData,
             double *globalF
            , void *globalSharedContext,void* model,
             CUDAConfig cudaConfig
    ) override {
        LBFGS::optimize<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(globalX,globalData,globalF,(LBFGS::GlobalData*)globalSharedContext,model);
    };

    void setupGlobalData(int populationSize) {
        if(dev_globalContext!= nullptr) {
            cudaFree(dev_globalContext);
        }
        cudaMalloc(&dev_globalContext, sizeof(LBFGS::GlobalData)*populationSize);
        printf("Allocating %lu global memory\n",sizeof(LBFGS::GlobalData)*populationSize);
    }
};

#endif //PARALLELLBFGS_LOCALSEARCH_CUH
