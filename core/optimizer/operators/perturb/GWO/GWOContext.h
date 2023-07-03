//
// Created by spaceman on 2023. 06. 02..
//

#ifndef PARALLELLBFGS_GWOCONTEXT_H
#define PARALLELLBFGS_GWOCONTEXT_H

#include "../../../../../../../../../../usr/local/cuda-11.4/include/curand_kernel.h"
#include "../../../../common/config/CUDAConfig.cuh"
#include "../../../../common/Random.cuh"
#include "../Perturbator.h"
#include "../../../../common/model/BoundedParameter.cuh"
#include "../../select/Selector.cuh"
#include "../../refine/FunctionEvaluation.cuh"
#include <cmath>

#ifndef  gpuErrchk
#define
gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif

__global__
void greyWolfOptimizerStep(double *oldX, double *newX, Model*model, curandState *curandState,double * currentCosts,double a) {
#ifdef SAFE
    assert(blockDim.x >= 4);
#endif

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ unsigned alphaIndex;
    __shared__ unsigned betaIndex;
    __shared__ unsigned deltaIndex;

     unsigned localAlphaIndex=0;
     unsigned localBetaIndex=0;
     unsigned localDeltaIndex=0;

    if (threadIdx.x == 0) {
        for (int i = 0; i < gridDim.x; i++) {
                if (currentCosts[i] < currentCosts[localAlphaIndex]) {
                    localAlphaIndex=i;
                }
        }
        for (int i = 0; i < gridDim.x; i++) {
            if (i != localAlphaIndex && currentCosts[i] < currentCosts[localBetaIndex]) {
                localBetaIndex = i;
            }
        }
        for (int i = 0; i < gridDim.x; i++) {
            if (i != localAlphaIndex && i != localBetaIndex && currentCosts[i] < currentCosts[localDeltaIndex]) {
                localDeltaIndex = i;
            }
        }
        alphaIndex=localAlphaIndex;
        betaIndex=localBetaIndex;
        deltaIndex=localDeltaIndex;
    }
    __syncthreads();

    for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
        double X=oldX[blockIdx.x * X_DIM + spanningTID];
        double A1=2.0 * a * curand_uniform(curandState + idx)-a;
        double C1=2.0 * curand_uniform(curandState + idx);
        double Xalpha=oldX[alphaIndex* X_DIM + spanningTID];
        double Dalpha=std::abs(C1 *Xalpha - X);

        double A2=2.0 * a * curand_uniform(curandState + idx)-a;
        double C2=2.0 * curand_uniform(curandState + idx);
        double Xbeta=oldX[betaIndex* X_DIM + spanningTID];
        double Dbeta=std::abs(C2*Xbeta - X);

        double A3=2.0 * a * curand_uniform(curandState + idx)-a;
        double C3=2.0 * curand_uniform(curandState + idx);
        double Xdelta=oldX[deltaIndex* X_DIM + spanningTID];
        double Ddelta=std::abs(C3*Xdelta - X);

        newX[blockIdx.x * X_DIM + spanningTID] =(1.0/3.0)*((Xalpha - A1 * Dalpha) + (Xbeta - A2 * Dbeta) + (Xdelta - A3 * Ddelta));
//        newX[blockIdx.x * X_DIM + spanningTID]=oldX[blockIdx.x * X_DIM + spanningTID];
    }
}

class GWOContext : public Perturbator {

    void* dev_globalContext;
    void setupGlobalData(int populationSize) {
        if(dev_globalContext!= nullptr) {
            gpuErrchk(cudaFree(dev_globalContext));
            dev_globalContext=0;
        }
        gpuErrchk(cudaMalloc(&dev_globalContext, sizeof(FuncEval::GlobalData)*populationSize));
        printf("Allocating %lu global memory for GWO \n",sizeof(FuncEval::GlobalData)*populationSize);
    }

public:

    GWOContext() {
        populationSize=POPULATION_SIZE;
        setupGlobalData(populationSize);
    }

    ~GWOContext() {
        if(dev_globalContext!= nullptr) {
            gpuErrchk(cudaFree(dev_globalContext));
            dev_globalContext=0;
        }
    }

    void operate(CUDAMemoryModel* cudaMemoryModel) override{
        if(fEvals>0) {
            perturb(cudaMemoryModel->cudaConfig,
                    cudaMemoryModel->model,
                    cudaMemoryModel->dev_Model,
                    cudaMemoryModel->dev_x1,
                    cudaMemoryModel->dev_x2,
                    cudaMemoryModel->dev_data,
                    cudaMemoryModel->dev_F1,
                    cudaMemoryModel->dev_F2,
                    &cudaMemoryModel->cudaRandom,
                    cudaMemoryModel->isBounded,
                    cudaMemoryModel->dev_lower_bounds,
                    cudaMemoryModel->dev_upper_bounds);
        }
    }

    void perturb(CUDAConfig &cudaConfig,Model *model, Model * dev_model,double * dev_x1, double * dev_x2, double* dev_data, double* oldCosts, double* newCosts, Random* cudaRandom,
                 bool isbounded,
                 double *globalLowerBounds,
                 double *globalUpperBounds) override {
        greyWolfOptimizerStep<<<model->populationSize, cudaConfig.threadsPerBlock>>>(dev_x1, dev_x2, dev_model,
                cudaRandom->dev_curandState,oldCosts,parameters.values["GWO_a"].value);
        if(isbounded){
            snapToBounds<<<model->populationSize, cudaConfig.threadsPerBlock>>>(dev_x2,globalLowerBounds,globalUpperBounds,model->modelSize);
        }
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        evaluateF(cudaConfig,dev_model,dev_x2,dev_data,newCosts);
    }

    void evaluateF(CUDAConfig &cudaConfig,Model * dev_model,double * dev_x,double* dev_data,double* costs) {
        FuncEval::evaluateF<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(dev_x,dev_data,costs,(FuncEval::GlobalData*)dev_globalContext,dev_model);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
};


#endif //PARALLELLBFGS_GWOCONTEXT_H
