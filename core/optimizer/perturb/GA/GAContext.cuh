//
// Created by spaceman on 2022. 10. 22..
//

#ifndef PARALLELLBFGS_GACONTEXT_CUH
#define PARALLELLBFGS_GACONTEXT_CUH
#include "../Perturbator.h"
__device__
int minIndex(int a, int b){
    if(a <b){
        return a;
    }
    return b;
}
__device__
int maxIndex(int a, int b){
    if(a <b){
        return b;
    }
    return a;
}

__global__
void geneticAlgorithmStep(double *oldX, double *newX,Model *model,
                          double * currentCosts,
                          double crossoverRate,
                          double crossoverPoint,
                          double mutationRate,
                          double mutationSize,
                          double parentPoolRatio,
                          double alpha,
                          curandState *curandState) {
#ifdef SAFE
    assert(blockDim.x >= 4);
    assert((int) truncf((blockDim.x-1)*parentPoolRatio)>2);
#endif

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ unsigned sharedA;
    __shared__ unsigned sharedB;
    __shared__ unsigned sharedR;
    unsigned localA;
    unsigned localB;
    unsigned localC;
    unsigned localR;
    __shared__ bool hasToMutate;
    if (threadIdx.x == 0) {
        hasToMutate=curand_uniform(curandState + idx) < crossoverRate;
        sharedR=(int)truncf( crossoverPoint * X_DIM);
        if(hasToMutate) {
            int parentA=-1;
            int parentB=-1;
            double minA=INT_MAX;
            double minB=INT_MAX;
            while(parentA == -1) {
                printf("Selecting parentA for %d\n",blockIdx.x);
                for (int i = 0; i < gridDim.x; i++) {
                    if (curand_uniform(curandState + idx) < parentPoolRatio && blockIdx.x != i) {
                        if (parentA == -1 || currentCosts[i] < minA) {
                            minA = currentCosts[i];
                            parentA = i;
                        }
                    }
                }
            }
            while(parentB == -1) {
                printf("Selecting parentB for %d\n",blockIdx.x);
                for (int i = 0; i < gridDim.x; i++) {
                    if (curand_uniform(curandState + idx) < parentPoolRatio && blockIdx.x != i && i != parentA) {
                        if (parentB == -1 || currentCosts[i] < minB) {
                            minB = currentCosts[i];
                            parentB = i;
                        }
                    }
                }
            }
            sharedA = parentA;
            sharedB = parentB;
            printf("DE: a:%d,b:%d,x:%d \n", sharedA, sharedB, blockIdx.x);
        }
    }
    __syncthreads();
    localR=sharedR;
    if(blockIdx.x % 2 == 0) {
        localA = minIndex(sharedA,sharedB);
        localB = maxIndex(sharedA,sharedB);
    } else {
        localA = maxIndex(sharedA,sharedB);
        localB = minIndex(sharedA,sharedB);
    }
//    if(blockIdx.x !=0) {
    for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
        if (hasToMutate) {
            if( spanningTID <= localR) {
                newX[blockIdx.x * X_DIM + spanningTID] =alpha * oldX[localA * X_DIM + spanningTID] + (1.0 - alpha)*oldX[localB * X_DIM + spanningTID];
            }else{
                newX[blockIdx.x * X_DIM + spanningTID] =alpha * oldX[localB * X_DIM + spanningTID] + (1.0 - alpha)*oldX[localA * X_DIM + spanningTID];
            }
            if(curand_uniform(curandState + idx) < mutationRate) {
                newX[blockIdx.x * X_DIM + spanningTID]+=curand_normal(curandState + idx)*mutationSize;
            }
        } else {
            newX[blockIdx.x * X_DIM + spanningTID] = oldX[blockIdx.x * X_DIM + spanningTID];
        }
    }
    // newX is complete in every thread in this block
}


class GAContext : public Perturbator {
public:
    GAContext() {
        populationSize=POPULATION_SIZE;
    }

    void perturb(CUDAConfig &cudaConfig, Model *model,double * dev_x1, double * dev_x2,double * currentCosts, Random* cudaRandom ) override {
        geneticAlgorithmStep<<<model->populationSize, cudaConfig.threadsPerBlock>>>(dev_x1, dev_x2, model, currentCosts,
                                                                                    crossoverRate,
                                                                                    crossoverPoint,
                                                                                    mutationRate,
                                                                                    mutationSize,
                                                                                    parentPoolRatio,
                                                                                    alpha,
                                                                                    cudaRandom->dev_curandState);
    }
    double force=DE_FORCE;
    double crossoverRate=0.9;
    double crossoverPoint=0.5;
    double mutationRate=0.5; // decrease
    double mutationSize=50; //
    double parentPoolRatio=0.3;
    double alpha=0.2;
};

#endif //PARALLELLBFGS_GACONTEXT_CUH
