//
// Created by spaceman on 2022. 10. 22..
//

#ifndef PARALLELLBFGS_GACONTEXT_CUH
#define PARALLELLBFGS_GACONTEXT_CUH
#include "../Perturbator.h"

__global__
void geneticAlgorithmStep(double *oldX, double *newX,Model *model,double * currentCosts, double populationMutationRate, double crossoverRate, double mutationRate, double mutationSize, double parentPoolRatio,curandState *curandState) {
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
        hasToMutate=curand_uniform(curandState + idx) < populationMutationRate;
        if(hasToMutate) {
//            localA = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
//            while (localA == blockIdx.x) {
//                localA = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
//            }
//            localB = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
//            while (localB == localA || localB == blockIdx.x) {
//                localB = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
//            }
//            localC = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
//            while (localC == localA || localC == localB || localC == blockIdx.x) {
//                localC = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
//            }
//            sharedR = (int) truncf( crossoverRate * X_DIM);
//            unsigned leastCost1=localA;
//            unsigned leastCost2=localB;
//            if (currentCosts[leastCost2] < currentCosts[leastCost1]){
//                unsigned tmp=leastCost1;
//                leastCost1=leastCost2;
//                leastCost2=tmp;
//            }
//            if (currentCosts[localC] < currentCosts[leastCost1] || currentCosts[localC] < currentCosts[leastCost2]){
//                leastCost2=localC;
//            }
            sharedR = (int) truncf( crossoverRate * X_DIM);
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
    localA = sharedA;
    localB = sharedB;
    localR = sharedR;

//  1,2,3,4
//  m[1,2,3,4]
//  1',2',3',4'
//  m[1',2',3',4']
//  min(m[1,2,3,4],m[1',2',3',4'])
//  epsilon kivétele
    // every thread has the same sharedA, sharedB, sharedC, blockId.x
//    if(blockIdx.x !=0) {
    for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
        if (hasToMutate) {
            if( spanningTID <= localR) {
                newX[blockIdx.x * X_DIM + spanningTID] =
                        oldX[localA * X_DIM + spanningTID];
            }else{
                newX[blockIdx.x * X_DIM + spanningTID] =
                        oldX[localB * X_DIM + spanningTID];
            }

            if(curand_uniform(curandState + idx) < mutationRate) {
                newX[blockIdx.x * X_DIM + spanningTID]+=curand_normal(curandState + idx)*mutationSize;
            }
        } else {
            newX[blockIdx.x * X_DIM + spanningTID] = oldX[blockIdx.x * X_DIM + spanningTID];
        }
    }
//    } else {
//        for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
//            newX[blockIdx.x * X_DIM + spanningTID] = oldX[blockIdx.x * X_DIM + spanningTID];
//        }
//    }
    // newX is complete in every thread in this block
}


class GAContext : public Perturbator {
public:
    GAContext() {
        populationSize=POPULATION_SIZE;
    }

    void perturb(CUDAConfig &cudaConfig, Model *model,double * dev_x1, double * dev_x2,double * currentCosts, Random* cudaRandom ) override {
        geneticAlgorithmStep<<<model->populationSize, cudaConfig.threadsPerBlock>>>(dev_x1, dev_x2, model, currentCosts,populationMutationRate, crossoverRate,mutationRate,mutationSize,parentPoolRatio,cudaRandom->dev_curandState);
    }

    double force=DE_FORCE;
    double populationMutationRate=0.9;
            double crossoverRate=0.5;
            double mutationRate=0.5;
            double mutationSize=50;
            double parentPoolRatio=0.3;
};

#endif //PARALLELLBFGS_GACONTEXT_CUH
