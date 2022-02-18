//
// Created by spaceman on 2022. 02. 16..
//

#ifndef PARALLELLBFGS_DIFFERENTIALEVOLUTION_CUH
#define PARALLELLBFGS_DIFFERENTIALEVOLUTION_CUH

#include <curand.h>
#include <curand_kernel.h>
#include "../common/Constants.cuh"

__global__
void setupCurand(curandState *state) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}

__global__
void differentialEvolutionStep(double *oldX, double *newX, curandState *curandState) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ unsigned sharedA;
    __shared__ unsigned sharedB;
    __shared__ unsigned sharedC;
    __shared__ unsigned sharedR;
    unsigned localA;
    unsigned localB;
    unsigned localC;
    unsigned localR;
    if (threadIdx.x == 0) {
        localA = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
        while (localA == blockIdx.x) {
            localA = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
        }
        localB = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
        while (localB == localA || localB == blockIdx.x) {
            localB = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
        }
        localC = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
        while (localC == localA || localC == localB || localC == blockIdx.x) {
            localC = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
        }
        sharedR = (int) truncf(curand_uniform(curandState + idx) * X_DIM);
        sharedA = localA;
        sharedB = localB;
        sharedC = localC;
        printf("DE: a:%d,b:%d,c:%d,x:%d \n", localA, localB, localC, blockIdx.x);
    }
    __syncthreads();
    localA = sharedA;
    localB = sharedB;
    localC = sharedC;
    localR = sharedR;

    // every thread has the same sharedA, sharedB, sharedC, blockId.x
    for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
        if (curand_uniform(curandState + idx) < CR || spanningTID == localR) {
            newX[blockIdx.x * X_DIM + spanningTID] = oldX[localA * X_DIM + spanningTID] + F * (oldX[localB * X_DIM +
                                                                                                    spanningTID] -
                                                                                               oldX[localC * X_DIM +
                                                                                                    spanningTID]);
        } else {
            newX[blockIdx.x * X_DIM + spanningTID] = oldX[blockIdx.x * X_DIM + spanningTID];
        }
    }
    // newX is complete in every thread in this block
}

__global__
void selectBestModels(const double *oldX, double *newX, const double *oldF, double *newF, unsigned generation) {
    if (oldF[blockIdx.x] < newF[blockIdx.x]) {
        for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
            newX[blockIdx.x * X_DIM + spanningTID] = oldX[blockIdx.x * X_DIM + spanningTID];
        }
        if (threadIdx.x == 0) {
            newF[blockIdx.x] = oldF[blockIdx.x];
        }
    }
    if (threadIdx.x == 0) {
        printf("Gen %u block %d f: %.10f\n", generation, blockIdx.x, newF[blockIdx.x]);
    }
}

#endif //PARALLELLBFGS_DIFFERENTIALEVOLUTION_CUH