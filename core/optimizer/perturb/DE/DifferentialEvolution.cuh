//
// Created by spaceman on 2022. 02. 16..
//

#ifndef PARALLELLBFGS_DIFFERENTIALEVOLUTION_CUH
#define PARALLELLBFGS_DIFFERENTIALEVOLUTION_CUH

#include <curand.h>
#include <curand_kernel.h>
#include "../../../common/Constants.cuh"
#include <assert.h>

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

__global__
void printBestF(double *fs,unsigned size) {
    if (threadIdx.x == 0 ) {
        double min = fs[0];
        for (
                unsigned i = 1; i < size; i++) {
            if (min > fs[i]) {
                min = fs[i];
            }
        }
        printf("\nfinal f: %.10f", min);
    }

}
#endif //PARALLELLBFGS_DIFFERENTIALEVOLUTION_CUH