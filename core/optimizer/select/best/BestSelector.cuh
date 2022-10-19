//
// Created by spaceman on 2022. 10. 19..
//

#ifndef PARALLELLBFGS_BESTSELECTOR_CUH
#define PARALLELLBFGS_BESTSELECTOR_CUH
#include "../Selector.cuh"

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

class BestSelector : public Selector {
public:
    void select(CUDAConfig& cudaConfig,const double *oldX, double *newX, const double *oldF, double *newF, unsigned generation) {
        selectBestModels<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(oldX,newX,oldF,newF,generation);
    }
};

#endif //PARALLELLBFGS_BESTSELECTOR_CUH
