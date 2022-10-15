//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_RANDOM_CUH
#define PARALLELLBFGS_RANDOM_CUH

#include "OptimizerContext.cuh"

__global__
void setupCurandState(curandState *state, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx<size) {
        curand_init(1234, idx, 0, &state[idx]);
    }
}

class Random {
public:
    curandState *dev_curandState;

    virtual ~Random() {
        if(dev_curandState!= nullptr){
            cudaFree(dev_curandState);
        }
    }

    void initialize( int size, OptimizerContext &optimizerContext) {

#ifdef SAFE
        assert(optimizerContext.getThreadsPerBlock()>0);
        assert(optimizerContext.getBlocksPerGrid()>0);
        assert(size>0);
        assert(dev_curandState==0);
        assert(size<=(optimizerContext.getBlocksPerGrid() * optimizerContext.getThreadsPerBlock()));
#endif
        cudaMalloc(&dev_curandState,size * sizeof(curandState) );
        setupCurandState<<<optimizerContext.getBlocksPerGrid(), optimizerContext.getThreadsPerBlock()>>>(
                dev_curandState, size);
    }
};

#endif //PARALLELLBFGS_RANDOM_CUH
