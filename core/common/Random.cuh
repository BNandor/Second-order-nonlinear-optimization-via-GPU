//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_RANDOM_CUH
#define PARALLELLBFGS_RANDOM_CUH


#include <curand_kernel.h>

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

    void initialize( int size, int blocksPerGrid, int threadsPerBlock) {

#ifdef SAFE
        assert(threadsPerBlock>0);
        assert(blocksPerGrid>0);
        assert(size>0);
        assert(dev_curandState==0);
        assert(size<=(threadsPerBlock * blocksPerGrid));
#endif
        cudaMalloc(&dev_curandState,size * sizeof(curandState) );
        setupCurandState<<<blocksPerGrid, threadsPerBlock>>>(
                dev_curandState, size);
    }
};

#endif //PARALLELLBFGS_RANDOM_CUH
