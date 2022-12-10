//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_RANDOM_CUH
#define PARALLELLBFGS_RANDOM_CUH

#include <assert.h>
#include <curand_kernel.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
__global__
void setupCurandState(curandState *state, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx<size) {
        curand_init(986096, idx, 0, &state[idx]);
    }
}

class Random {
public:
    curandState *dev_curandState;

    virtual ~Random() {
        if(dev_curandState!= nullptr){
            gpuErrchk(cudaFree(dev_curandState));
            dev_curandState=0;
        }
    }

    void initialize( int size, int blocksPerGrid, int threadsPerBlock) {

#ifdef SAFE
        assert(threadsPerBlock>0);
        assert(blocksPerGrid>0);
        assert(size>0);
        assert(size<=(threadsPerBlock * blocksPerGrid));
#endif
        if(dev_curandState!= nullptr){
            gpuErrchk(cudaFree(dev_curandState));
            dev_curandState=0;
        }
        gpuErrchk(cudaMalloc(&dev_curandState,size * sizeof(curandState) ));
        setupCurandState<<<blocksPerGrid, threadsPerBlock>>>(
                dev_curandState, size);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
};

#endif //PARALLELLBFGS_RANDOM_CUH
