//
// Created by spaceman on 2022. 10. 17..
//

#ifndef PARALLELLBFGS_CUDACONFIG_CUH
#define PARALLELLBFGS_CUDACONFIG_CUH


struct CUDAConfig {
    int threadsPerBlock=THREADS_PER_BLOCK;
    int blocksPerGrid;
    CUDAConfig()=default;
    CUDAConfig(Perturbator& perturbator) {
        blocksPerGrid=perturbator.populationSize;
    }
};


#endif //PARALLELLBFGS_CUDACONFIG_CUH