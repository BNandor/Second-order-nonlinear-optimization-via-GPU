//
// Created by spaceman on 2022. 10. 17..
//

#ifndef PARALLELLBFGS_CUDACONFIG_CUH
#define PARALLELLBFGS_CUDACONFIG_CUH


#include "../../optimizer/perturb/Perturbator.h"

struct CUDAConfig {
    int threadsPerBlock=THREADS_PER_BLOCK;
    int blocksPerGrid;
    CUDAConfig()=default;
    CUDAConfig(int populationSize) {
        blocksPerGrid=populationSize;
    }
};


#endif //PARALLELLBFGS_CUDACONFIG_CUH
