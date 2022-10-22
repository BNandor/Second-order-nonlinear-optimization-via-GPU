//
// Created by spaceman on 2022. 10. 19..
//

#ifndef PARALLELLBFGS_SELECTOR_CUH
#define PARALLELLBFGS_SELECTOR_CUH

class Selector {
public:
    virtual void select(CUDAConfig& cudaConfig,const double *oldX, double *newX, const double *oldF, double *newF, unsigned generation)=0;
};

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

#endif //PARALLELLBFGS_SELECTOR_CUH
