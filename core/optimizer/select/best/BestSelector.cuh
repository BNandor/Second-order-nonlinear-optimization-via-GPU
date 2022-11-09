//
// Created by spaceman on 2022. 10. 19..
//

#ifndef PARALLELLBFGS_BESTSELECTOR_CUH
#define PARALLELLBFGS_BESTSELECTOR_CUH
#include "../Selector.cuh"

__global__
void printGenerationalCosts( double *newF, unsigned generation) {
    if (threadIdx.x == 0) {
        printf("Gen %u block %d f: %.10f\n", generation, blockIdx.x, newF[blockIdx.x]);
    }
}

__global__
void selectBestModels(const double *oldX, double *newX, const double *oldF, double *newF) {
    if (oldF[blockIdx.x] < newF[blockIdx.x]) {
        for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
            newX[blockIdx.x * X_DIM + spanningTID] = oldX[blockIdx.x * X_DIM + spanningTID];
        }
        if (threadIdx.x == 0) {
            newF[blockIdx.x] = oldF[blockIdx.x];
        }
    }
}

class BestSelector : public Selector {
public:

    BestSelector() {
        std::unordered_map<std::string,BoundedParameter> bestSelectorParams=std::unordered_map<std::string,BoundedParameter>();
        parameters=OperatorParameters(bestSelectorParams);
    }

    void operate(CUDAMemoryModel* cudaMemoryModel) override {
        select(cudaMemoryModel->cudaConfig,
               cudaMemoryModel->dev_x1,
               cudaMemoryModel->dev_x2,
               cudaMemoryModel->dev_F1,
               cudaMemoryModel->dev_F2);
    }

    void select(CUDAConfig& cudaConfig,const double *oldX, double *newX, const double *oldF, double *newF) {
        selectBestModels<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(oldX,newX,oldF,newF);
    }

    void printPopulationCostAtGeneration(CUDAConfig& cudaConfig,double *newF, unsigned generation) {
        printGenerationalCosts<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(newF,generation);
    }

    int fEvaluationCount() {
        return 0;
    }
};

#endif //PARALLELLBFGS_BESTSELECTOR_CUH
