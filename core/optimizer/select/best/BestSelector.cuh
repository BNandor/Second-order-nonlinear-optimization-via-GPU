//
// Created by spaceman on 2022. 10. 19..
//

#ifndef PARALLELLBFGS_BESTSELECTOR_CUH
#define PARALLELLBFGS_BESTSELECTOR_CUH
#include "../Selector.cuh"

__global__
void printFirstModel(double *xCurrent) {
#ifdef PRINT
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("xCurrent ");
        for (unsigned j = 0; j < X_DIM - 1; j++) {
            printf("%f,", xCurrent[j]);
        }
        printf("%f\n", xCurrent[X_DIM - 1]);
    }
#endif
}

__global__
void printGenerationalCosts( double *newF, unsigned generation) {
#ifdef PRINT
    if (threadIdx.x == 0) {
        printf("f: %f gen %u block %u \n",newF[blockIdx.x], generation, blockIdx.x);
    }
#endif
}

__global__
void selectBestModels(const double *oldX, double *newX, const double *oldF, double *newF) {
    if (oldF[blockIdx.x] < newF[blockIdx.x]) {
        for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
            newX[blockIdx.x * X_DIM + spanningTID] = oldX[blockIdx.x * X_DIM + spanningTID];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            newF[blockIdx.x] = oldF[blockIdx.x];
        }
    }
}

class BestSelector : public Selector {
    unsigned selections=0;
public:

    BestSelector() {
        std::unordered_map<std::string,BoundedParameter> bestSelectorParams=std::unordered_map<std::string,BoundedParameter>();
        parameters=OperatorParameters(bestSelectorParams);
    }

    void operate(CUDAMemoryModel* cudaMemoryModel) override {
        select(cudaMemoryModel->cudaConfig,
               cudaMemoryModel->dev_x2,
               cudaMemoryModel->dev_x1,
               cudaMemoryModel->dev_F2,
               cudaMemoryModel->dev_F1);
        printPopulationCostAtGeneration(cudaMemoryModel->cudaConfig,cudaMemoryModel->dev_F1,++selections);
    }

    void select(CUDAConfig& cudaConfig,const double *oldX, double *newX, const double *oldF, double *newF) {
        selectBestModels<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(oldX,newX,oldF,newF);
    }

    void printPopulationCostAtGeneration(CUDAConfig& cudaConfig,double *newF, unsigned generation) {
        printGenerationalCosts<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(newF,generation);
    }

    void printPopulationCostAtGeneration(CUDAConfig& cudaConfig,double *xCurrent) {
        printFirstModel<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(xCurrent);
    }

    int fEvaluationCount() {
        return 0;
    }
};

#endif //PARALLELLBFGS_BESTSELECTOR_CUH
