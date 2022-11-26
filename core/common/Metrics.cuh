//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_METRICS_CUH
#define PARALLELLBFGS_METRICS_CUH

#include "ModelMetrics.cuh"
#include "./model/Model.cuh"

class CudaEventMetrics {
private:
    cudaEvent_t start, stop, startCopy, stopCopy;

public:
    void initializeEvents() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventCreate(&startCopy);
        cudaEventCreate(&stopCopy);
    };

    void recordStartCopy(){
        cudaEventRecord(startCopy);
    }

    void recordStopCopy(){
        cudaEventRecord(stopCopy);
    }

    void recordStartCompute(){
        cudaEventRecord(start);
    }

    void recordStopCompute() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    }

    float getElapsedCopyMilliSec() {
        float memcpyMilli = 0;
        cudaEventElapsedTime(&memcpyMilli, startCopy, stopCopy);
        return memcpyMilli;
    }

    float getElapsedKernelMilliSec() {
        float kernelMilli = 0;
        cudaEventElapsedTime(&kernelMilli, start, stop);
        return kernelMilli;
    }
};

class Metrics {
public:
    CudaEventMetrics cudaEventMetrics;
    ModelMetrics modelPerformanceMetrics;
    CUDAConfig *cudaConfig;

    Metrics(){

    }

    Metrics(Model& model,CUDAConfig *cudaConfig): cudaEventMetrics(CudaEventMetrics()),modelPerformanceMetrics(ModelMetrics(model.modelPopulationSize,model.populationSize)){
        cudaEventMetrics.initializeEvents();
        this->cudaConfig=cudaConfig;
    }

    CudaEventMetrics& getCudaEventMetrics() {
        return cudaEventMetrics;
    }

    void printFinalMetrics() {
        printf("printing metrics\n");
        printf("\ntime ms : %f\n", cudaEventMetrics.getElapsedKernelMilliSec());
        printf("\nthreads:%d", cudaConfig->threadsPerBlock);
        printf("\niterations:%d", modelPerformanceMetrics.markovIterations);
        printf("\nfevaluations: %d\n", modelPerformanceMetrics.fEvaluations);
    }
};

#endif //PARALLELLBFGS_METRICS_CUH