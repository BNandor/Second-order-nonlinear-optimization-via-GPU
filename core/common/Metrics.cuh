//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_METRICS_CUH
#define PARALLELLBFGS_METRICS_CUH

class CudaEventMetrics {
private:
    cudaEvent_t start, stop, startCopy, stopCopy;
public:
    void initialize() {
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

    float getElapsedCopyMilliSec(){
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
    CudaEventMetrics cudaEventMetrics=CudaEventMetrics();

public:
    void initialize() {
        cudaEventMetrics.initialize();
    }
    CudaEventMetrics& getCudaEventMetrics(){
        return cudaEventMetrics;
    }
};

#endif //PARALLELLBFGS_METRICS_CUH