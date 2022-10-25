#include <iostream>
#include <iomanip>
//#define SAFE
//#define PRINT

//#define PROBLEM_ROSENBROCK2D
//#define PROBLEM_PLANEFITTING
//#define PROBLEM_SNLP
//#define PROBLEM_SNLP3D

//#define GLOBAL_SHARED_MEM

#include "core/common/Constants.cuh"
#include "core/optimizer/refine/LBFGS.cuh"
#include "core/optimizer/refine/GradientDescent.cuh"
#include "core/common/Random.cuh"
#include "core/common/Metrics.cuh"
#include "core/optimizer/perturb/DE/DEContext.h"
#include "core/common/OptimizerContext.cuh"
//#include "core/optimizer/perturb/GA/GeneticAlgorithm.cu"
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <fstream>

//void testDFloat() {
//    DDouble *dev_c;
//    DDouble *c = (DDouble *) malloc(sizeof(DDouble));
//    unsigned *dev_global_id;
//    cudaMalloc((void **) &dev_global_id, sizeof(unsigned));
//    cudaMalloc((void **) &dev_c, sizeof(DDouble));
//
//
//    unsigned global_id_val = 0;
//    cudaMemcpy(dev_global_id, &global_id_val, sizeof(unsigned), cudaMemcpyHostToDevice);
//    testDFloatKernel<<<1, 1>>>(dev_c, dev_global_id);
//
//    cudaMemcpy(c, dev_c, sizeof(DDouble), cudaMemcpyDeviceToHost);
//    assert(c->value == 36);
//    cudaFree(dev_c);
//    cudaFree(dev_global_id);
//    free(c);
//}
//
//void testDFuncBFS() {
//
//
//    unsigned *dev_global_id;
//    cudaMalloc((void **) &dev_global_id, sizeof(unsigned));
//    unsigned global_id_val = 0;
//    cudaMemcpy(dev_global_id, &global_id_val, sizeof(unsigned), cudaMemcpyHostToDevice);
//    functionTestsKernel<<<1, 1>>>(dev_global_id);
//    cudaFree(dev_global_id);
//
//}
//
//void testF1() {
//    unsigned xSize = 2;
//    double *dev_x;
//    cudaMalloc((void **) &dev_x, xSize * sizeof(double));
//    double x[2] = {100.0, 2.0};
//    cudaMemcpy(dev_x, &x, xSize * sizeof(double), cudaMemcpyHostToDevice);
//    testF1DFloat<<<1, 1>>>(dev_x, xSize);
//    cudaFree(dev_x);
//}
//void testQueue() {
//    unsigned xSize = 2;
//    double *dev_x;
//    cudaMalloc((void **) &dev_x, xSize * sizeof(double));
//    double x[2] = {100.0, 2.0};
//    cudaMemcpy(dev_x, &x, xSize * sizeof(double), cudaMemcpyHostToDevice);
//    testQueue<<<1, 1>>>(dev_x);
//    cudaFree(dev_x);
//}
//void testDot() {
//    unsigned xSize = 2;
//    double *dev_x;
//    cudaMalloc((void **) &dev_x, xSize * sizeof(double));
//    double x[2] = {100.0, 2.0};
//    cudaMemcpy(dev_x, &x, xSize * sizeof(double), cudaMemcpyHostToDevice);
//    testDot<<<1, THREADS_PER_BLOCK>>>(dev_x);
//    cudaFree(dev_x);
//}/

 void generateInitialPopulation(double *x, unsigned xSize) {
    std::uniform_real_distribution<double> unif(-10000, 10000);
    std::default_random_engine re(time(NULL));
    for (int i = 0; i < xSize; i++) {
        x[i] = unif(re);
    }
}


void generatePlanePoints(double A, double B, double C, double *data, unsigned pointCount) {
    std::uniform_real_distribution<double> unif(0, 1);
    std::default_random_engine re;
    std::normal_distribution<double> normal(0.0, 1);

    for (int i = 0; i < pointCount; i++) {
        data[i * RESIDUAL_CONSTANTS_DIM_1] = unif(re);
        data[i * RESIDUAL_CONSTANTS_DIM_1 + 1] = unif(re);
        data[i * RESIDUAL_CONSTANTS_DIM_1 + 2] =
                A * data[i * RESIDUAL_CONSTANTS_DIM_1] + B * data[i * RESIDUAL_CONSTANTS_DIM_1 + 1] + C + normal(re);
    }
}

void testOptimizer() {
    Random cudaRandom = Random();

    DEContext deContext = DEContext();
    GAContext gaContext = GAContext();
    OptimizerContext optimizerContext = OptimizerContext(deContext,gaContext);
    optimizerContext.model = SNLPModel(deContext, ITERATION_COUNT);
    Metrics metrics = Metrics(optimizerContext.model);
    optimizerContext.getCurrentLocalSearch()->setupGlobalData(optimizerContext.getModelPopulationSize());

    optimizerContext.cudaMemoryModel.allocateFor(optimizerContext.model);
    optimizerContext.cudaMemoryModel.copyModelToDevice(optimizerContext.model);
    optimizerContext.model.loadModel(optimizerContext.cudaMemoryModel.dev_x, optimizerContext.cudaMemoryModel.dev_data,
                                     metrics);
    metrics.getCudaEventMetrics().recordStartCompute();
    cudaRandom.initialize(optimizerContext.getThreadsInGrid(), optimizerContext.getBlocksPerGrid(),
                          optimizerContext.getThreadsPerBlock());
    // EXECUTE KERNEL
    optimizerContext.cudaMemoryModel.initLoopPointers();

    optimizerContext.getCurrentLocalSearch()->optimize(optimizerContext.cudaMemoryModel.dev_x1, optimizerContext.cudaMemoryModel.dev_data,optimizerContext.cudaMemoryModel.dev_F1, optimizerContext.getCurrentLocalSearch()->getDevGlobalContext(),optimizerContext.cudaMemoryModel.dev_Model,optimizerContext.cudaConfig);

    for (unsigned i = 0; i < optimizerContext.totalIterations; i++) {
        optimizerContext.getCurrentPerturbator()->perturb(optimizerContext.cudaConfig,&optimizerContext.model,
                                                          optimizerContext.cudaMemoryModel.dev_x1,
                                                          optimizerContext.cudaMemoryModel.dev_x2,
                                                          optimizerContext.cudaMemoryModel.dev_F1, &cudaRandom);
        //dev_x2 is the differential model
        optimizerContext.getCurrentLocalSearch()->optimize(optimizerContext.cudaMemoryModel.dev_x2, optimizerContext.cudaMemoryModel.dev_data, optimizerContext.cudaMemoryModel.dev_F2, optimizerContext.getCurrentLocalSearch()->getDevGlobalContext(),optimizerContext.cudaMemoryModel.dev_Model,optimizerContext.cudaConfig);
        //evaluated differential model into F2
        //select the best models from current and differential models
        optimizerContext.getCurrentSelector()->select(optimizerContext.cudaConfig,
                                                      optimizerContext.cudaMemoryModel.dev_x1,
                                                      optimizerContext.cudaMemoryModel.dev_x2,
                                                      optimizerContext.cudaMemoryModel.dev_F1,
                                                      optimizerContext.cudaMemoryModel.dev_F2, i);

        std::swap(optimizerContext.cudaMemoryModel.dev_x1, optimizerContext.cudaMemoryModel.dev_x2);
        std::swap(optimizerContext.cudaMemoryModel.dev_F1, optimizerContext.cudaMemoryModel.dev_F2);
        // dev_x1 contains the next models, dev_F1 contains the associated costs
    }

    metrics.getCudaEventMetrics().recordStopCompute();
    optimizerContext.cudaMemoryModel.copyModelsFromDevice(metrics.modelPerformanceMetrics);
    metrics.modelPerformanceMetrics.printBestModel(&optimizerContext.model);
    metrics.modelPerformanceMetrics.persistBestModelTo(&optimizerContext.model,std::string("finalModel") + std::string(OPTIMIZER::name) + std::string(".csv"));
    printf("\ntime ms : %f\n", metrics.getCudaEventMetrics().getElapsedKernelMilliSec());
}

int main(int argc, char** argv) {
    testOptimizer();
    return 0;
}
