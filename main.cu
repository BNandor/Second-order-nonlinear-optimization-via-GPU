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
#include "core/optimizer/perturb/DE/DifferentialEvolution.cuh"
#include "core/common/Random.cuh"
#include "core/common/Metrics.cuh"
#include "core/optimizer/perturb/DE/DEContext.h"
#include "core/common/OptimizerContext.cuh"
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
//}

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

void persistBestSNLPModel(double *x, int modelSize, std::string filename) {
    std::ofstream output;
    output.open(filename.c_str());
    if (output.is_open()) {
        for (int i=0;i<modelSize;i++){
            output<<std::setprecision(17)<<x[i]<<std::endl;
        }
        output.close();
    } else {
        std::cout << "err: could not open " << filename << std::endl;
        exit(1);
    }
}

void testOptimizer() {
    Random cudaRandom=Random();
    Metrics metrics = Metrics();
    metrics.initialize();
    DEContext deContext=DEContext();
    OptimizerContext optimizerContext=OptimizerContext(deContext);
    optimizerContext.model=SNLPModel(deContext,ITERATION_COUNT);


//#if defined(PROBLEM_SNLP) || defined(PROBLEM_SNLP3D)
//    const unsigned dataSize = RESIDUAL_CONSTANTS_DIM_1 * RESIDUAL_CONSTANTS_COUNT_1 +
//                              RESIDUAL_CONSTANTS_DIM_2 * RESIDUAL_CONSTANTS_COUNT_2;
//#else
//    const unsigned dataSize = RESIDUAL_CONSTANTS_DIM_1 * RESIDUAL_CONSTANTS_COUNT_1;
//#endif

    optimizerContext.getCurrentLocalSearch()->setupGlobalData(optimizerContext.getModelPopulationSize());

    // ALLOCATE DEVICE MEMORY
    optimizerContext.cudaMemoryModel.allocateFor(optimizerContext.model);
    optimizerContext.cudaMemoryModel.copyModelToDevice(optimizerContext.model);

    // GENERATE PROBLEM
//    double x[optimizerContext.model.modelPopulationSize]={};
    double solution[optimizerContext.model.modelPopulationSize]={};
    double finalFs[optimizerContext.model.populationSize]={};
//    double data[optimizerContext.model.modelPopulationSize]={};

#ifdef PROBLEM_PLANEFITTING
    double A = -5.5;
    double B = 99;
    double C = -1;
    generatePlanePoints(A, B, C, data, RESIDUAL_CONSTANTS_COUNT_1);
    generateInitialPopulation(x, optimizerContext.model.modelPopulationSize);
#endif

#ifdef PROBLEM_ROSENBROCK2D
    data[0] = 1.0;
    data[1] = 100.0;
    x[0] = 100.0;
    x[1] = 2.0;
//    generateInitialPopulation(x, optimizerContext.model.modelPopulationSize);
#endif

    optimizerContext.model.loadModel(optimizerContext.cudaMemoryModel.dev_x,optimizerContext.cudaMemoryModel.dev_data,metrics);

    // COPY TO DEVICE

    metrics.getCudaEventMetrics().recordStartCompute();
    cudaRandom.initialize(optimizerContext.getThreadsInGrid(),optimizerContext.getBlocksPerGrid(),optimizerContext.getThreadsPerBlock());
    // EXECUTE KERNEL
    optimizerContext.cudaMemoryModel.dev_x1 = optimizerContext.cudaMemoryModel.dev_x;
    optimizerContext.cudaMemoryModel.dev_x2 = optimizerContext.cudaMemoryModel.dev_xDE;
    optimizerContext.cudaMemoryModel.dev_F1 = optimizerContext.cudaMemoryModel.dev_F;
    optimizerContext.cudaMemoryModel.dev_F2 = optimizerContext.cudaMemoryModel.dev_FDE;

#if  defined(OPTIMIZER_MIN_INIT_DE) || defined(OPTIMIZER_MIN_DE)
    optimizerContext.getCurrentLocalSearch()->optimize(optimizerContext.cudaMemoryModel.dev_x1, optimizerContext.cudaMemoryModel.dev_data,optimizerContext.cudaMemoryModel.dev_F1, optimizerContext.getCurrentLocalSearch()->getDevGlobalContext(),optimizerContext.cudaMemoryModel.dev_Model,optimizerContext.cudaConfig);
#endif

#ifdef OPTIMIZER_SIMPLE_DE
    OPTIMIZER::evaluateF<<<POPULATION_SIZE, THREADS_PER_BLOCK>>>(dev_x1, dev_data, dev_F1, optimizerContext.getCurrentLocalSearch()->getDevGlobalContext());
#endif

    for (unsigned i = 0; i < optimizerContext.totalIterations; i++) {
        optimizerContext.getCurrentPerturbator()->perturb(optimizerContext.cudaConfig,optimizerContext.cudaMemoryModel.dev_x1,optimizerContext.cudaMemoryModel.dev_x2,&cudaRandom);
        //dev_x2 is the differential model
#if  defined(OPTIMIZER_MIN_INIT_DE) || defined(OPTIMIZER_SIMPLE_DE)
        OPTIMIZER::evaluateF<<<POPULATION_SIZE, THREADS_PER_BLOCK>>>(optimizerContext.cudaMemoryModel.dev_x2, optimizerContext.cudaMemoryModel.dev_data, optimizerContext.cudaMemoryModel.dev_F2, optimizerContext.getCurrentLocalSearch()->getDevGlobalContext());
#elif defined(OPTIMIZER_MIN_DE)
        optimizerContext.getCurrentLocalSearch()->optimize(optimizerContext.cudaMemoryModel.dev_x2, optimizerContext.cudaMemoryModel.dev_data, optimizerContext.cudaMemoryModel.dev_F2, optimizerContext.getCurrentLocalSearch()->getDevGlobalContext(),optimizerContext.cudaMemoryModel.dev_Model,optimizerContext.cudaConfig);
#elif
        std::cerr<<"Incorrect optimizer configuration"<<std::endl;
        exit(1);
#endif

        //evaluated differential model into F2
        //select the best models from current and differential models
        optimizerContext.getCurrentSelector()->select(optimizerContext.cudaConfig,optimizerContext.cudaMemoryModel.dev_x1, optimizerContext.cudaMemoryModel.dev_x2, optimizerContext.cudaMemoryModel.dev_F1, optimizerContext.cudaMemoryModel.dev_F2, i);

        std::swap(optimizerContext.cudaMemoryModel.dev_x1, optimizerContext.cudaMemoryModel.dev_x2);
        std::swap(optimizerContext.cudaMemoryModel.dev_F1, optimizerContext.cudaMemoryModel.dev_F2);
        // dev_x1 contains the next models, dev_F1 contains the associated costs
    }
#if defined(OPTIMIZER_SIMPLE_DE) || defined(OPTIMIZER_MIN_INIT_DE)
        printf("\nthreads:%d\n", THREADS_PER_BLOCK);
        printf("\niterations:%d\n", DE_ITERATION_COUNT);
        printf("\nfevaluations: %d\n", DE_ITERATION_COUNT);
#endif
    printBestF<<<1,1>>>(optimizerContext.cudaMemoryModel.dev_F1,POPULATION_SIZE);

    cudaMemcpy(&finalFs, optimizerContext.cudaMemoryModel.dev_F1, POPULATION_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    int min=0;
    for(int ff=1;ff<POPULATION_SIZE;ff++){
        if(finalFs[min]>finalFs[ff]){
            min=ff;
        }
    }
    cudaMemcpy(&solution, optimizerContext.cudaMemoryModel.dev_x1, optimizerContext.model.modelPopulationSize * sizeof(double), cudaMemcpyDeviceToHost);
    printf("\nsolf: %f and solution: ",finalFs[min]);
    for(int ff=X_DIM*min;ff<X_DIM*(min+1)-1;ff++) {
        printf("%f,",solution[ff]);
    }
    printf("%f\n",solution[X_DIM*(min+1)-1]);
    persistBestSNLPModel(&solution[X_DIM*min],X_DIM, std::string("finalModel")+std::string(OPTIMIZER::name)+std::string(".csv"));

    metrics.getCudaEventMetrics().recordStopCompute();

//    printf("Memcpy,kernel elapsed time (ms): %f,%f\n", memcpyMilli, kernelMilli);
    printf("\ntime ms : %f\n", metrics.getCudaEventMetrics().getElapsedKernelMilliSec());
}

int main() {
    testOptimizer();
    return 0;
}
