#include <iostream>

//#define SAFE
//#define PRINT
//#define  OPTIMIZER LBFGS
//#define PROBLEM_ROSENBROCK2D
//#define PROBLEM_PLANEFITTING
//#define PROBLEM_SNLP
//#define PROBLEM_SNLP3D


#include "core/common/Constants.cuh"
#include "core/optimizer/LBFGS.cuh"
#include "core/optimizer/GradientDescent.cuh"
#include "core/optimizer/DifferentialEvolution.cuh"
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
    std::uniform_real_distribution<double> unif(-1000, 1000);
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

#if defined(PROBLEM_SNLP) || defined(PROBLEM_SNLP3D)

void readSNLPProblem(double *data, std::string filename) {
    std::fstream input;
    input.open(filename.c_str());
    if (input.is_open()) {
        unsigned cData = 0;
        while (input >> data[cData]) {
            cData++;
        }
        std::cout << "read: " << cData << " expected: " << RESIDUAL_CONSTANTS_COUNT_1 * RESIDUAL_CONSTANTS_DIM_1
                  << std::endl;
        assert(cData == RESIDUAL_CONSTANTS_COUNT_1 * RESIDUAL_CONSTANTS_DIM_1);
    } else {
        std::cerr << "err: could not open " << filename << std::endl;
        exit(1);
    }
}

void readSNLPAnchors(double *data, std::string filename) {
    std::fstream input;
    input.open(filename.c_str());
    if (input.is_open()) {
        unsigned cData = 0;
        while (input >> data[cData]) {
            cData++;
        }
        std::cout << "read: " << cData << " expected: " << RESIDUAL_CONSTANTS_COUNT_2 * RESIDUAL_CONSTANTS_DIM_2
                  << std::endl;
        assert(cData == RESIDUAL_CONSTANTS_COUNT_2 * RESIDUAL_CONSTANTS_DIM_2);
    } else {
        std::cerr << "err: could not open " << filename << std::endl;
        exit(1);
    }
}

#endif

void testPlaneFitting() {
    curandState *dev_curandState;
    cudaEvent_t start, stop, startCopy, stopCopy;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startCopy);
    cudaEventCreate(&stopCopy);

    const unsigned xSize = X_DIM * POPULATION_SIZE;

#if defined(PROBLEM_SNLP) || defined(PROBLEM_SNLP3D)
    const unsigned dataSize = RESIDUAL_CONSTANTS_DIM_1 * RESIDUAL_CONSTANTS_COUNT_1 +
                              RESIDUAL_CONSTANTS_DIM_2 * RESIDUAL_CONSTANTS_COUNT_2;
#else
    const unsigned dataSize = RESIDUAL_CONSTANTS_DIM_1 * RESIDUAL_CONSTANTS_COUNT_1;
#endif
    double *dev_x;
    double *dev_xDE;
    double *dev_x1;
    double *dev_x2;
    double *dev_data;
    double *dev_F;
    double *dev_FDE;
    double *dev_F1;
    double *dev_F2;


    // ALLOCATE DEVICE MEMORY
    cudaMalloc((void **) &dev_x, xSize * sizeof(double));
    cudaMalloc((void **) &dev_xDE, xSize * sizeof(double));
    cudaMalloc((void **) &dev_data, dataSize * sizeof(double));
    cudaMalloc((void **) &dev_F, POPULATION_SIZE * sizeof(double));
    cudaMalloc((void **) &dev_FDE, POPULATION_SIZE * sizeof(double));
    cudaMalloc(&dev_curandState, THREADS_PER_GRID * sizeof(curandState));

    // GENERATE PROBLEM
    double x[xSize] = {};
    double data[dataSize] = {};

#ifdef PROBLEM_PLANEFITTING
    double A = -5.5;
    double B = 99;
    double C = -1;
    generatePlanePoints(A, B, C, data, RESIDUAL_CONSTANTS_COUNT_1);
    generateInitialPopulation(x, xSize);
#endif

#ifdef PROBLEM_ROSENBROCK2D
    data[0] = 1.0;
    data[1] = 100.0;
    x[0] = 100.0;
    x[1] = 2.0;
//    generateInitialPopulation(x, xSize);
#endif

#if defined(PROBLEM_SNLP) || defined(PROBLEM_SNLP3D)
    readSNLPProblem(data, PROBLEM_PATH);

    readSNLPAnchors(data + RESIDUAL_CONSTANTS_DIM_1 * RESIDUAL_CONSTANTS_COUNT_1,
                    PROBLEM_ANCHOR_PATH);
    generateInitialPopulation(x, xSize);
#endif
    // COPY TO DEVICE
    cudaEventRecord(startCopy);
    cudaMemcpy(dev_x, &x, xSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data, &data, dataSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(stopCopy);
    cudaEventRecord(start);

    // EXECUTE KERNEL
    // initialize curand
    setupCurand<<<POPULATION_SIZE, THREADS_PER_BLOCK>>>(dev_curandState);
    dev_x1 = dev_x;
    dev_x2 = dev_xDE;
    dev_F1 = dev_F;
    dev_F2 = dev_FDE;

    OPTIMIZER::gradientDescent<<<POPULATION_SIZE, THREADS_PER_BLOCK>>>(dev_x1, dev_data, dev_F1);

    for (unsigned i = 0; i < DE_ITERATION_COUNT; i++) {
        differentialEvolutionStep<<<POPULATION_SIZE, THREADS_PER_BLOCK>>>(dev_x1, dev_x2, dev_curandState);
        //dev_x2 is the differential model
        OPTIMIZER::gradientDescent<<<POPULATION_SIZE, THREADS_PER_BLOCK>>>(dev_x2, dev_data, dev_F2);
        //evaluated differential model into F2
        selectBestModels<<<POPULATION_SIZE, THREADS_PER_BLOCK>>>(dev_x1, dev_x2, dev_F1, dev_F2, i);
        //select the best models from current and differential models
        std::swap(dev_x1, dev_x2);
        std::swap(dev_F1, dev_F2);
        // dev_x1 contains the next models, dev_F1 contains the associated costs
    }

    //dev_x2 contains the best models
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float memcpyMilli = 0;
    cudaEventElapsedTime(&memcpyMilli, startCopy, stopCopy);
    float kernelMilli = 0;
    cudaEventElapsedTime(&kernelMilli, start, stop);
    printf("Memcpy,kernel elapsed time (ms): %f,%f\n", memcpyMilli, kernelMilli);

    cudaFree(dev_x);
    cudaFree(dev_xDE);
    cudaFree(dev_data);
    cudaFree(dev_F);
    cudaFree(dev_FDE);
}

int main() {
//    testDFloat();
//    testDFuncBFS();
//    testF1();

    testPlaneFitting();
//    testQueue();
//    testDot();
    return 0;
}

// Create the Function concept: ([DDouble a])-> compute parameter index order once (BFS), and propagate derivatives that way
// will have: orderArray[operatorTreeSize] - container indices of parameter in order
//            parameters[maxIndex]-contains references of DDouble parameters
// calculate local stack size limit,
// keep a min heap of size operatorTreeSize and a statistical vector to check for duplicates.
// OPT: keep the orderArray in shared memory, to reduce