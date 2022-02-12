#include <iostream>

#define SAFE

#include "core/common/Tests.cuh"
#include <random>

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

void generatePlanePoints(double A, double B, double C, double *data, unsigned pointCount) {
    std::uniform_real_distribution<double> unif(0, 1);
    std::default_random_engine re;
    std::normal_distribution<double> normal(0.0, 1);

    for (int i = 0; i < pointCount; i++) {
        data[i * OBSERVARVATION_DIM] = unif(re);
        data[i * OBSERVARVATION_DIM + 1] = unif(re);
        data[i * OBSERVARVATION_DIM + 2] =
                A * data[i * OBSERVARVATION_DIM] + B * data[i * OBSERVARVATION_DIM + 1] + C + normal(re);
    }
}

void testPlaneFitting() {
    const unsigned xSize = X_DIM;
    const unsigned dataSize = OBSERVARVATION_DIM * OBSERVARVATION_COUNT;
    double *dev_x;
    cudaMalloc((void **) &dev_x, xSize * sizeof(double));
    double x[xSize] = {1, 1, 1.0};
    double A = 5.5;
    double B = 99;
    double C = -1;
    double data[dataSize] = {};
    generatePlanePoints(A, B, C, data, OBSERVARVATION_COUNT);
    cudaMemcpy(dev_x, &x, xSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dev_const_observations, &data, (OBSERVARVATION_COUNT * OBSERVARVATION_DIM) * sizeof(double), 0,
                       cudaMemcpyHostToDevice);
    testPlaneFitting<<<1, 128>>>(dev_x);
    cudaFree(dev_x);
}

int main() {
//    testDFloat();
//    testDFuncBFS();
//    testF1();
    testPlaneFitting();
    return 0;
}

// Create the Function concept: ([DDouble a])-> compute parameter index order once (BFS), and propagate derivatives that way
// will have: orderArray[operatorTreeSize] - container indices of parameter in order
//            parameters[maxIndex]-contains references of DDouble parameters
// calculate local stack size limit,
// keep a min heap of size operatorTreeSize and a statistical vector to check for duplicates.
// OPT: keep the orderArray in shared memory, to reduce