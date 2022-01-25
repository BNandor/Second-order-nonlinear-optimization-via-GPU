#include <iostream>
#include "core/common/Tests.cuh"

void testDFloat() {
    DDouble *dev_c;
    DDouble *c = (DDouble *) malloc(sizeof(DDouble));
    unsigned *dev_global_id;
    cudaMalloc((void **) &dev_global_id, sizeof(unsigned));
    cudaMalloc((void **) &dev_c, sizeof(DDouble));


    unsigned global_id_val = 0;
    cudaMemcpy(dev_global_id, &global_id_val, sizeof(unsigned), cudaMemcpyHostToDevice);
    testDFloatKernel<<<1, 1>>>(dev_c, dev_global_id);

    cudaMemcpy(c, dev_c, sizeof(DDouble), cudaMemcpyDeviceToHost);
    assert(c->value == 36);
    cudaFree(dev_c);
    cudaFree(dev_global_id);
    free(c);
}

void testDFuncBFS() {


    unsigned *dev_global_id;
    cudaMalloc((void **) &dev_global_id, sizeof(unsigned));
    unsigned global_id_val = 0;
    cudaMemcpy(dev_global_id, &global_id_val, sizeof(unsigned), cudaMemcpyHostToDevice);
    functionTestsKernel<<<1, 1>>>(dev_global_id);
    cudaFree(dev_global_id);

}

void testF1() {
    unsigned xSize = 2;
    double *dev_x;
    cudaMalloc((void **) &dev_x, xSize * sizeof(double));
    double x[2] = {100.0, 2.0};
    cudaMemcpy(dev_x, &x, xSize * sizeof(double), cudaMemcpyHostToDevice);
    testF1DFloat<<<1, 1>>>(dev_x, xSize);
    cudaFree(dev_x);
}

void testPlaneFitting() {
    const unsigned xSize = 3;
    const unsigned dataSize = 6;
    double *dev_x;
    double *dev_dx;
    double *dev_F;
    double *dev_data;
    cudaMalloc((void **) &dev_x, xSize * sizeof(double));
    cudaMalloc((void **) &dev_dx, xSize * sizeof(double));
    cudaMalloc((void **) &dev_F, sizeof(double));
    cudaMalloc((void **) &dev_data, dataSize * sizeof(double));
    double x[xSize] = {5.5, 99.0, -1.0};
    double data[dataSize] = {1.0, 1.0, 1.0, 2.0, 0.0, 0.0};
    cudaMemcpy(dev_x, &x, xSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data, &data, dataSize * sizeof(double), cudaMemcpyHostToDevice);
    testPlaneFitting<<<1, 2>>>(dev_x, dev_dx, dev_F, dev_data);
    cudaFree(dev_x);
    cudaFree(dev_dx);
    cudaFree(dev_F);
    cudaFree(dev_data);
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