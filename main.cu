#include <iostream>
#include "core/common/Tests.cuh"

void testDFloat() {
    DFloat *dev_c;
    DFloat *c = (DFloat *) malloc(sizeof(DFloat));
    unsigned *dev_global_id;
    cudaMalloc((void **) &dev_global_id, sizeof(unsigned));
    cudaMalloc((void **) &dev_c, sizeof(DFloat));


    unsigned global_id_val = 0;
    cudaMemcpy(dev_global_id, &global_id_val, sizeof(unsigned), cudaMemcpyHostToDevice);
    testDFloatKernel<<<1, 1>>>(dev_c, dev_global_id);

    cudaMemcpy(c, dev_c, sizeof(DFloat), cudaMemcpyDeviceToHost);
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

int main() {
//    testDFloat();
    testDFuncBFS();
    return 0;
}

// Create the Function concept: ([DFloat a])-> compute parameter index order once (BFS), and propagate derivatives that way
// will have: orderArray[operatorTreeSize] - container indices of parameter in order
//            parameters[maxIndex]-contains references of DFloat parameters
// calculate local stack size limit,
// keep a min heap of size operatorTreeSize and a statistical vector to check for duplicates.
// OPT: keep the orderArray in shared memory, to reduce