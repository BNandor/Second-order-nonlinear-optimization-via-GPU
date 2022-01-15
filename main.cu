#include <iostream>
#include "core/AD/DDouble.cuh"

int main() {
    DFloat *dev_c;
    DFloat *c = (DFloat *) malloc(sizeof(DFloat));
    unsigned *dev_global_id;
    cudaMalloc((void **) &dev_c, sizeof(DFloat));
    cudaMalloc((void **) &dev_global_id, sizeof(unsigned));

    unsigned global_id_val = 0;
    cudaMemcpy(dev_global_id, &global_id_val, sizeof(unsigned), cudaMemcpyHostToDevice);
    test<<<1, 1>>>(dev_c, dev_global_id);

    cudaMemcpy(c, dev_c, sizeof(DFloat), cudaMemcpyDeviceToHost);
    cudaFree(dev_c);
    cudaFree(dev_global_id);
    free(c);
    return 0;
}
