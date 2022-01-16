//
// Created by spaceman on 2022. 01. 16..
//

#ifndef PARALLELLBFGS_TESTS_CUH
#define PARALLELLBFGS_TESTS_CUH

#include "../AD/DFloat.cuh"
#include "../AD/DFunction.cuh"
#include <stdio.h>

__global__ void testDFloatKernel(DFloat *c, unsigned *globalIndex) {
    DFloat a = DFloat(2.0, atomicAdd(globalIndex, 1), globalIndex);
    DFloat b = DFloat(4.0, atomicAdd(globalIndex, 1), globalIndex);
//    *c = ((a * b).square().sqrt() - a / b.inverse()).sin().cos();
    DFloat sum = a + b;
    *c = sum * sum;
}

__global__ void functionTestsKernel(unsigned **BFS, unsigned *bfsSize, unsigned *globalIndex) {
//    DFloat a = DFloat(2.0, atomicAdd(globalIndex, 1), globalIndex);
//    DFloat b = DFloat(4.0, atomicAdd(globalIndex, 1), globalIndex);
//    DFloat c = a + b;
//    DFloat d = (c.square() + a).sqrt();
//    DFunction f1 = DFunction();
//    f1.BFS(d);
//
//    for (int i = 0; i < f1.BFSSize; i++) {
//# if __CUDA_ARCH__ >= 200
//        printf("%d \n", f1.BFSOrder[i]);
//#endif
//    }

    DFloat x = DFloat(2.0, atomicAdd(globalIndex, 1), globalIndex);
    DFloat y = DFloat(4.0, atomicAdd(globalIndex, 1), globalIndex);
    DFloat x1 = x * y;
    DFloat y2 = DFloat(4.0, atomicAdd(globalIndex, 1), globalIndex);
    DFloat x2 = x1 * y2;
    DFloat x3 = x1 + x2;
    DFunction f1 = DFunction();
    f1.BFS(x3);

    for (int i = 0; i < f1.BFSSize; i++) {
# if __CUDA_ARCH__ >= 200
        printf("%d \n", f1.parameterList[f1.BFSOrder[i]]->operation);
#endif
    }
    *BFS = f1.BFSOrder;
    *bfsSize = f1.BFSSize;
}

#endif //PARALLELLBFGS_TESTS_CUH
