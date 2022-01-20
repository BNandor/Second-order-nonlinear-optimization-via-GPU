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
//        printf("%d \n", f1.BFSOrder[i]);
//    }
//
//    DFloat x = DFloat(1.0, atomicAdd(globalIndex, 1), globalIndex);
//    DFloat y = DFloat(2.0, atomicAdd(globalIndex, 1), globalIndex);
//    DFloat x1 = x * y;
//    DFloat y2 = DFloat(4.0, atomicAdd(globalIndex, 1), globalIndex);
//    DFloat x2 = x1 * y2;
//    DFloat x3 = x1 + x2;
//    DFunction f1 = DFunction();
//
//    for (int i = 0; i < f1.BFSSize; i++) {
//        printf("%d \n", f1.parameterList[f1.BFSOrder[i]]->operation);
//    }
//    f1.setPartialDerivatives(x3);
//    printf("Derivatives \n");
//
////    for (int i = 0; i < f1.BFSSize; i++) {
////        printf("%d:%f \n", f1.parameterList[f1.BFSOrder[i]]->index, f1.parameterList[f1.BFSOrder[i]]->derivative);
////    }
//    printf("%f \n", x.derivative);
//    printf("%f \n", x1.derivative);
//    printf("%f \n", x2.derivative);
//    printf("%f \n", x3.derivative);
//    assert(x.derivative == 10.00);
//    assert(x1.derivative == 5.00);
//    assert(x2.derivative == 1.00);
//    assert(x3.derivative == 1.00);

//    DFloat x = DFloat(M_PI, atomicAdd(globalIndex, 1), globalIndex);
//    DFloat sinX = x.sin();
//    DFunction f1 = DFunction();
//    f1.setPartialDerivatives(sinX);
//    assert(x.derivative == -1.0);
//    printf("%f \n", x.derivative);


//    DFloat x = DFloat(2.0, atomicAdd(globalIndex, 1), globalIndex);
//    DFloat inv = x.inverse();
//    DFunction f1 = DFunction();
//    f1.setPartialDerivatives(inv);
//    assert(x.derivative == -0.25);
//    printf("%f \n", x.derivative);

    *BFS = f1.BFSOrder;
    *bfsSize = f1.BFSSize;
}

#endif //PARALLELLBFGS_TESTS_CUH
