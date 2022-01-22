//
// Created by spaceman on 2022. 01. 16..
//

#ifndef PARALLELLBFGS_TESTS_CUH
#define PARALLELLBFGS_TESTS_CUH

#include "../AD/DFloat.cuh"
#include "../AD/DFunction.cuh"
#include "../AD/DPlusFunction.cuh"
#include "../AD/DMultiplicationFunction.cuh"
#include "../AD/DMinusFunction.cuh"
#include "../AD/DIDFunction.cuh"
#include <stdio.h>

__global__ void testDFloatKernel(DFloat *c, unsigned *globalIndex) {
    DFloat a = DFloat(2.0, atomicAdd(globalIndex, 1), globalIndex);
    DFloat b = DFloat(4.0, atomicAdd(globalIndex, 1), globalIndex);
//    *c = ((a * b).square().sqrt() - a / b.inverse()).sin().cos();
    DFloat sum = a + b;
    *c = sum * sum;
    printf("testing \n");
}

__global__ void functionTestsKernel(unsigned *index) {
    unsigned globalIndex = 0;
    DFloat parameters[6] = {DFloat(1.0, globalIndex, &globalIndex),
                            DFloat(2.0, globalIndex, &globalIndex),
                            DFloat(4.0, globalIndex, &globalIndex)};

    DIDFunction idX = DIDFunction(0);
    DIDFunction idY = DIDFunction(1);
    DIDFunction idY2 = DIDFunction(2);
    DMultiplicationFunction x1 = DMultiplicationFunction(&idX, &idY);
    DMultiplicationFunction x2 = DMultiplicationFunction(&x1, &idY2);
    DPlusFunction x3 = DPlusFunction(&x1, &x2);

    DFloat &result = x3(parameters, 6);
    result.setPartialDerivatives(parameters);
    printf("globalindex: %d \n", globalIndex);
    printf("resultderivative: %f \n", result.derivative);

    for (int i = 0; i < 6; i++) {
        printf("%f: \n", parameters[i].derivative);
    }

//    printf("%f \n", parameters[idX.index].derivative);
//    printf("%f \n", parameters[x1.index].derivative);
//    printf("%f \n", parameters[x2.index].derivative);
//    printf("%f \n", parameters[x3.index].derivative);
    assert(parameters[idX.index].derivative == 10.00);
    assert(parameters[x1.index].derivative == 5.00);
    assert(parameters[x2.index].derivative == 1.00);
    assert(parameters[x3.index].derivative == 1.00);

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
}

#endif //PARALLELLBFGS_TESTS_CUH
