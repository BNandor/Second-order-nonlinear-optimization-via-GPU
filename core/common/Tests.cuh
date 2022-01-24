//
// Created by spaceman on 2022. 01. 16..
//

#ifndef PARALLELLBFGS_TESTS_CUH
#define PARALLELLBFGS_TESTS_CUH

#include "../AD/DDouble.cuh"
#include "../AD/function/DFunction.cuh"
#include "../AD/function/DPlusFunction.cuh"
#include "../AD/function/DMultiplicationFunction.cuh"
#include "../AD/function/DMinusFunction.cuh"
#include "../AD/function/DIDFunction.cuh"
#include "../AD/function/DSquareFunction.cuh"
#include "../AD/function/Operations.cuh"
#include "../problem/F1.cuh"
#include <stdio.h>

__global__ void testDFloatKernel(DDouble *c, unsigned *globalIndex) {
    DDouble a = DDouble(2.0, atomicAdd(globalIndex, 1), globalIndex);
    DDouble b = DDouble(4.0, atomicAdd(globalIndex, 1), globalIndex);
//    *c = ((a * b).square().sqrt() - a / b.inverse()).sin().cos();
    DDouble sum = a + b;
    *c = sum * sum;
    printf("testing \n");
}

__global__ void functionTestsKernel(unsigned *index) {
    unsigned globalIndex = 0;
//    DIDFunction idX = 0;
//    DIDFunction idY = 1;
//    DIDFunction idY2 = 2;
//    DMultiplicationFunction x1 = idX * idY;
//    DMultiplicationFunction x2 = x1 * idY2;
//    DPlusFunction x3 = x1 + x2;
//
//
//    DDouble parameters[6] = {DDouble(1.0, globalIndex, &globalIndex),
//                            DDouble(2.0, globalIndex, &globalIndex),
//                            DDouble(4.0, globalIndex, &globalIndex)};
//
//    DDouble *result = x3(parameters, 6);
//    result->setPartialDerivatives(parameters);
//    printf("globalindex: %d \n", globalIndex);
//    printf("resultderivative: %f \n", result->derivative);
//
//    for (auto &parameter : parameters) {
//        printf("%f: \n", parameter.derivative);
//    }
//
//    printf("%f \n", parameters[idX.index].derivative);
//    printf("%f \n", parameters[x1.index].derivative);
//    printf("%f \n", parameters[x2.index].derivative);
//    printf("%f \n", parameters[x3.index].derivative);
//    assert(parameters[idX.index].derivative == 10.00);
//    assert(parameters[x1.index].derivative == 5.00);
//    assert(parameters[x2.index].derivative == 1.00);
//    assert(parameters[x3.index].derivative == 1.00);

//    DIDFunction idX = 0;
//    DIDFunction two = 1;
//    DSquareFunction xsquare = DSquareFunction(&idX);
//    DMultiplicationFunction x1 = two * xsquare;
//
//    DDouble parameters[4] = {DDouble(1.0, globalIndex, &globalIndex),
//                            DDouble(2.0, globalIndex, &globalIndex, CONST)};
//
//    DDouble *result = x1(parameters, 4);
//    result->setPartialDerivatives(parameters);
//    printf("globalindex: %d \n", globalIndex);
//    printf("dx %f \n", parameters[idX.index].derivative);
//
//    for (auto &parameter : parameters) {
//        printf("%f: \n", parameter.derivative);
//    }
//    DDouble x = DDouble(M_PI, atomicAdd(globalIndex, 1), globalIndex);
//    DDouble sinX = x.sin();
//    DFunction f1 = DFunction();
//    f1.setPartialDerivatives(sinX);
//    assert(x.derivative == -1.0);
//    printf("%f \n", x.derivative);


//    DDouble x = DDouble(2.0, atomicAdd(globalIndex, 1), globalIndex);
//    DDouble inv = x.inverse();
//    DFunction f1 = DFunction();
//    f1.setPartialDerivatives(inv);
//    assert(x.derivative == -0.25);
//    printf("%f \n", x.derivative);
}

//__global__ void testF1Kernel(unsigned *params) {
//    unsigned globalIndex = 0;
//    DDouble operatorTree[14] = {DDouble(100.0, globalIndex, &globalIndex),
//                             DDouble(2.0, globalIndex, &globalIndex),
//                             DDouble(100.0, globalIndex, &globalIndex, CONST),
//                             DDouble(1.0, globalIndex, &globalIndex, CONST),
//                             DDouble(-1.0, globalIndex, &globalIndex, CONST)};
//    DIDFunction x0 = 0;
//    DIDFunction x1 = 1;
//    DIDFunction hundred = 2;
//    DIDFunction one = 3;
//    DIDFunction minOne = 4;
//    DSquareFunction x0square = DSquareFunction(&x0);
//    DMinusFunction t1 = x1 - x0square;
//    DSquareFunction t1square = DSquareFunction(&t1);
//    DMultiplicationFunction th = hundred * t1square;
//    DDouble *f2 = th(operatorTree, 13);
//    DMultiplicationFunction t2 = x0 * minOne;
//    DPlusFunction t3 = t2 + one;
//    DSquareFunction t4 = DSquareFunction(&t3);
//    DDouble *f1 = t4(operatorTree, 13);
//    DPlusFunction t5 = th + t4;
//    DDouble s = *f1 + *f2;
//    printf("\nt4 rv: %u\n", t4.resultValue);
//    printf("\nth rv: %u\n", th.resultValue);
////    DDouble *f3 = t5(operatorTree, 13);
//
//    printf("f3: %u\n", s.index);
//    printf("f2: %f\n", f2->value);
////    printf("f1: %f\n", f1->value);
//
////    result->setPartialDerivatives(operatorTree);
////    printf("globalindex: %d \n", globalIndex);
////    printf("result: %f", result->value);
////    for (auto &parameter : operatorTree) {
////        printf("%f: \n", parameter.derivative);
////    }
//}


__global__ void testF1DFloat(double *x, unsigned xSize) {
    F1 f1 = F1();

    DDouble *t5 = f1.costFunction(x, xSize, threadIdx.x);
    printf("result: %f\n", t5->value);
    f1.setJacobian();

    printf("Jacobian:\n");
    for (double i : f1.J) {
        printf("%f\n", i);
    }
    printf("derivative: %f\n", f1.operatorTree[3].derivative);
}

#endif //PARALLELLBFGS_TESTS_CUH
