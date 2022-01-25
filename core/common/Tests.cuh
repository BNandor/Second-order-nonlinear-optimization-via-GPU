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
#include "../problem/PlaneFitting.cuh"
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

__global__ void testF1DFloat(double *globalX, unsigned globalXSize) {
    unsigned iterationCount = 30000;
    double constants[3] = {100.0, 1.0, -1.0};
    F1 f1 = F1(constants, 3);

    const unsigned xSize = 2;
    double x1[xSize] = {globalX[2 * threadIdx.x], globalX[2 * threadIdx.x + 1]};
    double x2[xSize];
    double *x = x1;
    double *xNext = x2;
    double alpha = 100;
    double *tmp;
    for (unsigned i = 0; i < iterationCount; i++) {
        double f = f1.eval(x, xSize)->value;
//        printf("f %d: %f\n", i, f);
        f1.evalJacobian();
        alpha = 100;
        f1.evalStep(x, xNext, xSize, f1.J, alpha);
        double fNext = f1.eval(xNext, xSize)->value;
        while (fNext >= f) {
            alpha = alpha / 2;
            f1.evalStep(x, xNext, xSize, f1.J, alpha);
            fNext = f1.eval(xNext, xSize)->value;
//            printf("alpha: %.10f\n", alpha);
//            printf("fNext: %f.10\n", fNext);
        }
        tmp = x;
        x = xNext;
        xNext = tmp;
    }
    printf("x ");
    for (unsigned j = 0; j < xSize; j++) {
        printf("%f ", x[j]);
    }
    printf("\n");
//    printf("Jacobian:\n");
//    for (double i : f1.J) {
//        printf("%f\n", i);
//    }
//    printf("derivative: %f\n", f1.operatorTree[3].derivative);
}

__device__ void evalF() {

}

__global__ void testPlaneFitting(double *globalX, double *globalDX, double *globalF,
                                 double *data) { // use shared memory instead of global memory
    unsigned iterationCount = 100;
    const unsigned xSize = 3;
    const unsigned constantsSize = 3;
    double constants[constantsSize] = {data[3 * threadIdx.x], data[3 * threadIdx.x + 1], data[3 * threadIdx.x + 2]};

    PlaneFitting f1 = PlaneFitting(constants, constantsSize);
    double x1[xSize] = {globalX[0], globalX[1], globalX[2]};
    double J[xSize] = {};
    double x2[xSize];
    double *x = x1;
    double *xNext = x2;
    double alpha = 100;
    double *tmp;
    for (unsigned i = 0; i < iterationCount; i++) {
        // reset state
        *globalF = 0.0;
        if (threadIdx.x == 0) {
            for (unsigned j = 0; j < xSize; j++) {
                globalDX[j] = 0.0;
            }
        }
        __syncthreads(); // globalF, globalDX cleared // TODO this synchronizes over threads in a block, sync within grid required : https://on-demand.gputechconf.com/gtc/2017/presentation/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf

        atomicAdd(globalF, f1.eval(x, xSize)->value); // reduce over threads, not using atomicAdd

        f1.evalJacobian();
        for (unsigned j = 0; j < xSize; j++) {
            atomicAdd(&globalDX[j], f1.J[j]);// TODO add jacobian variable indexing
        }

        __syncthreads();// globalF, globalDX is complete for all threads
        double f = *globalF;
        printf("tid: %d f %d: %f\n", threadIdx.x, i, f);
        if (threadIdx.x == 0) {
            for (unsigned j = 0; j < xSize; j++) {
                J[j] = globalDX[j];// TODO add Jacobian variable indexing
            }

            alpha = 100;

            f1.evalStep(x, xNext, xSize, J, alpha);
            double fNext = f1.eval(xNext, xSize)->value;
            while (fNext > f) {
                alpha = alpha / 2;
                f1.evalStep(x, xNext, xSize, J, alpha);
                fNext = f1.eval(xNext, xSize)->value;
//            printf("alpha: %.10f\n", alpha);
//            printf("fNext: %.10f\n", fNext);
//            printf("fPrev: %.10f\n", f);
            }
            for (unsigned j = 0; j < xSize; j++) {
                globalX[j] = xNext[j];
            }
        }
        __syncthreads();//globalX is xNext for all threads
        for (unsigned j = 0; j < xSize; j++) {
            xNext[j] = globalX[j];
        }
        tmp = x;
        x = xNext;
        xNext = tmp;
        __syncthreads();//x,xNext is set for all threads
    }
    printf("x ");
    for (unsigned j = 0; j < xSize; j++) {
        printf("%f ", x[j]);
    }
    printf("\n");
//    printf("Jacobian:\n");
//    for (double i : f1.J) {
//        printf("%f\n", i);
//    }
//    printf("derivative: %f\n", f1.operatorTree[3].derivative);
}

#endif //PARALLELLBFGS_TESTS_CUH
