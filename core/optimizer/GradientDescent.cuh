//
// Created by spaceman on 2022. 01. 16..
//

#ifndef PARALLELLBFGS_GRADIENTDESCENT_CUH
#define PARALLELLBFGS_GRADIENTDESCENT_CUH

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
#include "../problem/Rosenbrock2D.cuh"
#include "../problem/SNLP.cuh"

#include <stdio.h>

//__global__ void testDFloatKernel(DDouble *c, unsigned *globalIndex) {
//    DDouble a = DDouble(2.0, atomicAdd(globalIndex, 1), globalIndex);
//    DDouble b = DDouble(4.0, atomicAdd(globalIndex, 1), globalIndex);
////    *c = ((a * b).square().sqrt() - a / b.inverse()).sin().cos();
//    DDouble sum = a + b;
//    *c = sum * sum;
//    printf("testing \n");
//}

//__global__ void functionTestsKernel(unsigned *index) {
//    unsigned globalIndex = 0;
//    DIDFunction idX = 0;
//    DIDFunction idY = 1;
//    DIDFunction idY2 = 2;
//    DMultiplicationFunction localContext.x1 = idX * idY;
//    DMultiplicationFunction localContext.x2 = localContext.x1 * idY2;
//    DPlusFunction x3 = localContext.x1 + localContext.x2;
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
//    printf("%f \n", parameters[localContext.x1.index].derivative);
//    printf("%f \n", parameters[localContext.x2.index].derivative);
//    printf("%f \n", parameters[x3.index].derivative);
//    assert(parameters[idX.index].derivative == 10.00);
//    assert(parameters[localContext.x1.index].derivative == 5.00);
//    assert(parameters[localContext.x2.index].derivative == 1.00);
//    assert(parameters[x3.index].derivative == 1.00);

//    DIDFunction idX = 0;
//    DIDFunction two = 1;
//    DSquareFunction xsquare = DSquareFunction(&idX);
//    DMultiplicationFunction localContext.x1 = two * xsquare;
//
//    DDouble parameters[4] = {DDouble(1.0, globalIndex, &globalIndex),
//                            DDouble(2.0, globalIndex, &globalIndex, CONST)};
//
//    DDouble *result = localContext.x1(parameters, 4);
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
//}

//__global__ void testF1Kernel(unsigned *params) {
//    unsigned globalIndex = 0;
//    DDouble operatorTree[14] = {DDouble(100.0, globalIndex, &globalIndex),
//                             DDouble(2.0, globalIndex, &globalIndex),
//                             DDouble(100.0, globalIndex, &globalIndex, CONST),
//                             DDouble(1.0, globalIndex, &globalIndex, CONST),
//                             DDouble(-1.0, globalIndex, &globalIndex, CONST)};
//    DIDFunction x0 = 0;
//    DIDFunction localContext.x1 = 1;
//    DIDFunction hundred = 2;
//    DIDFunction one = 3;
//    DIDFunction minOne = 4;
//    DSquareFunction x0square = DSquareFunction(&x0);
//    DMinusFunction t1 = localContext.x1 - x0square;
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

//__global__ void testF1DFloat(double *globalX, unsigned globalXSize) {
//    unsigned iterationCount = 30000;
//    double constants[3] = {100.0, 1.0, -1.0};
//    F1 f1 = F1(constants, 3);
//
//    const unsigned xSize = 2;
//    double localContext.x1[xSize] = {globalX[2 * threadIdx.x], globalX[2 * threadIdx.x + 1]};
//    double localContext.x2[xSize];
//    double *x = localContext.x1;
//    double *xNext = localContext.x2;
//    double alpha = 100;
//    double *tmp;
//    for (unsigned i = 0; i < iterationCount; i++) {
//        double f = f1.eval(x, xSize)->value;
////        printf("f %d: %f\n", i, f);
//        f1.evallocalContext.Jacobian();
//        alpha = 100;
//        f1.evalStep(x, xNext, xSize, f1.localContext.J, alpha);
//        double fNext = f1.eval(xNext, xSize)->value;
//        while (fNext >= f) {
//            alpha = alpha / 2;
//            f1.evalStep(x, xNext, xSize, f1.localContext.J, alpha);
//            fNext = f1.eval(xNext, xSize)->value;
////            printf("alpha: %.10f\n", alpha);
////            printf("fNext: %f.10\n", fNext);
//        }
//        tmp = x;
//        x = xNext;
//        xNext = tmp;
//    }
//    printf("x ");
//    for (unsigned j = 0; j < xSize; j++) {
//        printf("%f ", x[j]);
//    }
//    printf("\n");
////    printf("localContext.Jacobian:\n");
////    for (double i : f1.localContext.J) {
////        printf("%f\n", i);
////    }
////    printf("derivative: %f\n", f1.operatorTree[3].derivative);
//}
//__constant__ double dev_const_observations[CONSTANT_COUNT * CONSTANT_DIM];

struct SharedContext {
    double sharedX1[X_DIM];
    double sharedX2[X_DIM];
    double sharedDX[X_DIM];
    double sharedDXNormPartialSum[THREADS_PER_BLOCK];
    double *xCurrent;
    double *xNext;
    double sharedF;
    double sharedDXNorm;
};

struct LocalContext {
    double threadF;
    double alpha;
    void *residualProblems[RESIDUAL_COUNT];
};


__device__
void resetSharedState(SharedContext *sharedContext, unsigned threadIdx) {
    if (threadIdx == 0) {
        sharedContext->sharedF = 0.0;
    }
    for (unsigned spanningTID = threadIdx; spanningTID < X_DIM; spanningTID += blockDim.x) {
        sharedContext->sharedDX[spanningTID] = 0.0;
    }
    sharedContext->sharedDXNormPartialSum[threadIdx] = 0;
}

__device__
void reduceObservations(LocalContext *localContext,
                        SharedContext *sharedContext,
                        double *globalData) {
    localContext->threadF = 0;
#ifdef PROBLEM_PLANEFITTING
    PlaneFitting *f1 = ((PlaneFitting *) localContext->residualProblems[0]);
#endif
#ifdef PROBLEM_ROSENBROCK2D
    Rosenbrock2D *f1 = ((Rosenbrock2D *) localContext->residualProblems[0]);
#endif
#ifdef PROBLEM_SNLP
    SNLP *f1 = ((SNLP *) localContext->residualProblems[0]);
#endif

    for (unsigned spanningTID = threadIdx.x; spanningTID < RESIDUAL_CONSTANTS_COUNT_1; spanningTID += blockDim.x) {
        f1->setConstants(&globalData[CONSTANT_DIM * spanningTID],
                         CONSTANT_DIM);
        localContext->threadF += f1->eval(sharedContext->xCurrent,
                                          X_DIM)->value;
        f1->evalJacobian();
        for (unsigned j = 0; j < PARAMETER_DIM; j++) {
            atomicAdd(&sharedContext->sharedDX[f1->ThisJacobianIndices[j]],
                      f1->operatorTree[f1->constantSize + j].derivative);// TODO add jacobian variable indexing
        }
    }
}

__device__ void lineStep(double *x, double *xNext, unsigned xSize, double *jacobian, double alpha) {
    for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
        xNext[spanningTID] = x[spanningTID] - alpha * jacobian[spanningTID];
    }
}

__device__
void lineSearch(LocalContext *localContext,
                SharedContext *sharedContext,
                double *globalData,
                double currentF) {
    double fNext;
    localContext->alpha = ALPHA;
#ifdef PROBLEM_PLANEFITTING
    PlaneFitting *f1 = ((PlaneFitting *) localContext->residualProblems[0]);
#endif
#ifdef PROBLEM_ROSENBROCK2D
    Rosenbrock2D *f1 = ((Rosenbrock2D *) localContext->residualProblems[0]);
#endif
#ifdef PROBLEM_SNLP
    SNLP *f1 = ((SNLP *) localContext->residualProblems[0]);
#endif
    do {
        lineStep(sharedContext->xCurrent, sharedContext->xNext, X_DIM, sharedContext->sharedDX,
                 localContext->alpha);
        if (threadIdx.x == 0) {
            sharedContext->sharedF = 0;
        }
        fNext = 0;
        localContext->alpha = localContext->alpha / 2;
        __syncthreads();// sharedContext.sharedF is cleared
        for (unsigned spanningTID = threadIdx.x; spanningTID < RESIDUAL_CONSTANTS_COUNT_1; spanningTID += blockDim.x) {
            f1->setConstants(&globalData[CONSTANT_DIM * spanningTID], CONSTANT_DIM);
            fNext += f1->eval(sharedContext->xNext, X_DIM)->value;// TODO set xNext only once
        }
        atomicAdd(&sharedContext->sharedF, fNext); // TODO reduce over threads, not using atomicAdd
        __syncthreads();
    } while (sharedContext->sharedF > currentF);
}

__device__
void swapModels(SharedContext *sharedContext) {
    double *tmp = sharedContext->xCurrent;
    sharedContext->xCurrent = sharedContext->xNext;
    sharedContext->xNext = tmp;
}

__global__ void
gradientDescent(double *globalX, double *globalData,
                double *globalF) { // use shared memory instead of global memory
#ifdef PROBLEM_PLANEFITTING
    PlaneFitting f1 = PlaneFitting();
#endif
#ifdef PROBLEM_ROSENBROCK2D
    Rosenbrock2D f1 = Rosenbrock2D();
#endif
#ifdef PROBLEM_SNLP
    SNLP f1 = SNLP();
#endif
    // every thread has a local observation loaded into local memory

    // LOAD MODEL INTO SHARED MEMORY
    __shared__
    SharedContext sharedContext;
    const unsigned modelStartingIndex = X_DIM * blockIdx.x;
    for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
        sharedContext.sharedX1[spanningTID] = globalX[modelStartingIndex + spanningTID];
    }
    __syncthreads();
    // every thread can access the model in shared memory

    // INITIALIZE LOCAL MODEL
    LocalContext localContext;

    if (threadIdx.x == 0) {
        sharedContext.xCurrent = sharedContext.sharedX1;
        sharedContext.xNext = sharedContext.sharedX2;
    }

    localContext.alpha = ALPHA;
    localContext.residualProblems[0] = &f1;

    double fCurrent;
    // every thread has a copy of the shared model loaded, and an empty localContext.Jacobian
    double costDifference = INT_MAX;

    const double epsilon = 1e-7;
    sharedContext.sharedDXNorm = epsilon + 1;
    unsigned it;
    for (it = 0; it < ITERATION_COUNT && costDifference > epsilon && sharedContext.sharedDXNorm > epsilon; it++) {
        resetSharedState(&sharedContext, threadIdx.x);
        __syncthreads();
        // sharedContext.sharedF, sharedContext.sharedDX, is cleared // TODO this synchronizes over threads in a block, sync within grid required : https://on-demand.gputechconf.com/gtc/2017/presentation/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf
        reduceObservations(&localContext, &sharedContext, globalData);
        // localContext.threadF are calculated
        atomicAdd(&sharedContext.sharedF, localContext.threadF); // TODO reduce over threads, not using atomicAdd
        __syncthreads();
        // sharedContext.sharedF, sharedContext.sharedDX is complete for all threads
        fCurrent = sharedContext.sharedF;
        if (threadIdx.x == 0) {
            sharedContext.sharedDXNorm = 0;
        }
        __syncthreads();
        // fCurrent is set, sharedDXNorm is cleared for all threads,
        lineSearch(&localContext, &sharedContext, globalData, fCurrent);
        if (threadIdx.x == 0) {
            // sharedContext.xNext contains the model for the next iteration
            swapModels(&sharedContext);
        }

        double localDXNorm = 0;
        for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
            localDXNorm += std::pow(sharedContext.sharedDX[spanningTID], 2);
            // TODO reduce over threads, not using atomicAdd
        }
        sharedContext.sharedDXNormPartialSum[threadIdx.x] = localDXNorm;
        __syncthreads();

        if (threadIdx.x == 0) {
            double localDXNormFinal = 0;
            for (unsigned itdxnorm = 0; itdxnorm < THREADS_PER_BLOCK; itdxnorm++) {
                localDXNormFinal += sharedContext.sharedDXNormPartialSum[itdxnorm];
            }
            sharedContext.sharedDXNorm = std::sqrt(localDXNormFinal);
        }


        costDifference = std::abs(fCurrent - sharedContext.sharedF);
        __syncthreads();
        //xCurrent,xNext is set for all threads
        if (it % 500 == 0 && threadIdx.x == 0 && blockIdx.x == 0) {
            printf("xCurrent ");
            for (unsigned j = 0; j < X_DIM - 1; j++) {
                printf("%f,", sharedContext.xCurrent[j]);
            }
            printf("%f\n", sharedContext.xCurrent[X_DIM - 1]);
        }
    }

    if (threadIdx.x == 0) {
        printf("xCurrent ");
        for (unsigned j = 0; j < X_DIM - 1; j++) {
            printf("%f,", sharedContext.xCurrent[j]);
        }
        printf("%f\n", sharedContext.xCurrent[X_DIM - 1]);
        globalF[blockIdx.x] = sharedContext.sharedF;
        printf("\nWith: %d threads in block %d after it: %d f: %.10f\n", blockDim.x, blockIdx.x, it,
               sharedContext.sharedF);
    }
}

#endif //PARALLELLBFGS_GRADIENTDESCENT_CUH
