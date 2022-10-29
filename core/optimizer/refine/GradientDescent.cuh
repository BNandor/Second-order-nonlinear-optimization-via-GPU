//
// Created by spaceman on 2022. 01. 16..
//

#ifndef PARALLELLBFGS_GRADIENTDESCENT_CUH
#define PARALLELLBFGS_GRADIENTDESCENT_CUH

#include "../../AD/DDouble.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DPlusFunction.cuh"
#include "../../AD/function/DMultiplicationFunction.cuh"
#include "../../AD/function/DMinusFunction.cuh"
#include "../../AD/function/DIDFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/function/Operations.cuh"
#include "../../problem/F1.cuh"
#include "../../problem/PlaneFitting.cuh"
#include "../../problem/Rosenbrock2D.cuh"
#include "../../problem/SNLP/SNLP.cuh"
#include "../../problem/SNLP/SNLPAnchor.cuh"
#include "../../common/FIFOQueue.cuh"
#include "../../problem/SNLP/SNLP3D.cuh"
#include "../../problem/SNLP/SNLP3DAnchor.cuh"
#include "../../common/model/Model.cuh"
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
//__constant__ double dev_const_observations[CONSTANT_COUNT * RESIDUAL_CONSTANTS_DIM_1];

//__global__
//void testQueue(double *globalX) {
//    FIFOQueue queue1 = FIFOQueue();
//    for (unsigned i = 1; i <= queue1.queueSize - 1; i++) {
//        printf("enqueueuing %d\n", i);
//        queue1.enqueue((double *) i);
//    }
//    unsigned iterator = queue1.getIterator();
//    int i = 0;
//    while (queue1.hasNext(iterator)) {
//        i++;
//        double *next = queue1.next(iterator);
//        assert(next == (double *) i);
//        printf("%d\n", next);
//    }
//
//    FIFOQueue queue2 = FIFOQueue();
//    for (unsigned i = 1; i <= queue1.queueSize + 1; i++) {
//        printf("enqueueuing %d\n", i);
//        queue2.enqueue((double *) i);
//    }
//    unsigned iterator2 = queue2.getIterator();
//    i = 1;
//    while (queue2.hasNext(iterator2)) {
//        i++;
//        double *next = queue2.next(iterator2);
//        assert(next == (double *) i);
//        printf("%d\n", next);
//    }
//
//    unsigned iterator3 = queue2.getIterator();
//    i = queue2.queueSize + 1;
//    while (queue2.hasNext(iterator3)) {
//        double *next = queue2.reverseNext(iterator3);
//        assert(next == (double *) i);
//        printf("%d\n", next);
//        i--;
//    }
//}

//__global__
//void testDot(double *globalX) {
//    double a[3] = {1, 2, 3};
//    double b[3] = {1, 2, 3};
//    __shared__ SharedContext sharedContext;
//    printf("%f\n", dot(a, b, 3, sharedContext));
//}

namespace GD {
    std::string name="GD";
    struct GlobalData {
        double sharedX1[X_DIM];
        double sharedX2[X_DIM];
        double sharedDX[X_DIM];
    };

    struct SharedContext {
        GlobalData *globalData;
        double sharedScratchPad[THREADS_PER_BLOCK];
        double sharedResult;
        double *xCurrent;
        double *xNext;
        double sharedF;
        double sharedDXNorm;
    };

    struct LocalContext {
        double threadF;
        double alpha;
        void *residualProblems[RESIDUAL_COUNT];
        double *residualConstants[RESIDUAL_COUNT];
        unsigned fEvaluations;
    };

    __device__
    void resetSharedState(SharedContext *sharedContext, unsigned threadIdx) {
        if (threadIdx == 0) {
            sharedContext->sharedF = 0.0;
        }
        for (unsigned spanningTID = threadIdx; spanningTID < X_DIM; spanningTID += blockDim.x) {
            sharedContext->globalData->sharedDX[spanningTID] = 0.0;
        }
        sharedContext->sharedScratchPad[threadIdx] = 0; // TODO check if necessary
    }

    __device__
    void reduceObservations(LocalContext *localContext,
                            SharedContext *sharedContext,
                            double *globalData,void* modelP) {
        ++localContext->fEvaluations;
        localContext->threadF = 0;
        Model *model = (Model *) modelP;
        COMPUTE_RESIDUALS()
    }

    __device__ void lineStep(double *x, double *xNext, unsigned xSize, double *jacobian, double alpha) {
        for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
            xNext[spanningTID] = x[spanningTID] - alpha * jacobian[spanningTID];
        }
    }

    __device__
    bool lineSearch(LocalContext *localContext,
                    SharedContext *sharedContext,
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
        SNLPAnchor *f2 = ((SNLPAnchor *) localContext->residualProblems[1]);
#endif
#ifdef PROBLEM_SNLP3D
        SNLP3D *f1 = ((SNLP3D *) localContext->residualProblems[0]);
        SNLP3DAnchor *f2 = ((SNLP3DAnchor *) localContext->residualProblems[1]);
#endif
        do {
            ++localContext->fEvaluations;
            lineStep(sharedContext->xCurrent, sharedContext->xNext, X_DIM, sharedContext->globalData->sharedDX,
                     localContext->alpha);
            if (threadIdx.x == 0) {
                sharedContext->sharedF = 0;
            }
            fNext = 0;
            localContext->alpha = localContext->alpha / 2;
            __syncthreads();// sharedContext.sharedF is cleared
            for (unsigned spanningTID = threadIdx.x;
                 spanningTID < RESIDUAL_CONSTANTS_COUNT_1; spanningTID += blockDim.x) {
                f1->setConstants(&(localContext->residualConstants[0][RESIDUAL_CONSTANTS_DIM_1 * spanningTID]),
                                 RESIDUAL_CONSTANTS_DIM_1);
                fNext += f1->eval(sharedContext->xNext, X_DIM)->value;// TODO set xNext only once
            }
#if defined(PROBLEM_SNLP) || defined(PROBLEM_SNLP3D)
            for (unsigned spanningTID = threadIdx.x;
                 spanningTID < RESIDUAL_CONSTANTS_COUNT_2; spanningTID += blockDim.x) {
                f2->setConstants(&(localContext->residualConstants[1][RESIDUAL_CONSTANTS_DIM_2 * spanningTID]),
                                 RESIDUAL_CONSTANTS_DIM_2);
                fNext += f2->eval(sharedContext->xNext,
                                  X_DIM)->value;
            }
#endif
            atomicAdd(&sharedContext->sharedF, fNext); // TODO reduce over threads, not using atomicAdd
            __syncthreads();
        } while (sharedContext->sharedF > currentF && localContext->alpha!=0.0);
        return localContext->alpha!=0.0;
    }

    __device__
    void swapModels(SharedContext *sharedContext) {
        double *tmp = sharedContext->xCurrent;
        sharedContext->xCurrent = sharedContext->xNext;
        sharedContext->xNext = tmp;
    }
    __global__ void
    optimize(double *globalX, double *globalData,
             double *globalF
//#ifdef GLOBAL_SHARED_MEM
            , GlobalData *globalSharedContext,
            void * model,
int iterations
//#endif
    ) { // use shared memory instead of global memory
#ifdef PROBLEM_PLANEFITTING
        PlaneFitting f1 = PlaneFitting();
#endif
#ifdef PROBLEM_ROSENBROCK2D
        Rosenbrock2D f1 = Rosenbrock2D();
#endif
#ifdef PROBLEM_SNLP
        SNLP f1 = SNLP();
        SNLPAnchor f2 = SNLPAnchor();
#endif
#ifdef PROBLEM_SNLP3D
        SNLP3D f1 = SNLP3D();
        SNLP3DAnchor f2 = SNLP3DAnchor();
#endif
        if(blockIdx.x ==0 && threadIdx.x ==0) {
            Model *model1 = (Model *) model;
            printf("GD iterations: %d\n", model1->localIterations);
            printf("residuals: %d\n", model1->residuals.residualCount);

            for (int residualIndex = 0; residualIndex < model1->residuals.residualCount; residualIndex++) {
                printf("constants: %d\n", model1->residuals.residual[residualIndex].constantsCount);
                printf("constants dim: %d\n", model1->residuals.residual[residualIndex].constantsDim);
                printf("parameters: %d\n", model1->residuals.residual[residualIndex].parametersDim);
            }
        }


        // every thread has a local observation loaded into local memory
        __shared__
        SharedContext sharedContext;
//#ifdef GLOBAL_SHARED_MEM
        sharedContext.globalData = &globalSharedContext[blockIdx.x];

//#else
        // LOAD MODEL INTO SHARED MEMORY
//        __shared__
//        SharedContext sharedContext;
//#endif
        const unsigned modelStartingIndex = X_DIM * blockIdx.x;
        for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
            sharedContext.globalData->sharedX1[spanningTID] = globalX[modelStartingIndex + spanningTID];
        }
        __syncthreads();
        // every thread can access the model in shared memory

        // INITIALIZE LOCAL MODEL
        LocalContext localContext;

        if (threadIdx.x == 0) {
            sharedContext.xCurrent = sharedContext.globalData->sharedX1;
            sharedContext.xNext = sharedContext.globalData->sharedX2;
        }
        localContext.fEvaluations = 0;
        localContext.alpha = ALPHA;
        localContext.residualProblems[0] = &f1;
        localContext.residualConstants[0] = globalData;
#if defined(PROBLEM_SNLP) || defined(PROBLEM_SNLP3D)
        localContext.residualProblems[1] = &f2;
        localContext.residualConstants[1] =
                localContext.residualConstants[0] + RESIDUAL_CONSTANTS_COUNT_1 * RESIDUAL_CONSTANTS_DIM_1;
#endif
        double fCurrent;
        // every thread has a copy of the shared model loaded, and an empty localContext.Jacobian
        double costDifference = INT_MAX;

        const double epsilon = FEPSILON;
        sharedContext.sharedDXNorm = epsilon + 1;
        unsigned it;
//        for (it = 0; it < ITERATION_COUNT; it++) {
        for (it = 0; localContext.fEvaluations < iterations; it++) {

            resetSharedState(&sharedContext, threadIdx.x);
            __syncthreads();
            // sharedContext.sharedF, sharedContext.globalData->sharedX2, is cleared // TODO this synchronizes over threads in a block, sync within grid required : https://on-demand.gputechconf.com/gtc/2017/presentation/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf
            reduceObservations(&localContext, &sharedContext, globalData,model);
            // localContext.threadF are calculated
            atomicAdd(&sharedContext.sharedF, localContext.threadF); // TODO reduce over threads, not using atomicAdd
            __syncthreads();
            // sharedContext.sharedF, sharedContext-globalData->sharedDXis complete for all threads
            fCurrent = sharedContext.sharedF;
//            if (threadIdx.x == 0) {
//                sharedContext.sharedDXNorm = 0;
//            }
            __syncthreads();
#ifdef PRINT
            if (it % FRAMESIZE == 0 && threadIdx.x == 0 && blockIdx.x == 0) {
                printf("xCurrent ");
                for (unsigned j = 0; j < X_DIM - 1; j++) {
                    printf("%f,", sharedContext.xCurrent[j]);
                }
                printf("%f\n", sharedContext.xCurrent[X_DIM - 1]);
                printf("f: %f\n", fCurrent);
            }
#endif
            // fCurrent is set, sharedDXNorm is cleared for all threads,
            ;
            if(!lineSearch(&localContext, &sharedContext, fCurrent)){
                if(threadIdx.x == 0){
                    sharedContext.sharedF=fCurrent;
                }
                __syncthreads();
                break;
            }
            if (threadIdx.x == 0) {
                // sharedContext.xNext contains the model for the next iteration
                swapModels(&sharedContext);
            }

//            double localDXNorm = 0;
//            for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
//                localDXNorm += std::pow(sharedContext.globalData->sharedDX[spanningTID], 2);
//                // TODO reduce over threads, not using atomicAdd
//            }
//            sharedContext.sharedScratchPad[threadIdx.x] = localDXNorm;
//            __syncthreads();
//
//            if (threadIdx.x == 0) {
//                double localDXNormFinal = 0;
//                for (unsigned itdxnorm = 0; itdxnorm < THREADS_PER_BLOCK; itdxnorm++) {
//                    localDXNormFinal += sharedContext.sharedScratchPad[itdxnorm];
//                }
//                sharedContext.sharedDXNorm = std::sqrt(localDXNormFinal);
//            }
//
//            costDifference = std::abs(fCurrent - sharedContext.sharedF);
            __syncthreads();
            //xCurrent,xNext is set for all threads

        }

#ifdef PRINT
        if (threadIdx.x == 0 && blockIdx.x == 0) {

            printf("xCurrent ");
            for (unsigned j = 0; j < X_DIM - 1; j++) {
                printf("%f,", sharedContext.xCurrent[j]);
            }
            printf("%f\n", sharedContext.xCurrent[X_DIM - 1]);
            printf("\nthreads:%d", blockDim.x);
            printf("\niterations:%d", it);
            printf("\nfevaluations: %d\n", localContext.fEvaluations);
        }

#endif

        if (threadIdx.x == 0) {
            globalF[blockIdx.x] = sharedContext.sharedF;

//            printf("\nWith: %d threads in block %d after it: %d f: %.10f\n", blockDim.x, blockIdx.x, it,
//                   sharedContext.sharedF);
        }
        for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
            globalX[modelStartingIndex + spanningTID]=sharedContext.xCurrent[spanningTID];
        }
    }

    __global__ void
    evaluateF(double *globalX, double *globalData,
             double *globalF, GlobalData *globalSharedContext,void*model
    ) {
// use shared memory instead of global memory
#ifdef PROBLEM_PLANEFITTING
        PlaneFitting f1 = PlaneFitting();
#endif
#ifdef PROBLEM_ROSENBROCK2D
        Rosenbrock2D f1 = Rosenbrock2D();
#endif
#ifdef PROBLEM_SNLP
        SNLP f1 = SNLP();
        SNLPAnchor f2 = SNLPAnchor();
#endif
#ifdef PROBLEM_SNLP3D
        SNLP3D f1 = SNLP3D();
        SNLP3DAnchor f2 = SNLP3DAnchor();
#endif
        // every thread has a local observation loaded into local memory
        __shared__
        SharedContext sharedContext;
//#ifdef GLOBAL_SHARED_MEM
        sharedContext.globalData = &globalSharedContext[blockIdx.x];

//#else
        // LOAD MODEL INTO SHARED MEMORY
//        __shared__
//        SharedContext sharedContext;
//#endif
        const unsigned modelStartingIndex = X_DIM * blockIdx.x;
        for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
            sharedContext.globalData->sharedX1[spanningTID] = globalX[modelStartingIndex + spanningTID];
        }
        __syncthreads();
        // every thread can access the model in shared memory

        // INITIALIZE LOCAL MODEL
        LocalContext localContext;

        if (threadIdx.x == 0) {
            sharedContext.xCurrent = sharedContext.globalData->sharedX1;
            sharedContext.xNext = sharedContext.globalData->sharedX2;
        }
        localContext.fEvaluations = 0;
        localContext.alpha = ALPHA;
        localContext.residualProblems[0] = &f1;
        localContext.residualConstants[0] = globalData;
#if defined(PROBLEM_SNLP) || defined(PROBLEM_SNLP3D)
        localContext.residualProblems[1] = &f2;
        localContext.residualConstants[1] =
                localContext.residualConstants[0] + RESIDUAL_CONSTANTS_COUNT_1 * RESIDUAL_CONSTANTS_DIM_1;
#endif
        double fCurrent;
        // every thread has a copy of the shared model loaded, and an empty localContext.Jacobian
        double costDifference = INT_MAX;

        const double epsilon = FEPSILON;
        sharedContext.sharedDXNorm = epsilon + 1;
        unsigned it;

        resetSharedState(&sharedContext, threadIdx.x);
        __syncthreads();
        // sharedContext.sharedF, sharedContext.globalData->sharedX2, is cleared // TODO this synchronizes over threads in a block, sync within grid required : https://on-demand.gputechconf.com/gtc/2017/presentation/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf
        reduceObservations(&localContext, &sharedContext, globalData,model);
        // localContext.threadF are calculated
        atomicAdd(&sharedContext.sharedF, localContext.threadF); // TODO reduce over threads, not using atomicAdd
        __syncthreads();
#ifdef PRINT
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("xCurrent ");
            for (unsigned j = 0; j < X_DIM - 1; j++) {
                printf("%f,", sharedContext.xCurrent[j]);
            }
            printf("%f\n", sharedContext.xCurrent[X_DIM - 1]);
        }
#endif

        if (threadIdx.x == 0) {
            globalF[blockIdx.x] = sharedContext.sharedF;
//            printf("\nthreads:%d", blockDim.x);
//            printf("\niterations:%d", it);
//            printf("\nfevaluations: %d\n", localContext.fEvaluations);
            printf("f: %f\n", sharedContext.sharedF);
//            printf("\nWith: %d threads in block %d after it: %d f: %.10f\n", blockDim.x, blockIdx.x, it,
//                   sharedContext.sharedF);
        }
    }
}
#endif //PARALLELLBFGS_GRADIENTDESCENT_CUH
