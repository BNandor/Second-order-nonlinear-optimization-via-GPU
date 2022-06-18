//
// Created by spaceman on 2022. 03. 07..
//

#ifndef PARALLELLBFGS_LBFGS_CUH
#define PARALLELLBFGS_LBFGS_CUH

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
#include "../problem/SNLP/SNLP.cuh"
#include "../problem/SNLP/SNLPAnchor.cuh"
#include "../problem/SNLP/SNLP3D.cuh"
#include "../problem/SNLP/SNLP3DAnchor.cuh"
#include "../common/FIFOQueue.cuh"
#include <stdio.h>


namespace LBFGS {
    std::string name="LBFGS";
#define LBFGS_M 5
#define LBFGS_LINESEARCH_C1 0.0001
#define LBFGS_LINESEARCH_C2 0.9

    struct GlobalData {
        double sharedX1[X_DIM];
        double sharedX2[X_DIM];
        double sharedDX[X_DIM];
        double xdimScratchPad[X_DIM];
        double lbfgsQueueS[LBFGS_M][X_DIM];
        double lbfgsQueueY[LBFGS_M][X_DIM];
        double lbfgsR[X_DIM];
    };

    struct SharedContext {
        GlobalData *globalData;
        double sharedScratchPad[THREADS_PER_BLOCK]; // TODO make these shared (which fit)
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
    void minusNoSync(double *a, double *b, double *result, unsigned size) {
        for (unsigned spanningTID = threadIdx.x; spanningTID < size; spanningTID += blockDim.x) {
            result[spanningTID] = a[spanningTID] - b[spanningTID];
        }
    }

    __device__
    void aMinusBtimesCNoSync(double *a, double b, double *c, double *result, unsigned size) {
        for (unsigned spanningTID = threadIdx.x; spanningTID < size; spanningTID += blockDim.x) {
            result[spanningTID] = a[spanningTID] - b * c[spanningTID];
        }
    }

    __device__
    void setAll(double *a, const double b, unsigned size) {
        for (unsigned spanningTID = threadIdx.x; spanningTID < size; spanningTID += blockDim.x) {
            a[spanningTID] = b;
        }
    }

    __device__
    void mulNoSync(double *a, double b, double *result, unsigned size) {
        for (unsigned spanningTID = threadIdx.x; spanningTID < size; spanningTID += blockDim.x) {
            result[spanningTID] = a[spanningTID] * b;
        }
    }

    __device__
    void copyNoSync(double *to, double *from, unsigned size) {
        for (unsigned spanningTID = threadIdx.x; spanningTID < size; spanningTID += blockDim.x) {
            to[spanningTID] = from[spanningTID];
        }
    }

    __device__
    double dot(double *a, double *b, unsigned size, SharedContext &sharedContext) {
        double localSum = 0;
        for (unsigned spanningTID = threadIdx.x; spanningTID < size; spanningTID += blockDim.x) {
            localSum += a[spanningTID] * b[spanningTID];
        }

        sharedContext.sharedScratchPad[threadIdx.x] = localSum;
        __syncthreads();
        // every thread set its local sum in the scratchpad
        if (threadIdx.x == 0) {
            double finalSum = 0;
            for (unsigned itsum = 0; itsum < THREADS_PER_BLOCK; itsum++) {
                finalSum += sharedContext.sharedScratchPad[itsum];
            }
            sharedContext.sharedResult = finalSum;
        }
        __syncthreads();
        // every thread has sharedResult set
        return sharedContext.sharedResult;
    }

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
                            double *x,
                            double *dx) {
        ++localContext->fEvaluations;
        localContext->threadF = 0;
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
        for (unsigned spanningTID = threadIdx.x; spanningTID < RESIDUAL_CONSTANTS_COUNT_1; spanningTID += blockDim.x) {
            f1->setConstants(&(localContext->residualConstants[0][RESIDUAL_CONSTANTS_DIM_1 * spanningTID]),
                             RESIDUAL_CONSTANTS_DIM_1);
            localContext->threadF += f1->eval(x,
                                              X_DIM)->value;
            f1->evalJacobian();
            for (unsigned j = 0; j < RESIDUAL_PARAMETERS_DIM_1; j++) {
                atomicAdd(&dx[f1->ThisJacobianIndices[j]],
                          f1->operatorTree[f1->constantSize + j].derivative);// TODO add jacobian variable indexing
            }
        }

#if defined(PROBLEM_SNLP) || defined(PROBLEM_SNLP3D)
        for (unsigned spanningTID = threadIdx.x; spanningTID < RESIDUAL_CONSTANTS_COUNT_2; spanningTID += blockDim.x) {
            f2->setConstants(&(localContext->residualConstants[1][RESIDUAL_CONSTANTS_DIM_2 * spanningTID]),
                             RESIDUAL_CONSTANTS_DIM_2);
            localContext->threadF += f2->eval(x,
                                              X_DIM)->value;
            f2->evalJacobian();
            for (unsigned j = 0; j < RESIDUAL_PARAMETERS_DIM_2; j++) {
                atomicAdd(&dx[f2->ThisJacobianIndices[j]],
                          f2->operatorTree[f2->constantSize + j].derivative);// TODO add jacobian variable indexing
            }
        }
#endif
    }

    __device__ void lineStep(double *x, double *xNext, unsigned xSize, double *jacobian, double alpha) {
        for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
            xNext[spanningTID] = x[spanningTID] - alpha * jacobian[spanningTID];
        }
    }

    __device__
    int lineSearch(LocalContext *localContext,
                    SharedContext *sharedContext,
                    double *DX,
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
            lineStep(sharedContext->xCurrent, sharedContext->xNext, X_DIM, DX,
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
//            if(threadIdx.x == 0) {
////            printf("Global data at %lu for block %d\n", sizeof(GlobalData) * blockIdx.x, blockIdx.x);
//                printf("after: %d",blockIdx.x);
//            }
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
    void LBFGSlineSearch(LocalContext *localContext,
                         SharedContext *sharedContext,
                         double *DX,
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
//            if (threadIdx.x == 0) {
//                printf("nextF:%f, currentF:%f costDiff %.20f: alpha: %f\n", sharedContext->sharedF, currentF,
//                       sharedContext->sharedF - currentF, localContext->alpha);
//            }
            aMinusBtimesCNoSync(sharedContext->xCurrent, -localContext->alpha, DX, sharedContext->xNext, X_DIM);
//            lineStep(sharedContext->xCurrent, sharedContext->xNext, X_DIM, DX,
//            0);
//            copyNoSync(sharedContext->xNext, sharedContext->xCurrent, X_DIM);
            if (threadIdx.x == 0) {
                sharedContext->sharedF = 0;
            }
            fNext = 0;
            localContext->alpha = localContext->alpha / 2;
            __syncthreads();// sharedContext.sharedF is cleared
//            if (threadIdx.x == 0) {
//                // print Queues after GD
//                printf("\nxNext top 10");
//                for (unsigned j = 0; j < 10; j++) {
//                    printf("%f,", sharedContext->xNext[j]);
//                }
//            }
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
        } while (sharedContext->sharedF > currentF);
    }

    __device__
    int zoom(LocalContext
              *localContext, SharedContext *sharedContext, double alpha0, double alpha1, double *r,
              double *DXNext, double currentF, double JdotR) {
//        printf("zoom  %f to %f\n", alpha0, alpha1);
        double alphaLow = alpha0;
        double alphaHigh = alpha1;
        double alphaMid;
        double fAlphaMid;
        double gradientAlphaMid;
        do {
//      println("alphaLow "+alphaLow)
//      println("alphaHigh "+alphaHigh)
            alphaMid = (alphaLow + alphaHigh) / 2.0;
            aMinusBtimesCNoSync(sharedContext->xCurrent, -alphaMid, r, sharedContext->xNext, X_DIM);
            // xNext = x + alphaMid*r
            if (threadIdx.x == 0) {
                sharedContext->sharedF = 0;
            }
            setAll(DXNext, 0, X_DIM);
            __syncthreads();
            // DXNext, sharedF is 0
            reduceObservations(localContext, sharedContext->xNext, DXNext);
            // localContext.threadF are calculated
            atomicAdd(&sharedContext->sharedF, localContext->threadF);
            __syncthreads();
            // DXNext, sharedF =  f(x + alphaMid.r),xNext = x + alphaMid*r
            fAlphaMid = sharedContext->sharedF;

            if (fAlphaMid > currentF + LBFGS_LINESEARCH_C1 * alphaMid * JdotR) {
                alphaHigh = alphaMid;
            } else {
                aMinusBtimesCNoSync(sharedContext->xCurrent, -alphaLow, r, sharedContext->xNext, X_DIM);
                // xNext = x + alphaLow*r
                __syncthreads();//
                if (threadIdx.x == 0) {
                    sharedContext->sharedF = 0;
                }
                setAll(sharedContext->globalData->xdimScratchPad, 0, X_DIM);
                __syncthreads();
                // xdimScratchPad, sharedF is 0
                reduceObservations(localContext, sharedContext->xNext, sharedContext->globalData->xdimScratchPad);
                // localContext.threadF are calculated
                atomicAdd(&sharedContext->sharedF, localContext->threadF);
                __syncthreads();
                // xdimScratchPad, sharedF =  f(x + alphaLow.r),xNext = x + alphaLow*r
                if (fAlphaMid >= sharedContext->sharedF) {
                    alphaHigh = alphaMid;
                } else {
                    gradientAlphaMid = dot(DXNext, r, X_DIM, *sharedContext);
                    if (abs(gradientAlphaMid) <= -LBFGS_LINESEARCH_C2 * JdotR) {
//                    return alphaMid
                        aMinusBtimesCNoSync(sharedContext->xCurrent, -alphaMid, r, sharedContext->xNext, X_DIM);
                        __syncthreads();
                        if (threadIdx.x == 0) {
                            sharedContext->sharedF = fAlphaMid;
                        }
                        __syncthreads();
                        // DXNext, sharedF =  f(x + alphaMid.r),xNext = x + alphaMid*r
                        return 0;
                    }
                    if (gradientAlphaMid * (alphaHigh - alphaLow) >= 0) {
                        alphaHigh = alphaLow;
                    }
                    alphaLow = alphaMid;
                }
            }
            __syncthreads();
        } while (alphaLow != alphaHigh);
#ifdef PRINT
        printf("Error, could not zoom!\n");
#endif

        return -1;
    }

    __device__
    int LBFGSlineSearchWolfeConditions(LocalContext
                                        *localContext,
                                        SharedContext *sharedContext,
                                        double *J,
                                        double *r,
                                        double *DXNext,
                                        double currentF
    ) {
//        LBFGS_LINESEARCH_C2
        double alphaMax = 10000;
        double alpha0 = 0.0;
        double alpha1 = 1.0;
        unsigned i = 1;
        bool zoomed = false;
        double falpha1;
        double falpha0 = currentF;
        double gradientAlpha1;
        double JdotR = dot(J, r, X_DIM, *sharedContext);
        do {
            aMinusBtimesCNoSync(sharedContext->xCurrent, -alpha1, r, sharedContext->xNext, X_DIM);
            // xNext = x + alpha1*r
            if (threadIdx.x == 0) {
                sharedContext->sharedF = 0;
            }
            setAll(DXNext, 0, X_DIM);
            __syncthreads();
            // DXNext, sharedF is 0
            reduceObservations(localContext, sharedContext->xNext, DXNext);
            // localContext.threadF are calculated
            atomicAdd(&sharedContext->sharedF, localContext->threadF);
            __syncthreads();
            // DXNext, sharedF =  f(x + alpha1.r),xNext = x + alpha1*r
            falpha1 = sharedContext->sharedF;

            if (falpha1 > currentF + alpha1 * LBFGS_LINESEARCH_C1 * JdotR ||
                (i > 1 && falpha1 >= falpha0)) {
                return zoom(localContext, sharedContext, alpha0, alpha1, r, DXNext, currentF, JdotR);
            }
            gradientAlpha1 = dot(DXNext, r, X_DIM, *sharedContext);
            if (abs(gradientAlpha1) <= -LBFGS_LINESEARCH_C2 * JdotR) {
                return 0;
            }
            if (gradientAlpha1 >= 0) {
                return zoom(localContext, sharedContext, alpha1, alpha0, r, DXNext, currentF, JdotR);
            }
            alpha0 = alpha1;
            falpha0 = falpha1;
            alpha1 = alpha1 * 2;
            i += 1;
        } while (alpha1 < alphaMax);
#ifdef PRINT
        printf("error: reached max bracket in linesearch");
#endif
        return -2;
    }

    __device__
    void swapModels(SharedContext *sharedContext) {
        double *tmp = sharedContext->xCurrent;
        sharedContext->xCurrent = sharedContext->xNext;
        sharedContext->xNext = tmp;
    }

    __device__
    void approximateImplicitHessian(double *DX, int k, FIFOQueue *sQueue, FIFOQueue *yQueue,
                                    SharedContext *sharedContext) {
        double alphas[LBFGS_M] = {};
        double ros[LBFGS_M] = {};
        unsigned sRev = sQueue->getIterator();
        unsigned yRev = yQueue->getIterator();
        double *s = sQueue->reverseNext(sRev);
        double *y = yQueue->reverseNext(yRev);
        int j = k - 1;
        copyNoSync(sharedContext->globalData->lbfgsR, DX, X_DIM);
        __syncthreads();
        // R = J for all threads
        while (j >= k - LBFGS_M) {
            if (j >= 0) { //TODO check if this is necessary
                ros[k - j - 1] = 1.0 / dot(y, s, X_DIM, *sharedContext);
                alphas[k - j - 1] = dot(s, sharedContext->globalData->lbfgsR, X_DIM, *sharedContext) * ros[k - j - 1];
                aMinusBtimesCNoSync(sharedContext->globalData->lbfgsR, alphas[k - j - 1], y,
                                    sharedContext->globalData->lbfgsR, X_DIM);
                __syncthreads();
                // ros[k - j - 1],alphas[k - j - 1], R(q) updated for all threads
                if (yQueue->hasNext(yRev)) {
                    s = sQueue->reverseNext(sRev);
                    y = yQueue->reverseNext(yRev);
                }
            }
            --j;
        }
        // TODO chech gamma : var r = q*gamma// Nocedal 178, TODO test if eye can be dropped

        unsigned sIt = sQueue->getIterator();
        unsigned yIt = yQueue->getIterator();
        for (int i = k - LBFGS_M; i < k; i++) {
            if (i >= 0) {
                s = sQueue->next(sIt);
                y = yQueue->next(yIt);
                double t = (alphas[k - i - 1] -
                            dot(y, sharedContext->globalData->lbfgsR, X_DIM, *sharedContext) * ros[k - i - 1]);
                aMinusBtimesCNoSync(sharedContext->globalData->lbfgsR, -t, s, sharedContext->globalData->lbfgsR, X_DIM);
                __syncthreads();
            }
        }
        mulNoSync(sharedContext->globalData->lbfgsR, -1, sharedContext->globalData->lbfgsR, X_DIM);
    }

    __global__ void
    optimize(double *globalX, double *globalData,
             double *globalF
//#ifdef GLOBAL_SHARED_MEM
            , GlobalData *globalSharedContext
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
        // every thread has a local observation loaded into local memory
        FIFOQueue sQueue = FIFOQueue();
        FIFOQueue yQueue = FIFOQueue();
        __shared__
        SharedContext sharedContext;

//#ifdef GLOBAL_SHARED_MEM
        sharedContext.globalData = &globalSharedContext[blockIdx.x];

//#else
//        // LOAD MODEL INTO SHARED MEMORY
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

        localContext.alpha = ALPHA;
        localContext.fEvaluations = 0;
        localContext.residualProblems[0] = &f1;
        localContext.residualConstants[0] = globalData;
#if defined(PROBLEM_SNLP) || defined(PROBLEM_SNLP3D)
        localContext.residualProblems[1] = &f2;
        localContext.residualConstants[1] =
                localContext.residualConstants[0] + RESIDUAL_CONSTANTS_COUNT_1 * RESIDUAL_CONSTANTS_DIM_1;
#endif
        double fCurrent;
        // every thread has a copy of the shared model loaded, and an empty localContext.Jacobian


        const double epsilon = FEPSILON;
        sharedContext.sharedDXNorm = epsilon + 1;
        int it;

        for (it = 1; it <= LBFGS_M;
             it++) {

            resetSharedState(&sharedContext, threadIdx.x);

            __syncthreads();
            // sharedContext.sharedF, sharedContext.sharedDX, is cleared // TODO this synchronizes over threads in a block, sync within grid required : https://on-demand.gputechconf.com/gtc/2017/presentation/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf
            reduceObservations(&localContext, sharedContext.xCurrent, sharedContext.globalData->sharedDX);
            // localContext.threadF are calculated
            atomicAdd(&sharedContext.sharedF,
                      localContext.threadF); // TODO reduce over threads, not using atomicAdd
            __syncthreads();

            // sharedContext.sharedF, sharedContext.sharedDX is complete for all threads
            fCurrent = sharedContext.sharedF;

//            if (threadIdx.x == 0) {
//                printf("it: %d f: %f\n", it, fCurrent);
//            }

            __syncthreads();
            if (threadIdx.x == 0 && blockIdx.x == 0) {
#ifdef PRINT
                printf("xCurrent ");
                for (unsigned j = 0; j < X_DIM - 1; j++) {
                    printf("%f,", sharedContext.xCurrent[j]);
                }
                printf("%f\n", sharedContext.xCurrent[X_DIM - 1]);
#endif
                printf("f: %f\n", fCurrent);
            }
            // fCurrent is set, sharedDXNorm is cleared for all threads,
            lineSearch(&localContext, &sharedContext, sharedContext.globalData->sharedDX, fCurrent);

            // sharedContext.xNext contains the model for the next iteration, sharedContext.sharedDX is for sharedContext.xCurrent model
            // sharedContext.sharedF is set for xNext
            minusNoSync(sharedContext.xNext, sharedContext.xCurrent, sharedContext.globalData->lbfgsQueueS[sQueue.back],
                        X_DIM);
            // sharedContext.globalData->lbfgsQueueS[queue.back] = xNext - xCurrent
            //setAll(sharedContext.globalData->lbfgsQueueY[yQueue.back], 0, X_DIM); // not necessary initially
            reduceObservations(&localContext, sharedContext.xNext, sharedContext.globalData->lbfgsQueueY[yQueue.back]);
            __syncthreads();
            // DX[xNext] in sharedContext.globalData->lbfgsQueueY[queue.back]
            // DX[xCurrent] in sharedContext.sharedDX
            minusNoSync(sharedContext.globalData->lbfgsQueueY[yQueue.back], sharedContext.globalData->sharedDX,
                        sharedContext.globalData->lbfgsQueueY[yQueue.back], X_DIM);
            // sharedContext.globalData->lbfgsQueueY[queue.back]: DX[xNext] - DX[xCurrent]
            __syncthreads();
            // sharedContext.globalData->lbfgsQueueS[queue.back] = xNext - xCurrent
            // sharedContext.globalData->lbfgsQueueY[queue.back] = DX[xNext] - DX[xCurrent]
            sQueue.enqueue(sharedContext.globalData->lbfgsQueueS[sQueue.back]);
            yQueue.enqueue(sharedContext.globalData->lbfgsQueueY[yQueue.back]);
            if (threadIdx.x == 0) {
                // sharedContext.xNext contains the model for the next iteration
                swapModels(&sharedContext);
            }
            __syncthreads();
            //xCurrent,xNext is set for all threads
        }
        double costDifference = INT_MAX;
        for (; localContext.fEvaluations < ITERATION_COUNT; it++) {

            // reset states
            resetSharedState(&sharedContext, threadIdx.x);
//            if (threadIdx.x == 0) {
//                sharedContext.sharedDXNorm = 0;
//            }

            __syncthreads();// sharedContext.sharedF is cleared
            reduceObservations(&localContext, sharedContext.xCurrent, sharedContext.globalData->sharedDX);
            atomicAdd(&sharedContext.sharedF,
                      localContext.threadF); // TODO reduce over threads, not using atomicAdd

            __syncthreads();
            // sharedContext.sharedF, sharedContext.sharedDX is complete for all threads
            approximateImplicitHessian(sharedContext.globalData->sharedDX, it, &sQueue, &yQueue, &sharedContext);
            // sharedContext.globalData->lbfgsR is set / threads
            fCurrent = sharedContext.sharedF;
            __syncthreads();
#ifdef PRINT
            if (it % FRAMESIZE == 0 && threadIdx.x == 0 && blockIdx.x == 0) {
//                printf("\n %d f: %.16f , f - fPrev: %f\n", it, sharedContext.sharedF, costDifference);
//                printf("S: ");
//                unsigned sIterator = sQueue.getIterator();
//                while (sQueue.hasNext(sIterator)) {
//                    double *s = sQueue.next(sIterator);
//                    for (int i = 0; i < X_DIM; i++) {
//                        printf("%.16f ", s[i]);
//                    }
//                    printf(",");
//                }
//
//                printf("\nY: ");
//                unsigned yIterator = yQueue.getIterator();
//                while (yQueue.hasNext(yIterator)) {
//                    double *y = yQueue.next(yIterator);
//                    for (int i = 0; i < X_DIM; i++) {
//                        printf("%.16f ", y[i]);
//                    }
//                    printf(",");
//                }
//
//                printf("\nR:");
//                for (int i = 0; i < X_DIM; i++) {
//                    printf("%.16f ", sharedContext.globalData->lbfgsR[i]);
//                }

                printf("\nxCurrent ");
                for (unsigned j = 0; j < X_DIM - 1; j++) {
                    printf("%.16f,", sharedContext.xCurrent[j]);
                }
                printf("%.16f\n", sharedContext.xCurrent[X_DIM - 1]);
                printf("f: %f\n", fCurrent);
            }
#endif

            if(LBFGSlineSearchWolfeConditions(&localContext, &sharedContext, sharedContext.globalData->sharedDX,
                                              sharedContext.globalData->lbfgsR,
                                              sharedContext.globalData->lbfgsQueueY[yQueue.back], fCurrent)!=0) {
                if(!lineSearch(&localContext, &sharedContext, sharedContext.globalData->sharedDX, fCurrent)){
                    // sharedContext.xNext contains the model for the next iteration, sharedContext.sharedDX is for sharedContext.xCurrent model
                    // sharedContext.sharedF is set for xNext
                    if(threadIdx.x == 0){
                        sharedContext.sharedF=fCurrent;
                    }
                    __syncthreads();
                    break;
                }
            }

            // sharedContext.globalData->lbfgsQueueY[yQueue.back], sharedContext.sharedF =f(xNext), xNext set
            __syncthreads();// TODO check if necessary
            // sharedContext.globalData->lbfgsR is set for all threads
            // sharedContext.xNext contains the model for the next iteration, sharedContext.globalData->lbfgsR is for sharedContext.xCurrent model
            minusNoSync(sharedContext.xNext, sharedContext.xCurrent, sharedContext.globalData->lbfgsQueueS[sQueue.back],
                        X_DIM);
            // sharedContext.globalData->lbfgsQueueS[queue.back] = xNext - xCurrent
            __syncthreads();

            // DX[xNext] in sharedContext.globalData->lbfgsQueueY[queue.back]
            // DX[xCurrent] in sharedContext.sharedDX
            minusNoSync(sharedContext.globalData->lbfgsQueueY[yQueue.back], sharedContext.globalData->sharedDX,
                        sharedContext.globalData->lbfgsQueueY[yQueue.back], X_DIM);
            // sharedContext.globalData->lbfgsQueueY[queue.back]: DX[xNext] - DX[xCurrent]
            __syncthreads();
            // sharedContext.globalData->lbfgsQueueS[queue.back] = xNext - xCurrent
            // sharedContext.globalData->lbfgsQueueY[queue.back] = DX[xNext] - DX[xCurrent]
            sQueue.enqueue(sharedContext.globalData->lbfgsQueueS[sQueue.back]);
            yQueue.enqueue(sharedContext.globalData->lbfgsQueueY[yQueue.back]);
            if (threadIdx.x == 0) {
                // sharedContext.xNext contains the model for the next iteration
                swapModels(&sharedContext);
            }

            // xCurrent<->xNext
            // Stopping criteria
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
        }
#ifdef PRINT
        if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("\nxCurrent ");
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
            // print Queues after GD

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
             double *globalF
//#ifdef GLOBAL_SHARED_MEM
            , GlobalData *globalSharedContext
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
        // every thread has a local observation loaded into local memory
        FIFOQueue sQueue = FIFOQueue();
        FIFOQueue yQueue = FIFOQueue();
        __shared__
        SharedContext sharedContext;

//#ifdef GLOBAL_SHARED_MEM
        sharedContext.globalData = &globalSharedContext[blockIdx.x];

//#else
//        // LOAD MODEL INTO SHARED MEMORY
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

        localContext.alpha = ALPHA;
        localContext.fEvaluations = 0;
        localContext.residualProblems[0] = &f1;
        localContext.residualConstants[0] = globalData;
#if defined(PROBLEM_SNLP) || defined(PROBLEM_SNLP3D)
        localContext.residualProblems[1] = &f2;
        localContext.residualConstants[1] =
                localContext.residualConstants[0] + RESIDUAL_CONSTANTS_COUNT_1 * RESIDUAL_CONSTANTS_DIM_1;
#endif
        double fCurrent;
        // every thread has a copy of the shared model loaded, and an empty localContext.Jacobian


        const double epsilon = FEPSILON;
        sharedContext.sharedDXNorm = epsilon + 1;
        int it;
        resetSharedState(&sharedContext, threadIdx.x);

        __syncthreads();
        // sharedContext.sharedF, sharedContext.sharedDX, is cleared // TODO this synchronizes over threads in a block, sync within grid required : https://on-demand.gputechconf.com/gtc/2017/presentation/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf
        reduceObservations(&localContext, sharedContext.xCurrent, sharedContext.globalData->sharedDX);
        // localContext.threadF are calculated
        atomicAdd(&sharedContext.sharedF,
                      localContext.threadF);
        // TODO reduce over threads, not using atomicAd
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
#endif //PARALLELLBFGS_LBFGS_CUH
