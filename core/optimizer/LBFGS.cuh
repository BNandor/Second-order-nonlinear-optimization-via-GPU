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
#include "../problem/SNLP.cuh"
#include "../problem/SNLPAnchor.cuh"
#include "../common/FIFOQueue.cuh"
#include <stdio.h>


namespace LBFGS {

#define LBFGS_M 5

    struct SharedContext {
        double sharedX1[X_DIM];
        double sharedX2[X_DIM];
        double sharedDX[X_DIM];
        double lbfgsQueueS[LBFGS_M][X_DIM];
        double lbfgsQueueY[LBFGS_M][X_DIM];
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
    };

    __device__
    void minusNoSync(double *a, double *b, double *result, unsigned size) {
        for (unsigned spanningTID = threadIdx.x; spanningTID < size; spanningTID += blockDim.x) {
            result[spanningTID] = a[spanningTID] - b[spanningTID];
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
                printf("finalSum %f\n", sharedContext.sharedScratchPad[itsum]);
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
            sharedContext->sharedDX[spanningTID] = 0.0;
        }
        sharedContext->sharedScratchPad[threadIdx] = 0; // TODO check if necessary
    }

    __device__
    void reduceObservations(LocalContext *localContext,
                            double *x,
                            double *dx,
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
#ifdef PROBLEM_SNLP
        SNLPAnchor *f2 = ((SNLPAnchor *) localContext->residualProblems[1]);
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
    void lineSearch(LocalContext *localContext,
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
        do {
            lineStep(sharedContext->xCurrent, sharedContext->xNext, X_DIM, sharedContext->sharedDX,
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
#ifdef PROBLEM_SNLP
            for (unsigned spanningTID = threadIdx.x;
                 spanningTID < RESIDUAL_CONSTANTS_COUNT_2; spanningTID += blockDim.x) {
                f2->setConstants(&(localContext->residualConstants[1][RESIDUAL_CONSTANTS_DIM_2 * spanningTID]),
                                 RESIDUAL_CONSTANTS_DIM_2);
                fNext += f2->eval(sharedContext->xCurrent,
                                  X_DIM)->value;
            }
#endif
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
        SNLPAnchor f2 = SNLPAnchor();
#endif
        // every thread has a local observation loaded into local memory
        FIFOQueue sQueue = FIFOQueue();
        FIFOQueue yQueue = FIFOQueue();

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
        localContext.residualConstants[0] = globalData;
#ifdef  PROBLEM_SNLP
        localContext.residualProblems[1] = &f2;
        localContext.residualConstants[1] =
                localContext.residualConstants[0] + RESIDUAL_CONSTANTS_COUNT_1 * RESIDUAL_CONSTANTS_DIM_1;
#endif
        double fCurrent;
        // every thread has a copy of the shared model loaded, and an empty localContext.Jacobian
        double costDifference = INT_MAX;

        const double epsilon = 1e-7;
        sharedContext.sharedDXNorm = epsilon + 1;
        unsigned it;

        for (it = 0; it < LBFGS_M; it++) {

            resetSharedState(&sharedContext, threadIdx.x);
            __syncthreads();
            // sharedContext.sharedF, sharedContext.sharedDX, is cleared // TODO this synchronizes over threads in a block, sync within grid required : https://on-demand.gputechconf.com/gtc/2017/presentation/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf
            reduceObservations(&localContext, sharedContext.xCurrent, sharedContext.sharedDX, globalData);
            // localContext.threadF are calculated
            atomicAdd(&sharedContext.sharedF, localContext.threadF); // TODO reduce over threads, not using atomicAdd
            __syncthreads();
            // sharedContext.sharedF, sharedContext.sharedDX is complete for all threads
            fCurrent = sharedContext.sharedF;

            if (threadIdx.x == 0) {
                printf("it: %d f: %f\n", it, fCurrent);
                sharedContext.sharedDXNorm = 0;
            }
            __syncthreads();
            // fCurrent is set, sharedDXNorm is cleared for all threads,
            lineSearch(&localContext, &sharedContext, fCurrent);
            // sharedContext.xNext contains the model for the next iteration, sharedContext.sharedDX is for sharedContext.xCurrent model
            minusNoSync(sharedContext.xNext, sharedContext.xCurrent, sharedContext.lbfgsQueueS[sQueue.back], X_DIM);
            // sharedContext.lbfgsQueueS[queue.back] = xNext - xCurrent
            reduceObservations(&localContext, sharedContext.xNext, sharedContext.lbfgsQueueY[sQueue.back], globalData);
            // DX[xNext] in sharedContext.lbfgsQueueY[queue.back]
            // DX[xCurrent] in sharedContext.sharedDX
            minusNoSync(sharedContext.lbfgsQueueY[yQueue.back], sharedContext.sharedDX,
                        sharedContext.lbfgsQueueY[yQueue.back], X_DIM);
            // sharedContext.lbfgsQueueY[queue.back]: DX[xNext] - DX[xCurrent]
            __syncthreads();
            // sharedContext.lbfgsQueueS[queue.back] = xNext - xCurrent
            // sharedContext.lbfgsQueueY[queue.back] = DX[xNext] - DX[xCurrent]
            sQueue.enqueue(sharedContext.lbfgsQueueS[sQueue.back]);
            yQueue.enqueue(sharedContext.lbfgsQueueY[yQueue.back]);
            if (threadIdx.x == 0) {
                // sharedContext.xNext contains the model for the next iteration
                swapModels(&sharedContext);
            }

            double localDXNorm = 0;
            for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
                localDXNorm += std::pow(sharedContext.sharedDX[spanningTID], 2);
                // TODO reduce over threads, not using atomicAdd
            }
            sharedContext.sharedScratchPad[threadIdx.x] = localDXNorm;
            __syncthreads();

            if (threadIdx.x == 0) {
                double localDXNormFinal = 0;
                for (unsigned itdxnorm = 0; itdxnorm < THREADS_PER_BLOCK; itdxnorm++) {
                    localDXNormFinal += sharedContext.sharedScratchPad[itdxnorm];
                }
                sharedContext.sharedDXNorm = std::sqrt(localDXNormFinal);
            }

            costDifference = std::abs(fCurrent - sharedContext.sharedF);
            __syncthreads();
            //xCurrent,xNext is set for all threads
            if (it % 5 == 0 && threadIdx.x == 0 && blockIdx.x == 0) {
                printf("xCurrent ");
                for (unsigned j = 0; j < X_DIM - 1; j++) {
                    printf("%f,", sharedContext.xCurrent[j]);
                }
                printf("%f\n", sharedContext.xCurrent[X_DIM - 1]);
            }
        }

        if (threadIdx.x == 0) {
            // print Queues after GD
            printf("S: ");
            unsigned sIterator = sQueue.getIterator();
            while (sQueue.hasNext(sIterator)) {
                double *s = sQueue.next(sIterator);
                for (int i = 0; i < X_DIM; i++) {
                    printf("%f ", s[i]);
                }
                printf(",");
            }

            printf("\nY: ");
            unsigned yIterator = yQueue.getIterator();
            while (yQueue.hasNext(yIterator)) {
                double *y = yQueue.next(yIterator);
                for (int i = 0; i < X_DIM; i++) {
                    printf("%f ", y[i]);
                }
                printf(",");
            }

            printf("\nxCurrent ");
            for (unsigned j = 0; j < X_DIM - 1; j++) {
                printf("%f,", sharedContext.xCurrent[j]);
            }
            printf("%f\n", sharedContext.xCurrent[X_DIM - 1]);
            globalF[blockIdx.x] = sharedContext.sharedF;
            printf("\nWith: %d threads in block %d after it: %d f: %.10f\n", blockDim.x, blockIdx.x, it,
                   sharedContext.sharedF);
        }
    }
}
#endif //PARALLELLBFGS_LBFGS_CUH
