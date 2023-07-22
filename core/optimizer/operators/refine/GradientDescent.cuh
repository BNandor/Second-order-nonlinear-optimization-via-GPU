//
// Created by spaceman on 2022. 01. 16..
//

#ifndef PARALLELLBFGS_GRADIENTDESCENT_CUH
#define PARALLELLBFGS_GRADIENTDESCENT_CUH

#include "../../../AD/DDouble.cuh"
#include "../../../AD/function/DFunction.cuh"
#include "../../../AD/function/DPlusFunction.cuh"
#include "../../../AD/function/DMultiplicationFunction.cuh"
#include "../../../AD/function/DMinusFunction.cuh"
#include "../../../AD/function/DIDFunction.cuh"
#include "../../../AD/function/DSquareFunction.cuh"
#include "../../../AD/function/Operations.cuh"
#include "../../../problem/F1.cuh"
#include "../../../problem/PlaneFitting.cuh"
#include "../../../problem/Rosenbrock2D.cuh"
#include "../../../problem/SNLP/SNLP.cuh"
#include "../../../problem/SNLP/SNLPAnchor.cuh"
#include "../../../common/FIFOQueue.cuh"
#include "../../../problem/SNLP/SNLP3D.cuh"
#include "../../../problem/SNLP/SNLP3DAnchor.cuh"
#include "../../../common/model/Model.cuh"
#include "../../../problem/SNLP/SNLPModel.cuh"
#include <stdio.h>

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
        double *lowerbounds;
        double *upperbounds;
        bool isBounded;
        double sharedF;
        double sharedDXNorm;
    };

    struct LocalContext {
        double threadF;
        double alpha;
        double initialAlpha;
        void* modelP;
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
                            double *x,
                            double *dx) {
        ++localContext->fEvaluations;
        localContext->threadF = 0;
        Model *model = (Model *) localContext->modelP;
        CAST_RESIDUAL_FUNCTIONS()
        COMPUTE_RESIDUALS()
    }

    __device__ void lineStep(double *x, double *xNext, unsigned xSize, double *jacobian, double alpha) {
        for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
            xNext[spanningTID] = x[spanningTID] - alpha * jacobian[spanningTID];
        }
    }

    __device__ void boundedLineStep(double *x, double *xNext,double *lowerBounds,double *upperBounds, unsigned xSize, double *jacobian, double alpha) {
        double next;
        double l;
        double u;
        for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
             next= x[spanningTID] - alpha * jacobian[spanningTID];
             l=lowerBounds[spanningTID];
            if(next < l){
                xNext[spanningTID]=l;
            }else{
                u=upperBounds[spanningTID];
                if(next > u){
                    xNext[spanningTID]=u;
                }else{
                    xNext[spanningTID]=next;
                }
            }
        }
    }

    __device__
    bool lineSearch(LocalContext *localContext,
                    SharedContext *sharedContext,
                    double currentF) {
        double fNext;
        localContext->alpha = localContext->initialAlpha;
        Model *model = (Model *) localContext->modelP;
        CAST_RESIDUAL_FUNCTIONS()
        do {
            ++localContext->fEvaluations;
            if(!sharedContext->isBounded) {
                lineStep(sharedContext->xCurrent, sharedContext->xNext, X_DIM, sharedContext->globalData->sharedDX,
                         localContext->alpha);
            }else{
                boundedLineStep(sharedContext->xCurrent, sharedContext->xNext,sharedContext->lowerbounds,sharedContext->upperbounds, X_DIM, sharedContext->globalData->sharedDX,
                         localContext->alpha);
            }
            if (threadIdx.x == 0) {
                sharedContext->sharedF = 0;
            }
            fNext = 0;
            localContext->alpha = localContext->alpha / 2;
            __syncthreads();// sharedContext.sharedF is cleared
            COMPUTE_LINESEARCH()
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
             double *globalF,
             GlobalData *globalSharedContext,
             void * model,
             int localIterations, double  alpha,
             bool isBounded, double *globalLowerBounds, double *globalUpperBounds)
             {

//        if( localIterations == 0 ) {
//            evaluateF(globalX, globalData,
//                      globalF, globalSharedContext, model
//            );
//            return;
//        }

        DEFINE_RESIDUAL_FUNCTIONS()
//        if(blockIdx.x ==0 && threadIdx.x ==0) {
//            Model *model1 = (Model *) model;
//            printf("GD iterations: %d\n", localIterations);
//            printf("residuals: %d\n", model1->residuals.residualCount);
//
//            for (int residualIndex = 0; residualIndex < model1->residuals.residualCount; residualIndex++) {
//                printf("constants: %d\n", model1->residuals.residual[residualIndex].constantsCount);
//                printf("constants dim: %d\n", model1->residuals.residual[residualIndex].constantsDim);
//                printf("chainParameters: %d\n", model1->residuals.residual[residualIndex].parametersDim);
//            }
//        }

        // every thread has a local observation loaded into local memory
        __shared__
        SharedContext sharedContext;
        sharedContext.globalData = &globalSharedContext[blockIdx.x];

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
            sharedContext.isBounded = isBounded;
            if(isBounded){
                sharedContext.lowerbounds = globalLowerBounds;
                sharedContext.upperbounds = globalUpperBounds;
            }
        }

        localContext.fEvaluations = 0;
        localContext.initialAlpha = alpha;
        localContext.alpha = localContext.initialAlpha;
        localContext.modelP=model;
        INJECT_RESIDUAL_FUNCTIONS()

        double fCurrent;
        // every thread has a copy of the shared model loaded, and an empty localContext.Jacobian
        double costDifference = INT_MAX;

        const double epsilon = FEPSILON;
        sharedContext.sharedDXNorm = epsilon + 1;
        unsigned it;
//        for (it = 0; it < ITERATION_COUNT; it++) {
        for (it = 0; localContext.fEvaluations < localIterations; it++) {

            resetSharedState(&sharedContext, threadIdx.x);
            __syncthreads();
            // sharedContext.sharedF, sharedContext.globalData->sharedX2, is cleared // TODO this synchronizes over threads in a block, sync within grid required : https://on-demand.gputechconf.com/gtc/2017/presentation/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf
            reduceObservations(&localContext, sharedContext.xCurrent, sharedContext.globalData->sharedDX);
            // localContext.threadF are calculated
            atomicAdd(&sharedContext.sharedF, localContext.threadF); // TODO reduce over threads, not using atomicAdd
            __syncthreads();
            fCurrent = sharedContext.sharedF;
            __syncthreads();
#ifdef PRINT
//            if (it % FRAMESIZE == 0 && threadIdx.x == 0 && blockIdx.x == 0) {
//                printf("xCurrent ");
//                for (unsigned j = 0; j < X_DIM - 1; j++) {
//                    printf("%f,", sharedContext.xCurrent[j]);
//                }
//                printf("%f\n", sharedContext.xCurrent[X_DIM - 1]);
//                printf("f: %f\n", fCurrent);
//            }
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

            __syncthreads();
            //xCurrent,xNext is set for all threads
        }

#ifdef PRINT
//        if (threadIdx.x == 0 && blockIdx.x == 0) {
//
//            printf("xCurrent ");
//            for (unsigned j = 0; j < X_DIM - 1; j++) {
//                printf("%f,", sharedContext.xCurrent[j]);
//            }
//            printf("%f\n", sharedContext.xCurrent[X_DIM - 1]);
//            printf("\nthreads:%d", blockDim.x);
//            printf("\niterations:%d", it);
//            printf("\nfevaluations: %d\n", localContext.fEvaluations);
//        }

#endif

        if (threadIdx.x == 0) {
            globalF[blockIdx.x] = sharedContext.sharedF;
        }
        for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
            globalX[modelStartingIndex + spanningTID]=sharedContext.xCurrent[spanningTID];
        }
    }
}
#endif //PARALLELLBFGS_GRADIENTDESCENT_CUH
