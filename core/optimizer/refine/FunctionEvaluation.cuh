//
// Created by spaceman on 2022. 11. 06..
//

#ifndef PARALLELLBFGS_FUNCTIONEVALUATION_CUH
#define PARALLELLBFGS_FUNCTIONEVALUATION_CUH

#include "../../common/model/Model.cuh"
#include "../../common/Constants.cuh"

struct GlobalData;
namespace FuncEval {

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

    __global__ void
    evaluateF(double *globalX, double *globalData,
              double *globalF, GlobalData *globalSharedContext,
              void*model
    ) {
        DEFINE_RESIDUAL_FUNCTIONS()
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
        }
        localContext.fEvaluations = 0;
        localContext.modelP= model;
        INJECT_RESIDUAL_FUNCTIONS()
        double fCurrent;
        // every thread has a copy of the shared model loaded, and an empty localContext.Jacobian
        double costDifference = INT_MAX;

        const double epsilon = FEPSILON;
        sharedContext.sharedDXNorm = epsilon + 1;
        unsigned it;

        resetSharedState(&sharedContext, threadIdx.x);
        __syncthreads();
        // sharedContext.sharedF, sharedContext.globalData->sharedX2, is cleared // TODO this synchronizes over threads in a block, sync within grid required : https://on-demand.gputechconf.com/gtc/2017/presentation/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf
        reduceObservations(&localContext, sharedContext.xCurrent, sharedContext.globalData->sharedDX);
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
            printf("evaluate f: %f in %d\n", sharedContext.sharedF,blockIdx.x);
        }
    }
}
#endif //PARALLELLBFGS_FUNCTIONEVALUATION_CUH