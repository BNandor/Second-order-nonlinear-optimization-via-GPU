#include <iostream>
#include <iomanip>
#include <algorithm>
//#define SAFE
//#define PRINT

//#define PROBLEM_ROSENBROCK2D
//#define PROBLEM_PLANEFITTING
//#define PROBLEM_SNLP
//#define PROBLEM_SNLP3D

//#define GLOBAL_SHARED_MEM

#include "core/common/Constants.cuh"
#include "core/optimizer/refine/LBFGS.cuh"
#include "core/optimizer/refine/GradientDescent.cuh"
#include "core/common/Random.cuh"
#include "core/common/Metrics.cuh"
#include "core/optimizer/perturb/DE/DEContext.h"
#include "core/common/OptimizerContext.cuh"
#include "core/common/model/BoundedParameter.cuh"
//#include "core/optimizer/perturb/GA/GeneticAlgorithm.cu"
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <fstream>

void testOptimizer() {
    Random cudaRandom = Random();
    DEContext deContext = DEContext();
    GAContext gaContext = GAContext();

    OptimizerContext optimizerContext = OptimizerContext(deContext,gaContext);
    // Set f
    optimizerContext.model = SNLPModel(deContext);

    Metrics metrics = Metrics(optimizerContext.model); //idempotent
    optimizerContext.getCurrentLocalSearch()->setupGlobalData(optimizerContext.getModelPopulationSize()); //idempotent
    optimizerContext.cudaMemoryModel.allocateFor(optimizerContext.model); // idempotent
    optimizerContext.cudaMemoryModel.copyModelToDevice(optimizerContext.model); // idempotent

    optimizerContext.model.loadModel(optimizerContext.cudaMemoryModel.dev_x, optimizerContext.cudaMemoryModel.dev_data,
                                     metrics); // idempotent
    metrics.getCudaEventMetrics().recordStartCompute(); // idempotent
    cudaRandom.initialize(optimizerContext.getThreadsInGrid(), optimizerContext.getBlocksPerGrid(),
                          optimizerContext.getThreadsPerBlock()); // idempotent

    // EXECUTE KERNEL
    optimizerContext.cudaMemoryModel.initLoopPointers();
    optimizerContext.getCurrentPerturbator()->evaluateF(optimizerContext.cudaConfig,optimizerContext.cudaMemoryModel.dev_Model,
                                                        optimizerContext.cudaMemoryModel.dev_x1,
                                                        optimizerContext.cudaMemoryModel.dev_data,
                                                        optimizerContext.cudaMemoryModel.dev_F1);
    unsigned currentFEvaluations=1;
    unsigned currentGeneration=0;
    while(currentFEvaluations < optimizerContext.totalFunctionEvaluations) {
        //dev_F1 contains the costs of the current model
        //dev_x1 is the current model
        optimizerContext.getCurrentPerturbator()->perturb(optimizerContext.cudaConfig,
                                                          &optimizerContext.model,
                                                          optimizerContext.cudaMemoryModel.dev_Model,
                                                          optimizerContext.cudaMemoryModel.dev_x1,
                                                          optimizerContext.cudaMemoryModel.dev_x2,
                                                          optimizerContext.cudaMemoryModel.dev_data,
                                                          optimizerContext.cudaMemoryModel.dev_F1,
                                                          optimizerContext.cudaMemoryModel.dev_F2,
                                                          &cudaRandom);

        //dev_F2 contains the costs of the differential model
        //dev_x2 is the differential model
        optimizerContext.getCurrentLocalSearch()->optimize(optimizerContext.cudaMemoryModel.dev_x2, optimizerContext.cudaMemoryModel.dev_data, optimizerContext.cudaMemoryModel.dev_F2, optimizerContext.getCurrentLocalSearch()->getDevGlobalContext(),optimizerContext.cudaMemoryModel.dev_Model,optimizerContext.cudaConfig);
        //evaluated differential model into F2
        //select the best models from current and differential models
        optimizerContext.getCurrentSelector()->select(optimizerContext.cudaConfig,
                                                      optimizerContext.cudaMemoryModel.dev_x1,
                                                      optimizerContext.cudaMemoryModel.dev_x2,
                                                      optimizerContext.cudaMemoryModel.dev_F1,
                                                      optimizerContext.cudaMemoryModel.dev_F2);
        optimizerContext.getCurrentSelector()->printPopulationCostAtGeneration(optimizerContext.cudaConfig,optimizerContext.cudaMemoryModel.dev_F2,currentGeneration);

        optimizerContext.cudaMemoryModel.swapModels();
        std::for_each(optimizerContext.getCurrentOperators().begin(),optimizerContext.getCurrentOperators().end(),[&currentFEvaluations](auto op){
                currentFEvaluations+=op->fEvaluationCount();
        });

        ++currentGeneration;
        // dev_x1 contains the next models, dev_F1 contains the associated costs
    }

    metrics.getCudaEventMetrics().recordStopCompute();
    optimizerContext.cudaMemoryModel.copyModelsFromDevice(metrics.modelPerformanceMetrics);
    metrics.modelPerformanceMetrics.printBestModel(&optimizerContext.model);
    metrics.modelPerformanceMetrics.persistBestModelTo(&optimizerContext.model,std::string("finalModel") + std::string(OPTIMIZER::name) + std::string(".csv"));
    printf("\ntime ms : %f\n", metrics.getCudaEventMetrics().getElapsedKernelMilliSec());
}

int main(int argc, char** argv) {
    testOptimizer();
    return 0;
}

// TODO make x1,x2,F1,F2 consistent in every operator (i.e evaluate F2 after every perturbation) DONE

// TODO add Simulated Annealing to hyper level
// TODO as a first step, skip mutating the operator parameters

