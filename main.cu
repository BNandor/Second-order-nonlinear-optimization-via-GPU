#include <iostream>
#include <iomanip>
#include <algorithm>

#include "core/common/Constants.cuh"
#include "core/optimizer/operators/refine/LBFGS.cuh"
#include "core/optimizer/operators/refine/GradientDescent.cuh"
#include "core/common/Random.cuh"
#include "core/common/Metrics.cuh"
#include "core/optimizer/operators/perturb/DE/DEContext.h"
#include "core/optimizer/operators/perturb/GA/GAContext.cuh"
#include "core/common/OptimizerContext.cuh"
#include "core/common/model/BoundedParameter.cuh"
#include "core/optimizer/markov/optimizer/OptimizingMarkovChain.cuh"
#include "core/problem/SNLP/SNLPModel.cuh"
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <fstream>
#include <utility>


double baseLevel(std::unordered_map<std::string,OperatorParameters*>& optimizerParameters,int totalFunctionEvaluations) {
    OptimizerContext optimizerContext = OptimizerContext();
    // Set f
    optimizerContext.model = SNLPModel(optimizerContext.differentialEvolutionContext);
    optimizerContext.lbfgsLocalSearch.setupGlobalData(optimizerContext.getPopulationSize());
    optimizerContext.gdLocalSearch.setupGlobalData(optimizerContext.getPopulationSize());
    optimizerContext.cudaMemoryModel.allocateFor(optimizerContext.model);
    optimizerContext.cudaMemoryModel.copyModelToDevice(optimizerContext.model);

    Metrics metrics = Metrics(optimizerContext.model,&optimizerContext.cudaMemoryModel.cudaConfig);
    optimizerContext.model.loadModel(optimizerContext.cudaMemoryModel.dev_x, optimizerContext.cudaMemoryModel.dev_data,
                                     metrics);
    metrics.getCudaEventMetrics().recordStartCompute();
    optimizerContext.cudaMemoryModel.cudaRandom.initialize(optimizerContext.getThreadsInGrid(), optimizerContext.getBlocksPerGrid(),
                          optimizerContext.getThreadsPerBlock());

    // EXECUTE KERNEL
    optimizerContext.cudaMemoryModel.initLoopPointers();

    optimizerContext.getCurrentPerturbator()->evaluateF(optimizerContext.cudaMemoryModel.cudaConfig,optimizerContext.cudaMemoryModel.dev_Model,
                                                        optimizerContext.cudaMemoryModel.dev_x1,
                                                        optimizerContext.cudaMemoryModel.dev_data,
                                                        optimizerContext.cudaMemoryModel.dev_F1);
    metrics.modelPerformanceMetrics.fEvaluations=1;
    metrics.modelPerformanceMetrics.markovIterations=0;
    OptimizingMarkovChain markovChain=OptimizingMarkovChain(&optimizerContext, &metrics);
    markovChain.setParameters(std::move(optimizerParameters),&optimizerContext);
    while(metrics.modelPerformanceMetrics.fEvaluations < totalFunctionEvaluations) {
        markovChain.operate();
        markovChain.hopToNext();
    }
    cudaDeviceSynchronize();
    markovChain.printParameters();
    metrics.getCudaEventMetrics().recordStopCompute();
    optimizerContext.cudaMemoryModel.copyModelsFromDevice(metrics.modelPerformanceMetrics);
    metrics.modelPerformanceMetrics.printBestModel(&optimizerContext.model);
    metrics.modelPerformanceMetrics.persistBestModelTo(&optimizerContext.model,std::string("finalModel-Hyper")+ std::string(".csv"));
    metrics.printFinalMetrics();
    return metrics.modelPerformanceMetrics.bestModelCost();
}

void setRandomUniform(std::unordered_map<std::string,OperatorParameters*> &chainParameters) {
    std::for_each(chainParameters.begin(),chainParameters.end(),[](auto& operatorParameter){
        std::get<1>(operatorParameter)->setRandomUniform();
    });
}

std::unordered_map<std::string,OperatorParameters*> createOptimizerParameters();
int main(int argc, char** argv) {
    int totalFunctionEvaluations=DE_ITERATION_COUNT*ITERATION_COUNT;
    std::unordered_map<std::string,OperatorParameters*> defaultParameters= createOptimizerParameters();
    setRandomUniform(defaultParameters);
    baseLevel(defaultParameters,totalFunctionEvaluations);
    return 0;
}


std::unordered_map<std::string,OperatorParameters*> createOptimizerParameters() {
    auto chainParameters=std::unordered_map<std::string,OperatorParameters*>();
    // Optimizer Chain simplex
    chainParameters["OptimizerChainInitializerSimplex"]=new SimplexParameters(
            {
                    {std::string("perturbator"),BoundedParameter(1.0,0,1)},
                    {std::string("refiner"),BoundedParameter(0.0,0,1)}
            });
    chainParameters["OptimizerChainPerturbatorSimplex"]=new SimplexParameters(
            {
                    {std::string("selector"),BoundedParameter(0.5,0,1)},
                    {std::string("refiner"),BoundedParameter(0.5,0,1)}
            });
    chainParameters["OptimizerChainRefinerSimplex"]=new SimplexParameters(
            {
                    {std::string("selector"),BoundedParameter(1.0,0,1)},
                    {std::string("refiner"),BoundedParameter(0.0,0,1)}
            });
    chainParameters["OptimizerChainSelectorSimplex"]=new SimplexParameters(
            {
                    {std::string("perturbator"),BoundedParameter(1.0,0,1)}
            });

    // Operator Chain simplex
    chainParameters["PerturbatorInitializerSimplex"]=new SimplexParameters(
            {
                    {std::string("DE"),BoundedParameter(0.5,0,1)},
                    {std::string("GA"),BoundedParameter(0.5,0,1)}
            });
    chainParameters["PerturbatorDESimplex"]=new SimplexParameters(
            {
                    {std::string("DE"),BoundedParameter(0.5,0,1)},
                    {std::string("GA"),BoundedParameter(0.5,0,1)}
            });
    chainParameters["PerturbatorGASimplex"]=new SimplexParameters(
            {
                    {std::string("DE"),BoundedParameter(0.5,0,1)},
                    {std::string("GA"),BoundedParameter(0.5,0,1)}
            });

    chainParameters["RefinerInitializerSimplex"]=new SimplexParameters(
            {
                    {std::string("GD"),BoundedParameter(0.5,0,1)},
                    {std::string("LBFGS"),BoundedParameter(0.5,0,1)}
            });
    chainParameters["RefinerGDSimplex"]=new SimplexParameters(
            {
                    {std::string("GD"),BoundedParameter(0.5,0,1)},
                    {std::string("LBFGS"),BoundedParameter(0.5,0,1)}
            });
    chainParameters["RefinerLBFGSSimplex"]=new SimplexParameters(
            {
                    {std::string("GD"),BoundedParameter(0.5,0,1)},
                    {std::string("LBFGS"),BoundedParameter(0.5,0,1)}
            });
    chainParameters["SelectorInitializerSimplex"]=new SimplexParameters(
            {
                    {std::string("best"),BoundedParameter(1.0,0,1)}
            });
    chainParameters["SelectorBestSimplex"]=new SimplexParameters(
            {
                    {std::string("best"),BoundedParameter(1.0,0,1)}
            });

    // Operator parameters
    // Perturbator parameters
    std::unordered_map<std::string,BoundedParameter> deParams=std::unordered_map<std::string,BoundedParameter>();
    deParams["DE_CR"]=BoundedParameter(0.99,0.9,1.0);
    deParams["DE_FORCE"]=BoundedParameter(0.6,0.4,0.7);
    chainParameters["PerturbatorDEOperatorParams"]=new OperatorParameters(deParams);

    std::unordered_map<std::string,BoundedParameter> gaParams=std::unordered_map<std::string,BoundedParameter>();
    gaParams["GA_CR"]=BoundedParameter(0.9, 0.0, 1.0);
    gaParams["GA_CR_POINT"]=BoundedParameter(0.5, 0.0, 1.0);
    gaParams["GA_MUTATION_RATE"]=BoundedParameter(0.5, 0.0, 1.0);
    gaParams["GA_MUTATION_SIZE"]=BoundedParameter(50, 0.0, 100000);
    gaParams["GA_PARENTPOOL_RATIO"]=BoundedParameter(0.3, 0.2, 1.0);
    gaParams["GA_ALPHA"]=BoundedParameter(0.2, 0.0, 1.0);
    chainParameters["PerturbatorGAOperatorParams"]=new OperatorParameters(gaParams);

    // Refiner parameters
    std::unordered_map<std::string,BoundedParameter> gdParams=std::unordered_map<std::string,BoundedParameter>();
    gdParams["GD_ALPHA"]=BoundedParameter(ALPHA, 0.5, 5);
    gdParams["GD_FEVALS"]=BoundedParameter(ITERATION_COUNT, 1,ITERATION_COUNT);
//        gdParams["GD_ALPHA"]=BoundedParameter(ALPHA, 0.5, 100);
//        gdParams["GD_FEVALS"]=BoundedParameter(ITERATION_COUNT, 0, optimizerContext->to
    chainParameters["RefinerGDOperatorParams"]=new OperatorParameters(gdParams);

    std::unordered_map<std::string,BoundedParameter> lbfgsParams=std::unordered_map<std::string,BoundedParameter>();
    lbfgsParams["LBFGS_ALPHA"]=BoundedParameter(ALPHA, 0.5, 5);
    lbfgsParams["LBFGS_FEVALS"]=BoundedParameter(ITERATION_COUNT, 1,ITERATION_COUNT);
    lbfgsParams["LBFGS_C1"]=BoundedParameter(0.0001, 0.0, 0.1);
    lbfgsParams["LBFGS_C2"]=BoundedParameter(0.9, 0.8, 1.0);

//        lbfgsParams["LBFGS_ALPHA"]=BoundedParameter(ALPHA, 0.5, 100);
//        lbfgsParams["LBFGS_FEVALS"]=BoundedParameter(ITERATION_COUNT, 0, optimizerContext->totalFunctionEvaluations/100);
//        lbfgsParams["LBFGS_C1"]=BoundedParameter(0.0001, 0.0, 1.0);
//        lbfgsParams["LBFGS_C2"]=BoundedParameter(0.9, 0.0, 1.0);
    chainParameters["RefinerLBFGSOperatorParams"]=new OperatorParameters(lbfgsParams);
    return chainParameters;
}

// TODO add Simulated Annealing to hyper level

