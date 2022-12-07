//
// Created by spaceman on 2022. 11. 26..
//

#ifndef PARALLELLBFGS_BASELEVEL_CUH
#define PARALLELLBFGS_BASELEVEL_CUH
#include <iomanip>
#include <algorithm>
#include "../../problem/SNLP/SNLPModel.cuh"
#include "../../problem/Rosenbrock/RosenbrockModel.cuh"
#include "../../common/Constants.cuh"
#include "../../optimizer/operators/refine/LBFGS.cuh"
#include "../../optimizer/operators/refine/GradientDescent.cuh"
#include "../../common/Random.cuh"
#include "../../common/Metrics.cuh"
#include "../../optimizer/operators/perturb/DE/DEContext.h"
#include "../../optimizer/operators/perturb/GA/GAContext.cuh"
#include "../../common/OptimizerContext.cuh"
#include "../../common/model/BoundedParameter.cuh"
#include "../../optimizer/markov/optimizer/OptimizingMarkovChain.cuh"
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <fstream>
#include <utility>

class BaseLevel {
    OptimizerContext optimizerContext=OptimizerContext();
    Metrics metrics;
    Metrics globalMetrics;
public:
    void init() {
#ifdef PROBLEM_SNLP
        optimizerContext.model =new  SNLPModel(optimizerContext.differentialEvolutionContext);
#endif
#ifdef PROBLEM_ROSENBROCK
        optimizerContext.model =new  RosenbrockModel(optimizerContext.differentialEvolutionContext);
#endif
        optimizerContext.cudaMemoryModel.allocateFor(*optimizerContext.model);
        optimizerContext.cudaMemoryModel.copyModelToDevice(*optimizerContext.model);
        optimizerContext.lbfgsLocalSearch.setupGlobalData(optimizerContext.getPopulationSize());
        optimizerContext.gdLocalSearch.setupGlobalData(optimizerContext.getPopulationSize());
        metrics=*(new Metrics(*optimizerContext.model,&optimizerContext.cudaMemoryModel.cudaConfig));
        globalMetrics=*(new Metrics(*optimizerContext.model,&optimizerContext.cudaMemoryModel.cudaConfig));
    }

    ~BaseLevel(){
        delete optimizerContext.model;
    }

    void loadInitialModel(){
        optimizerContext.model->loadModel(optimizerContext.cudaMemoryModel.dev_x,optimizerContext.cudaMemoryModel.dev_xDE, optimizerContext.cudaMemoryModel.dev_data,
                                         metrics);
    }

    double optimize(std::unordered_map<std::string,OperatorParameters*>* optimizerParameters,int totalFunctionEvaluations) {
        //        cudaDeviceReset();
        // Set f
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
        markovChain.setParameters(optimizerParameters,&optimizerContext);
        while(metrics.modelPerformanceMetrics.fEvaluations < totalFunctionEvaluations) {
            markovChain.operate();
            cudaDeviceSynchronize();
            markovChain.hopToNext();
        }
        markovChain.selectBest();
        cudaDeviceSynchronize();
        metrics.getCudaEventMetrics().recordStopCompute();
        optimizerContext.cudaMemoryModel.copyModelsFromDevice(metrics.modelPerformanceMetrics);
//        metrics.printFinalMetrics();
//        printCurrentBestModel();
        cudaDeviceSynchronize();
        return metrics.modelPerformanceMetrics.bestModelCost();
    }

    void updateCurrentBestGlobalModel(){
        optimizerContext.cudaMemoryModel.copyModelsFromDevice(globalMetrics.modelPerformanceMetrics);
    }

    void persistCurrentBestModel() {
        metrics.modelPerformanceMetrics.persistBestModelTo(optimizerContext.model,std::string("finalModel-Hyper")+ std::string(".csv"));
    }

    void printCurrentBestModel() {
        metrics.modelPerformanceMetrics.printBestModel(optimizerContext.model);
    }

    void printCurrentBestGlobalModel() {
        globalMetrics.modelPerformanceMetrics.printBestModel(optimizerContext.model);
    }


};
#endif //PARALLELLBFGS_BASELEVEL_CUH
