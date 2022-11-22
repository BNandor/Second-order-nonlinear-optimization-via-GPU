//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_OPTIMIZERCONTEXT_CUH
#define PARALLELLBFGS_OPTIMIZERCONTEXT_CUH

#include "../optimizer/operators/perturb/DE/DEContext.h"
#include "../optimizer/operators/perturb/Perturbator.h"
#include "../optimizer/operators/refine/LocalSearch.cuh"
#include "config/CUDAConfig.cuh"
#include "Metrics.cuh"
#include "../optimizer/operators/select/best/BestSelector.cuh"
#include "../optimizer/operators/select/Selector.cuh"
#include "model/Model.cuh"
#include "../optimizer/operators/perturb/GA/GAContext.cuh"
#include <cstring>
#include <fstream>
#include "./model/CudaMemoryModel.cuh"
#include "../problem/SNLP/SNLPModel.cuh"
#include "../optimizer/operators/initialize/Initializer.cuh"
#include <list>

class OptimizerContext {

public:

    // Initializer
    Initializer initializer;

    // Perturbators
    DEContext differentialEvolutionContext;
    GAContext geneticAlgorithmContext;
    Perturbator* currentPerturbator;

    // Selectors
    BestSelector bestSelector;

    // Local searches
    GDLocalSearch gdLocalSearch;
    LBFGSLocalSearch lbfgsLocalSearch;

    LocalSearch* currentLocalSearch;

    CUDAMemoryModel cudaMemoryModel;
    SNLPModel model;

    explicit OptimizerContext() {
        // Initializers
        initializer=Initializer();

        // Configure perturbators
        differentialEvolutionContext=DEContext();
        geneticAlgorithmContext=GAContext();

        // Select currentPerturbator
//        currentPerturbator = &differentialEvolutionContext;
        currentPerturbator = &geneticAlgorithmContext;

        //Configure Selectors
        bestSelector=BestSelector();

        //Configure Local Searches
        gdLocalSearch=GDLocalSearch();
        lbfgsLocalSearch=LBFGSLocalSearch();

        cudaMemoryModel=CUDAMemoryModel();
        cudaMemoryModel.cudaConfig=CUDAConfig(currentPerturbator->populationSize);
    }

    int getThreadsPerBlock() const {
#ifdef SAFE
        assert(cudaMemoryModel.cudaConfig.threadsPerBlock>0);
#endif
        return cudaMemoryModel.cudaConfig.threadsPerBlock;
    }

    int getBlocksPerGrid() const {
#ifdef SAFE
        assert(cudaMemoryModel.cudaConfig.blocksPerGrid>0);
#endif
        return cudaMemoryModel.cudaConfig.blocksPerGrid;
    }

    int getThreadsInGrid() {
#ifdef SAFE
        assert(cudaMemoryModel.cudaConfig.threadsPerBlock * cudaMemoryModel.cudaConfig.blocksPerGrid>0);
#endif
        return cudaMemoryModel.cudaConfig.threadsPerBlock * cudaMemoryModel.cudaConfig.blocksPerGrid;
    }

    int getPopulationSize() const {
#ifdef SAFE
        assert(currentPerturbator->populationSize>0);
#endif
        return currentPerturbator->populationSize;
    }

    Perturbator * getCurrentPerturbator() const {
        return currentPerturbator;
    }
};
#endif //PARALLELLBFGS_OPTIMIZERCONTEXT_CUH
