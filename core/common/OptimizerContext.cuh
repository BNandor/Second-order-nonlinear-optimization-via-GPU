//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_OPTIMIZERCONTEXT_CUH
#define PARALLELLBFGS_OPTIMIZERCONTEXT_CUH

#include "../optimizer/perturb/DE/DEContext.h"
#include "../optimizer/perturb/Perturbator.h"
#include "../optimizer/refine/LocalSearch.cuh"
#include "config/CUDAConfig.cuh"
#include "Metrics.cuh"
#include "../optimizer/select/best/BestSelector.cuh"
#include "../optimizer/select/Selector.cuh"
#include "model/Model.cuh"
#include "../optimizer/perturb/GA/GAContext.cuh"
#include <cstring>
#include <fstream>
#include "./model/CudaMemoryModel.cuh"
#include "../problem/SNLP/SNLPModel.cuh"

class OptimizerContext {

private:
    // Perturbators
    DEContext differentialEvolutionContext;
    GAContext geneticAlgorithmContext;
    Perturbator* currentPerturbator;

    // Selectors
    BestSelector bestSelector;

    Selector* currentSelector;

    // Local searches
    GDLocalSearch gdLocalSearch;
    LBFGSLocalSearch lbfgsLocalSearch;

    LocalSearch* currentLocalSearch;

public:
    CUDAConfig cudaConfig;
    CUDAMemoryModel cudaMemoryModel;
    SNLPModel model;
    int totalIterations=DE_ITERATION_COUNT;

    explicit OptimizerContext(DEContext &deContext,GAContext &gaContext) {
        // Configure perturbators
        differentialEvolutionContext=deContext;
        geneticAlgorithmContext=gaContext;
        // Select currentPerturbator
//        currentPerturbator = &differentialEvolutionContext;
        currentPerturbator = &geneticAlgorithmContext;

        //Configure Selectors
        bestSelector=BestSelector();

        // Select currentSelector
        currentSelector=&bestSelector;

        //Configure Local Searches
        gdLocalSearch=GDLocalSearch(ALPHA,ITERATION_COUNT);
        lbfgsLocalSearch=LBFGSLocalSearch(ITERATION_COUNT);

        //Select currentLocalsearch
        if (strcmp(OPTIMIZER::name.c_str(),"GD" ) == 0) {
            printf("localSearch is GD\n");
            currentLocalSearch=&gdLocalSearch;
        }

        if (strcmp(OPTIMIZER::name.c_str(),"LBFGS" ) == 0) {
            printf("localSearch is LBFGS\n");
            currentLocalSearch = &lbfgsLocalSearch;
        }

#ifdef SAFE
        assert(currentLocalSearch!=0);
#endif

        cudaConfig=CUDAConfig(currentPerturbator->populationSize);
        cudaMemoryModel=CUDAMemoryModel();
    }

    int getThreadsPerBlock() const {
#ifdef SAFE
        assert(cudaConfig.threadsPerBlock>0);
#endif
        return cudaConfig.threadsPerBlock;
    }

    int getBlocksPerGrid() const {
#ifdef SAFE
        assert(cudaConfig.blocksPerGrid>0);
#endif
        return cudaConfig.blocksPerGrid;
    }

    int getThreadsInGrid() {
#ifdef SAFE
        assert(cudaConfig.threadsPerBlock * cudaConfig.blocksPerGrid>0);
#endif
        return cudaConfig.threadsPerBlock * cudaConfig.blocksPerGrid;
    }

    int getPopulationSize() const {
#ifdef SAFE
        assert(currentPerturbator->populationSize>0);
#endif
        return currentPerturbator->populationSize;
    }

    int getModelPopulationSize() {
#ifdef SAFE
        assert(model.modelPopulationSize>0);
#endif
        return model.modelPopulationSize;
    }

    int getResidualDataSize() {
        return model.residuals.residualDataSize();
    }

    Perturbator *getCurrentPerturbator() const {
        return currentPerturbator;
    }

    Selector *getCurrentSelector() const {
        return currentSelector;
    }

    LocalSearch* getCurrentLocalSearch() {
        return currentLocalSearch;
    }
};
#endif //PARALLELLBFGS_OPTIMIZERCONTEXT_CUH
