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
#include "../optimizer/initialize/Initializer.cuh"
#include <list>

class OptimizerContext {

private:


    std::list<Operator*> currentOperatorList;
public:

    // Initializer
    Initializer initializer;
    Initializer* currentInitializer;

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

    CUDAMemoryModel cudaMemoryModel;
    SNLPModel model;
    int totalFunctionEvaluations=DE_ITERATION_COUNT*ITERATION_COUNT;

    explicit OptimizerContext(DEContext &deContext,GAContext &gaContext) {
        // Initializers
        initializer=Initializer();
        currentInitializer=&initializer;

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
        lbfgsLocalSearch=LBFGSLocalSearch(ALPHA,ITERATION_COUNT);

        //Select currentLocalsearch
        if (strcmp(OPTIMIZER::name.c_str(),"GD" ) == 0) {
            printf("localSearch is GD\n");
            currentLocalSearch=&gdLocalSearch;
        }

        if (strcmp(OPTIMIZER::name.c_str(),"LBFGS" ) == 0) {
            printf("localSearch is LBFGS\n");
            currentLocalSearch = &lbfgsLocalSearch;
        }

        // Selected  operators
        currentOperatorList={currentInitializer,currentPerturbator,currentSelector,currentLocalSearch};
#ifdef SAFE
        assert(currentLocalSearch!=0);
#endif

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
