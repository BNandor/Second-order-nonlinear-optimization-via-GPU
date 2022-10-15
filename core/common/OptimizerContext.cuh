//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_OPTIMIZERCONTEXT_CUH
#define PARALLELLBFGS_OPTIMIZERCONTEXT_CUH



#include "../optimizer/perturb/DE/DEContext.h"
#include "../optimizer/perturb/Perturbator.h"

class Model {
public:
    int modelSize;
    int modelPopulationSize;

    Model()=default;
    Model(Perturbator& perturbator) {
        modelSize=X_DIM;
        modelPopulationSize=perturbator.populationSize*modelSize;
    }
};

struct CUDAConfig {
    int threadsPerBlock=THREADS_PER_BLOCK;
    int blocksPerGrid;
    CUDAConfig()=default;
    CUDAConfig(Perturbator& perturbator){
        blocksPerGrid=perturbator.populationSize;
    }
};

class OptimizerContext{

private:
    CUDAConfig cudaConfig;
    DEContext differentialEvolutionContext;
    Perturbator* currentPerturbator;
    Model model;

public:

    explicit OptimizerContext(DEContext &deContext) {
        // Configure perturbators
        differentialEvolutionContext=deContext;

        // Select currentPerturbator
        currentPerturbator = &differentialEvolutionContext;

        model=Model(*currentPerturbator);
        cudaConfig=CUDAConfig(*currentPerturbator);
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

};
#endif //PARALLELLBFGS_OPTIMIZERCONTEXT_CUH
