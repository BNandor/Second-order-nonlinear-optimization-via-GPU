//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_OPTIMIZERCONTEXT_CUH
#define PARALLELLBFGS_OPTIMIZERCONTEXT_CUH


class DEContext {
public:
    double crossoverRate=CR;
    double force=F;
};

class OptimizerContext{
private:
    int threadsPerBlock=THREADS_PER_BLOCK;
    int populationSize=POPULATION_SIZE;
    int blocksPerGrid=populationSize;
    DEContext differentialEvolutionContext=DEContext();

public:
    int getThreadsPerBlock() const {
        return threadsPerBlock;
    }

    int getPopulationSize() const {
        return populationSize;
    }

    int getBlocksPerGrid() const {
        return blocksPerGrid;
    }

    const DEContext &getDifferentialEvolutionContext() const {
        return differentialEvolutionContext;
    }

    int getThreadsInGrid(){
        return threadsPerBlock * blocksPerGrid;
    }

};
#endif //PARALLELLBFGS_OPTIMIZERCONTEXT_CUH
