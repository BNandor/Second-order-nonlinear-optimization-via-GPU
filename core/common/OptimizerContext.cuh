//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_OPTIMIZERCONTEXT_CUH
#define PARALLELLBFGS_OPTIMIZERCONTEXT_CUH



#include "../optimizer/perturb/DE/DEContext.h"
#include "../optimizer/perturb/Perturbator.h"
#include "../optimizer/refine/LocalSearch.cuh"
#include "config/CUDAConfig.cuh"
#include <cstring>

class Residual {
public:
    int parametersDim;
    int constantsDim;
    int constantsCount;
};

class Residuals {
public:

    int residualCount;
    Residual *residual;

    int residualDataSize() const {
        int size=0;
#ifdef SAFE
        assert(residualCount>0);
        assert(residual!=0);
#endif
        printf("pointer %p\n",residual);
        Residual* it=residual;
        for(int i=0;i<residualCount;i++) {
            size+=(it->constantsCount)*(it->constantsDim);
            it++;
            printf("count %i\n",it->constantsCount);
            printf("dim %i\n",it->constantsDim);
        }

#ifdef SAFE
        assert(size>0);
#endif
        return size;
    }
};

class Model {
public:
    int modelSize;
    int modelPopulationSize;
    Residuals residuals;

    Model()=default;
    Model(Perturbator& perturbator) {
        modelSize=X_DIM;
        modelPopulationSize=perturbator.populationSize*modelSize;
    }
};

class SNLPModel: public Model {
    Residual SNLPResidual[2]{};
public:
    SNLPModel():Model(){};
    explicit SNLPModel(Perturbator& perturbator) :Model(perturbator) {
        residuals.residualCount=2;
        SNLPResidual[0].constantsCount=RESIDUAL_CONSTANTS_COUNT_1;
        SNLPResidual[0].constantsDim=RESIDUAL_CONSTANTS_DIM_1;
        SNLPResidual[0].parametersDim=RESIDUAL_PARAMETERS_DIM_1;
        SNLPResidual[1].constantsCount=RESIDUAL_CONSTANTS_COUNT_2;
        SNLPResidual[1].constantsDim=RESIDUAL_CONSTANTS_DIM_2;
        SNLPResidual[1].parametersDim=RESIDUAL_PARAMETERS_DIM_2;
        residuals.residual= reinterpret_cast<Residual *>(&SNLPResidual[0]);
    }
};


class OptimizerContext {

private:
    CUDAConfig cudaConfig;
    DEContext differentialEvolutionContext;
    Perturbator* currentPerturbator;
    GDLocalSearch gdLocalSearch;
    LBFGSLocalSearch lbfgsLocalSearch;
    LocalSearch* currentLocalSearch;

public:
    SNLPModel model;
    explicit OptimizerContext(DEContext &deContext) {
        // Configure perturbators
        differentialEvolutionContext=deContext;

        // Select currentPerturbator
        currentPerturbator = &differentialEvolutionContext;

        //Configure Local Searches
        gdLocalSearch=GDLocalSearch();
        lbfgsLocalSearch=LBFGSLocalSearch();

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

    int getResidualDataSize() {
        return model.residuals.residualDataSize();
    }

    LocalSearch* getCurrentLocalSearch(){
        return currentLocalSearch;
    }

    CUDAConfig getCUDAConfig(){
        return  cudaConfig;
    }
};
#endif //PARALLELLBFGS_OPTIMIZERCONTEXT_CUH
