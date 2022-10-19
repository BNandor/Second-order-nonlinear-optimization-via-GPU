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
#include <cstring>
#include <fstream>

class Residual {
public:
    int parametersDim;
    int constantsDim;
    int constantsCount;

    int getConstantsDim(){
        return constantsDim*constantsCount;
    }
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
    int populationSize;
    Residuals residuals;

    Model()=default;
    Model(Perturbator& perturbator) {
        modelSize=X_DIM;
        modelPopulationSize=perturbator.populationSize*modelSize;
        populationSize=perturbator.populationSize;
    }

    virtual void loadModel(void* dev_x,void* dev_constantData,Metrics &metrics)=0;

    void readPopulation(double *x, unsigned xSize, std::string filename) {
        std::fstream input;
        input.open(filename.c_str());
        if (input.is_open()) {
            unsigned cData = 0;
            while (input >> x[cData]) {
                cData++;
            }
            std::cout << "read: " << cData << " expected: " << xSize
                      << std::endl;
            assert(cData == xSize);
        } else {
            std::cerr << "err: could not open " << filename << std::endl;
            exit(1);
        }
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

    void loadModel(void* dev_x,void* dev_constantData, Metrics &metrics ) override {
        const int constantDataSize=residuals.residualDataSize();
        double x[modelPopulationSize]={};
        double data[constantDataSize]={};

        readSNLPProblem(data, PROBLEM_PATH);
        readSNLPAnchors(data + residuals.residual[0].getConstantsDim(),
                        PROBLEM_ANCHOR_PATH);
        readPopulation(x, modelPopulationSize,PROBLEM_INPUT_POPULATION_PATH);

        metrics.getCudaEventMetrics().recordStartCopy();
        cudaMemcpy(dev_x, &x, modelPopulationSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_constantData, &data, constantDataSize * sizeof(double), cudaMemcpyHostToDevice);
        metrics.getCudaEventMetrics().recordStopCopy();
    }

    void readSNLPProblem(double *data, std::string filename) {
        std::fstream input;
        input.open(filename.c_str());
        if (input.is_open()) {
            unsigned cData = 0;
            while (input >> data[cData]) {
                cData++;
            }
            std::cout << "read: " << cData << " expected: " << residuals.residual[0].getConstantsDim()
                      << std::endl;
            assert(cData == residuals.residual[0].getConstantsDim());
        } else {
            std::cerr << "err: could not open " << filename << std::endl;
            exit(1);
        }
    }

    void readSNLPAnchors(double *data, std::string filename) {
        std::fstream input;
        input.open(filename.c_str());
        if (input.is_open()) {
            unsigned cData = 0;
            while (input >> data[cData]) {
                cData++;
            }
            std::cout << "read: " << cData << " expected: " << residuals.residual[1].getConstantsDim()
                      << std::endl;
            assert(cData == residuals.residual[1].getConstantsDim());
        } else {
            std::cerr << "err: could not open " << filename << std::endl;
            exit(1);
        }
    }
};

class CUDAMemoryModel{
public:
    double *dev_x;
    double *dev_xDE;
    double *dev_x1;
    double *dev_x2;
    double *dev_data;
    double *dev_F;
    double *dev_FDE;
    double *dev_F1;
    double *dev_F2;

    void allocateFor(Model &model) {
        cudaMalloc((void **) &dev_x, model.modelPopulationSize * sizeof(double));
        cudaMalloc((void **) &dev_xDE, model.modelPopulationSize * sizeof(double));
        cudaMalloc((void **) &dev_data, model.residuals.residualDataSize() * sizeof(double));
        cudaMalloc((void **) &dev_F, model.populationSize * sizeof(double));
        cudaMalloc((void **) &dev_FDE, model.populationSize * sizeof(double));
    }

    ~CUDAMemoryModel(){

        if(dev_x!=nullptr){
            cudaFree(dev_x);
        }
        if(dev_xDE!=nullptr){
            cudaFree(dev_xDE);
        }
        if(dev_x1!=nullptr){
            cudaFree(dev_x1);
        }
        if(dev_x2!=nullptr){
            cudaFree(dev_x2);
        }
        if(dev_data!=nullptr){
            cudaFree(dev_data);
        }
        if(dev_F!=nullptr){
            cudaFree(dev_F);
        }
        if(dev_FDE!=nullptr){
            cudaFree(dev_FDE);
        }
        if(dev_F1!=nullptr){
            cudaFree(dev_F1);
        }
        if(dev_F2!=nullptr){
            cudaFree(dev_F2);
        }
    }
};


class OptimizerContext {

private:
    // Perturbators
    DEContext differentialEvolutionContext;

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

    explicit OptimizerContext(DEContext &deContext) {
        // Configure perturbators
        differentialEvolutionContext=deContext;

        // Select currentPerturbator
        currentPerturbator = &differentialEvolutionContext;

        //Configure Selectors
        bestSelector=BestSelector();

        // Select currentSelector
        currentSelector=&bestSelector;

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
