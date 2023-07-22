//
// Created by spaceman on 2022. 11. 26..
//

#ifndef PARALLELLBFGS_BASELEVEL_CUH
#define PARALLELLBFGS_BASELEVEL_CUH
#include <iomanip>
#include <algorithm>
#include "../../problem/SNLP/SNLPModel.cuh"
#include "../../problem/Rosenbrock/RosenbrockModel.cuh"
#include "../../problem/StyblinskiTang/StyblinskiTangModel.cuh"
#include "../../problem/Trid/TridModel.cuh"
#include "../../problem/Rastrigin/RastriginModel.cuh"
#include "../../problem/Schwefel/223/Schwefel223Model.cuh"
#include "../../problem/Qing/QingModel.cuh"
#include "../../problem/Michalewicz/MichalewiczModel.cuh"
#include "../../problem/DixonPrice/DixonPriceModel.cuh"
#include "../../problem/Levy/LevyModel.cuh"
#include "../../problem/Schwefel/SchwefelModel.cuh"
#include "../../problem/SumSquares/SumSquaresModel.cuh"
#include "../../problem/Sphere/SphereModel.cuh"
#include "../../common/Constants.cuh"
#include "../../optimizer/operators/refine/LBFGS.cuh"
#include "../../optimizer/operators/refine/GradientDescent.cuh"
#include "../../common/Random.cuh"
#include "../../common/Metrics.cuh"
#include "../../optimizer/operators/perturb/DE/DEContext.h"
#include "../../optimizer/operators/perturb/GA/GAContext.cuh"
#include "../../optimizer/operators/perturb/GWO/GWOContext.h"
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
    std::string  problemId;
    int totalEvaluations;
    void init(json& HHlogJson) {
#ifdef PROBLEM_SNLP
        optimizerContext.model =new  SNLPModel(optimizerContext.differentialEvolutionContext);
        problemId="SNLP";
#endif
#ifdef PROBLEM_ROSENBROCK
        optimizerContext.model =new  RosenbrockModel(optimizerContext.differentialEvolutionContext);
        problemId="ROSENBROCK";
#endif
#ifdef PROBLEM_STYBLINSKITANG
        optimizerContext.model =new  StyblinskiTangModel(optimizerContext.differentialEvolutionContext);
        problemId="STYBLINSKITANG";
#endif
#ifdef PROBLEM_TRID
        optimizerContext.model =new  TridModel(optimizerContext.differentialEvolutionContext);
        problemId="TRID";
#endif
#ifdef PROBLEM_RASTRIGIN
        optimizerContext.model =new  RastriginModel(optimizerContext.differentialEvolutionContext);
        problemId="RASTRIGIN";
#endif
#ifdef PROBLEM_SCHWEFEL223
        optimizerContext.model =new  Schwefel223Model(optimizerContext.differentialEvolutionContext);
        problemId="SCHWEFEL223";
#endif
#ifdef PROBLEM_QING
        optimizerContext.model =new  QingModel(optimizerContext.differentialEvolutionContext);
        problemId="QING";
#endif
#ifdef PROBLEM_MICHALEWICZ
        optimizerContext.model =new MichalewiczModel(optimizerContext.differentialEvolutionContext);
        problemId="MICHALEWICZ";
#endif
#ifdef PROBLEM_DIXONPRICE
        optimizerContext.model = new DixonPriceModel(optimizerContext.differentialEvolutionContext);
        problemId="DIXONPRICE";
#endif
#ifdef  PROBLEM_LEVY
        optimizerContext.model = new LevyModel(optimizerContext.differentialEvolutionContext);
        problemId="LEVY";
#endif
#ifdef  PROBLEM_SCHWEFEL
        optimizerContext.model = new SchwefelModel(optimizerContext.differentialEvolutionContext);
        problemId="SCHWEFEL";
#endif
#ifdef PROBLEM_SUMSQUARES
        optimizerContext.model = new SumSquaresModel(optimizerContext.differentialEvolutionContext);
        problemId="SUMSQUARES";
#endif
#ifdef  PROBLEM_SPHERE
        optimizerContext.model = new SphereModel(optimizerContext.differentialEvolutionContext);
        problemId="SPHERE";
#endif


        optimizerContext.cudaMemoryModel.allocateFor(*optimizerContext.model);
        optimizerContext.cudaMemoryModel.copyModelToDevice(*optimizerContext.model);
        optimizerContext.lbfgsLocalSearch.setupGlobalData(optimizerContext.getPopulationSize());
        optimizerContext.gdLocalSearch.setupGlobalData(optimizerContext.getPopulationSize());
        metrics=*(new Metrics(*optimizerContext.model,&optimizerContext.cudaMemoryModel.cudaConfig));
        globalMetrics=*(new Metrics(*optimizerContext.model,&optimizerContext.cudaMemoryModel.cudaConfig));
        totalEvaluations=0;
        initHHLogJson(HHlogJson);
    }

    void initHHLogJson(json& logJson){
        logJson["baseLevel-problemId"]=problemId;
        logJson["baseLevel-popSize"]=popSize();
        logJson["baseLevel-xDim"]=xdim();
    }
    ~BaseLevel(){
        delete optimizerContext.model;
    }

    void loadInitialModel(){
        optimizerContext.model->loadModel(optimizerContext.cudaMemoryModel.dev_x,optimizerContext.cudaMemoryModel.dev_xDE, optimizerContext.cudaMemoryModel.dev_data,
                                         metrics,&optimizerContext.cudaMemoryModel);
    }

    double optimize(std::unordered_map<std::string,OperatorParameters*>* optimizerParameters,int maxFunctionEvaluations) {
        //        cudaDeviceReset();
        std::unordered_map<std::string,OperatorParameters*> currentParameters=std::unordered_map<std::string,OperatorParameters*>();
        cloneParameters(*optimizerParameters,currentParameters);
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
        optimizerContext.getCurrentPerturbator()->evaluateF(optimizerContext.cudaMemoryModel.cudaConfig,optimizerContext.cudaMemoryModel.dev_Model,
                                                            optimizerContext.cudaMemoryModel.dev_x2,
                                                            optimizerContext.cudaMemoryModel.dev_data,
                                                            optimizerContext.cudaMemoryModel.dev_F2);
        metrics.modelPerformanceMetrics.fEvaluations=4;
        metrics.modelPerformanceMetrics.markovIterations=0;
        OptimizingMarkovChain markovChain=OptimizingMarkovChain(&optimizerContext, &metrics);
        markovChain.setParameters(&currentParameters,&optimizerContext);
        while(metrics.modelPerformanceMetrics.fEvaluations < maxFunctionEvaluations) {
            markovChain.operate(maxFunctionEvaluations);
            cudaDeviceSynchronize();
            markovChain.hopToNext();
        }
        optimizerContext.getCurrentPerturbator()->evaluateF(optimizerContext.cudaMemoryModel.cudaConfig,optimizerContext.cudaMemoryModel.dev_Model,
                                                            optimizerContext.cudaMemoryModel.dev_x2,
                                                            optimizerContext.cudaMemoryModel.dev_data,
                                                            optimizerContext.cudaMemoryModel.dev_F2);
        optimizerContext.getCurrentPerturbator()->evaluateF(optimizerContext.cudaMemoryModel.cudaConfig,optimizerContext.cudaMemoryModel.dev_Model,
                                                            optimizerContext.cudaMemoryModel.dev_x1,
                                                            optimizerContext.cudaMemoryModel.dev_data,
                                                            optimizerContext.cudaMemoryModel.dev_F1);
        markovChain.selectBest();
        cudaDeviceSynchronize();
        metrics.getCudaEventMetrics().recordStopCompute();
        optimizerContext.cudaMemoryModel.copyModelsFromDevice(metrics.modelPerformanceMetrics);
//        metrics.printFinalMetrics();
//        printCurrentBestModel();
        cudaDeviceSynchronize();

        double currentBestF= metrics.modelPerformanceMetrics.updateBestModelCost();
        if(currentBestF<globalMetrics.modelPerformanceMetrics.minF){
            updateCurrentBestGlobalModel();
        }
        totalEvaluations+=metrics.modelPerformanceMetrics.fEvaluations;
        freeOperatorParamMap(currentParameters);
        return currentBestF;
    }

    void cloneParameters(std::unordered_map<std::string,OperatorParameters*> &from,std::unordered_map<std::string,OperatorParameters*> &to){
        std::for_each(from.begin(),from.end(),[&to,&from](auto& operatorParameter) {
            if(to.count(std::get<0>(operatorParameter))>0) {
                delete to[std::get<0>(operatorParameter)];
            }
            to[std::get<0>(operatorParameter)]=std::get<1>(operatorParameter)->clone();
        });
    }


    void freeOperatorParamMap(std::unordered_map<std::string,OperatorParameters*> &parameters){
        std::for_each(parameters.begin(),parameters.end(),[](auto keyParameter){
            delete std::get<1>(keyParameter);
        });
    }

    void updateCurrentBestGlobalModel(){
        cudaDeviceSynchronize();
        optimizerContext.cudaMemoryModel.copyModelsFromDevice(globalMetrics.modelPerformanceMetrics);
        globalMetrics.modelPerformanceMetrics.updateBestModelCost();
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

    int popSize() {
        return optimizerContext.getPopulationSize();
    }

    int xdim() const{
        return optimizerContext.model->modelSize;
    }
};
#endif //PARALLELLBFGS_BASELEVEL_CUH
