//
// Created by spaceman on 2022. 10. 25..
//

#ifndef PARALLELLBFGS_CUDAMEMORYMODEL_CUH
#define PARALLELLBFGS_CUDAMEMORYMODEL_CUH

#include "Model.cuh"

class CUDAMemoryModel{
public:
    Random cudaRandom = Random();
    CUDAConfig cudaConfig;
    double *dev_x;
    double *dev_xDE;
    double *dev_x1;
    double *dev_x2;
    double *dev_data;
    double *dev_F;
    double *dev_FDE;
    double *dev_F1;
    double *dev_F2;
    Model* dev_Model;
    Model* model;

    void allocateFor(Model &model) {
        freeIfPresent();
        cudaMalloc((void **) &dev_x, model.modelPopulationSize * sizeof(double));
        cudaMalloc((void **) &dev_xDE, model.modelPopulationSize * sizeof(double));
        cudaMalloc((void **) &dev_data, model.residuals.residualDataSize() * sizeof(double));
        cudaMalloc((void **) &dev_F, model.populationSize * sizeof(double));
        cudaMalloc((void **) &dev_FDE, model.populationSize * sizeof(double));
        cudaMalloc((void **) &dev_Model, sizeof(Model));
    }

    void copyModelToDevice(Model &model) {
        Residual* modelResiduals=model.residuals.residual;
        cudaMalloc((void **) &model.residuals.residual, sizeof(Residual) * model.residuals.residualCount);
        cudaMemcpy(model.residuals.residual, modelResiduals, sizeof(Residual) * model.residuals.residualCount, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Model, &model, sizeof(Model), cudaMemcpyHostToDevice);
        model.residuals.residual=modelResiduals;
        this->model=&model;
    }

    void copyModelsFromDevice(ModelMetrics& modelMetrics) {
        cudaMemcpy(modelMetrics.finalFs, dev_F1, model->populationSize * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(modelMetrics.solution, dev_x1,
                   model->modelPopulationSize * sizeof(double), cudaMemcpyDeviceToHost);
    }

    void initLoopPointers() {
        dev_x1 = dev_x;
        dev_x2 = dev_xDE;
        dev_F1 = dev_F;
        dev_F2 = dev_FDE;
    }

    void swapModels() {
        std::swap(dev_x1, dev_x2);
        std::swap(dev_F1, dev_F2);
    }


    void freeModelIfPresent() {
        if(dev_Model!= nullptr) {
            Model* host_dev_Model=(Model*)malloc(sizeof(Model));
            cudaMemcpy(host_dev_Model, dev_Model, sizeof(Model),
                       cudaMemcpyDeviceToHost);
            cudaFree(host_dev_Model->residuals.residual);
            cudaFree(dev_Model);
            free(host_dev_Model);
        }
    }
    void freeIfPresent() {

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
        if(dev_F1!=nullptr) {
            cudaFree(dev_F1);
        }
        if(dev_F2!=nullptr) {
            cudaFree(dev_F2);
        }
        freeModelIfPresent();
    }
    ~CUDAMemoryModel(){
        freeIfPresent();
//        if(dev_Model!= nullptr) {
//
//        }
    }
};

#endif //PARALLELLBFGS_CUDAMEMORYMODEL_CUH
