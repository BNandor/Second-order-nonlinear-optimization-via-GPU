//
// Created by spaceman on 2022. 10. 25..
//

#ifndef PARALLELLBFGS_CUDAMEMORYMODEL_CUH
#define PARALLELLBFGS_CUDAMEMORYMODEL_CUH

#include "Model.cuh"
#include <stdio.h>
#include <stdlib.h>
#ifndef gpuErrchk
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif
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
    double *dev_lower_bounds;
    double *dev_upper_bounds;
    bool isBounded=false;
    Model* dev_Model;
    Model* model;

    void allocateFor(Model &theModel) {
        freeIfPresent();
        gpuErrchk(cudaMalloc((void **) &dev_lower_bounds, theModel.modelSize * sizeof(double)));
        gpuErrchk(cudaMalloc((void **) &dev_upper_bounds, theModel.modelSize * sizeof(double)));
        gpuErrchk(cudaMalloc((void **) &dev_x, theModel.modelPopulationSize * sizeof(double)));
        gpuErrchk(cudaMalloc((void **) &dev_xDE, theModel.modelPopulationSize * sizeof(double)));
        gpuErrchk(cudaMalloc((void **) &dev_data, theModel.residuals.residualDataSize() * sizeof(double)));
        gpuErrchk(cudaMalloc((void **) &dev_F, theModel.populationSize * sizeof(double)));
        gpuErrchk(cudaMalloc((void **) &dev_FDE, theModel.populationSize * sizeof(double)));
        gpuErrchk(cudaMalloc((void **) &dev_Model, sizeof(Model)));
    }

    void copyModelToDevice(Model &model) {
        Residual* modelResiduals=model.residuals.residual;
        gpuErrchk(cudaMalloc((void **) &model.residuals.residual, sizeof(Residual) * model.residuals.residualCount));
        gpuErrchk(cudaMemcpy(model.residuals.residual, modelResiduals, sizeof(Residual) * model.residuals.residualCount, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_Model, &model, sizeof(Model), cudaMemcpyHostToDevice));
        model.residuals.residual=modelResiduals;
        this->model=&model;
    }

    void copyModelsFromDevice(ModelMetrics& modelMetrics) {
        gpuErrchk(cudaMemcpy(modelMetrics.finalFs, dev_F1, model->populationSize * sizeof(double),
                   cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(modelMetrics.solution, dev_x1,
                   model->modelPopulationSize * sizeof(double), cudaMemcpyDeviceToHost));
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
            gpuErrchk(cudaMemcpy(host_dev_Model, dev_Model, sizeof(Model),
                       cudaMemcpyDeviceToHost));
            gpuErrchk(cudaFree(host_dev_Model->residuals.residual));
            gpuErrchk(cudaFree(dev_Model));
            dev_Model=0;
            free(host_dev_Model);
        }
    }
    void freeIfPresent() {
        if(dev_upper_bounds != nullptr){
            gpuErrchk(cudaFree(dev_upper_bounds));
            dev_upper_bounds=0;
        }
        if(dev_lower_bounds != nullptr){
            gpuErrchk(cudaFree(dev_lower_bounds));
            dev_lower_bounds=0;
        }
        if(dev_x!=nullptr) {
            gpuErrchk(cudaFree(dev_x));
            dev_x=0;
        }
        if(dev_xDE!=nullptr){
            gpuErrchk(cudaFree(dev_xDE));
            dev_xDE=0;
        }
//        if(dev_x1!=nullptr){
//            gpuErrchk(cudaFree(dev_x1));
//        }
//        if(dev_x2!=nullptr){
//            gpuErrchk(cudaFree(dev_x2));
//        }
        if(dev_data!=nullptr){
            gpuErrchk(cudaFree(dev_data));
            dev_data=0;
        }
        if(dev_F!=nullptr){
            gpuErrchk(cudaFree(dev_F));
            dev_F=0;
        }
        if(dev_FDE!=nullptr){
            gpuErrchk(cudaFree(dev_FDE));
            dev_FDE=0;
        }
//        if(dev_F1!=nullptr) {
//            gpuErrchk(cudaFree(dev_F1));
//        }
//        if(dev_F2!=nullptr) {
//            gpuErrchk(cudaFree(dev_F2));
//        }
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
