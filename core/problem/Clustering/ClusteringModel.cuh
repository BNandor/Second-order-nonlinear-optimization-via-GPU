//
// Created by spaceman on 2023. 08. 16..
//

#ifndef PARALLELLBFGS_CLUSTERINGMODEL_CUH
#define PARALLELLBFGS_CLUSTERINGMODEL_CUH

#include "../../optimizer/operators/perturb/Perturbator.h"
#include "../../common/Metrics.cuh"
#include "Clustering.cuh"
//#ifdef PROBLEM_CLUSTERING

class ClusteringModel: public Model {
    std::mt19937 generator=std::mt19937(std::random_device()());
    Residual ClusteringResidual[1]{};
    int samples;
    int dimension;
    int clusters;

public:

    ClusteringModel():Model(){};
    explicit ClusteringModel(Perturbator& perturbator) : Model(perturbator.populationSize,X_DIM) {
        residuals.residualCount=1;
        readDatasetSize(samples,dimension,clusters);
        ClusteringResidual[0].constantsCount = samples;
        ClusteringResidual[0].constantsDim=dimension;
        ClusteringResidual[0].parametersDim=dimension;
        residuals.residual= reinterpret_cast<Residual *>(&ClusteringResidual[0]);
    }

    void loadModel(void* dev_x, void* dev_xDE, void* dev_constantData, Metrics &metrics,CUDAMemoryModel* model ) override {
        const int constantDataSize=residuals.residualDataSize();
        double x[modelPopulationSize]={};
        double lowerbounds[modelSize]={};
        double upperbounds[modelSize]={};
        double data[constantDataSize]={};
        double points[samples*dimension]={};
        int clusterIds[samples]={};
        for(int i=0;i<modelPopulationSize;i++) {
            x[i]=std::uniform_real_distribution<double>(0,10)(generator);
        }
        readProblem(points,clusterIds,PROBLEM_PATH);
        setBounds(points,lowerbounds,upperbounds);
        int cit=0;
        for(int i=0; i < samples; i++) {
            for(int j=0; j < dimension; j++) {
                    data[i*dimension+j]=points[i*dimension +j];
            }
        }

        metrics.getCudaEventMetrics().recordStartCopy();
        model->isBounded=true;
        cudaMemcpy(model->dev_lower_bounds, &lowerbounds, modelSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(model->dev_upper_bounds, &upperbounds, modelSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_x, &x, modelPopulationSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_xDE, &x, modelPopulationSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_constantData, &data, constantDataSize * sizeof(double), cudaMemcpyHostToDevice);
        metrics.getCudaEventMetrics().recordStopCopy();
    }

    void readDatasetSize(int &samples, int &dimension,int &clusters){
        std::fstream input;
        input.open(PROBLEM_PATH);
        if (input.is_open()) {
            input >> samples >> dimension >> clusters;
            std::cout<<" clustering data samples: "<<samples<<" with dimensionality: "<<dimension<<" and number of clusters: "<<clusters<<std::endl;
            input.close();
        } else{
            std::cerr<<"Could not open"<<PROBLEM_PATH<<std::endl;
            exit(1);
        }
    }

    void setBounds(double *points, double * lower,double * upper) {
        for(int j=0; j < modelSize; j++) {
            lower[j]=std::numeric_limits<double>::max();
            upper[j]=std::numeric_limits<double>::min();
        }
        for(int i=0; i < samples; i++) {
            for(int j=0; j < dimension; j++) {
                if(points[i*dimension + j]<lower[j]){
                    lower[j]=points[i*dimension + j];
                }
                if(points[i*dimension + j]>upper[j]){
                    upper[j]=points[i*dimension + j];
                }
            }
        }
        for(int j=0; j < dimension; j++) {
            for(int k=1;k<clusters;k++){
                lower[dimension*k+j]=lower[j];
                upper[dimension*k+j]=upper[j];
            }
        }
    }

    void readProblem(double *data, int * clusterIds,std::string filename) {
        std::fstream input;
        input.open(filename.c_str());
        if (input.is_open()) {
            input >> samples >> dimension >> clusters;
            std::cout<<" clustering data samples: "<<samples<<" with dimensionality: "<<dimension<<" and number of clusters: "<<clusters<<std::endl;
            unsigned cData = 0;
            for(int i=0; i < samples; i++) {
                for(int j=0; j < dimension; j++) {
                    input >> data[cData];
                    cData++;
                }
                input >> clusterIds[i];
            }
            std::cout << "read: " << cData << " expected: " << samples*dimension
                      << std::endl;
            assert(cData == samples*dimension);
            input.close();
        } else {
            std::cerr << "err: could not open " << filename << std::endl;
            exit(1);
        }
    }
};



#define DEFINE_RESIDUAL_FUNCTIONS() \
        Clustering f1 = Clustering();

#define INJECT_RESIDUAL_FUNCTIONS() \
        ((Model*)localContext.modelP)->residuals.residual[0].residualProblem=&f1; \
        ((Model*)localContext.modelP)->residuals.residual[0].constants = globalData;


#define CAST_RESIDUAL_FUNCTIONS() \
        Clustering *f1 = ((Clustering *) model->residuals.residual[0].residualProblem);

#define COMPUTE_RESIDUALS() \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[0].constantsCount; spanningTID += blockDim.x) { \
            f1->setConstants(&(model->residuals.residual[0].constants[model->residuals.residual[0].constantsDim * spanningTID]), model->residuals.residual[0].constantsDim); \
            localContext->threadF += f1->eval(x, X_DIM)->value; \
            f1->evalJacobian(); \
            for (unsigned j = 0; j < model->residuals.residual[0].parametersDim; j++) { \
                atomicAdd(&dx[f1->ThisJacobianIndices[j]], f1->operatorTree[f1->constantSize + j].derivative); } \
        }

#define COMPUTE_LINESEARCH() \
        for (unsigned spanningTID = threadIdx.x; spanningTID < model->residuals.residual[0].constantsCount; spanningTID += blockDim.x) {  \
            f1->setConstants(&(model->residuals.residual[0].constants[model->residuals.residual[0].constantsDim * spanningTID]), model->residuals.residual[0].constantsDim); \
            fNext += f1->eval(sharedContext->xNext, X_DIM)->value; \
        }

//#endif

#endif //PARALLELLBFGS_CLUSTERINGMODEL_CUH
