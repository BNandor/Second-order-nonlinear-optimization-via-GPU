//
// Created by spaceman on 2022. 10. 25..
//

#ifndef PARALLELLBFGS_MODELMETRICS_CUH
#define PARALLELLBFGS_MODELMETRICS_CUH
#include "model/Model.cuh"
#include <limits>
class ModelMetrics {
public:
    double *solution;
    double *finalFs;
    double minF=std::numeric_limits<double>::max();
    int populationSize;
    int markovIterations;
    int fEvaluations;
    ModelMetrics(){}
    ~ModelMetrics(){
        delete solution;
        delete finalFs;
    }

    explicit ModelMetrics(int solutionSize, int populationSize){
        solution = new double[solutionSize];
        finalFs = new double [populationSize];
        this->populationSize=populationSize;
    }

    int bestModelIndex(){
        int min = 0;
//        std::cout<<"model values:"<<finalFs[0]<<" ";
        for (int ff = 1; ff < populationSize; ff++) {
//            std::cout<<finalFs[ff]<<" ";
            if (finalFs[min] > finalFs[ff]) {
                min = ff;
            }
        }
//        std::cout<<"\nminIndex="<<min<<std::endl;
        return min;
    }

    double updateBestModelCost() {
        minF=finalFs[bestModelIndex()];
        return minF;
    }

    void printBestModel(Model*model) {
        printf(" \ncalculating best model so far\n");
        int min =bestModelIndex();
        printf("\nsolf: %f and solution: ", finalFs[min]);
        for (int ff = model->modelSize * min; ff < model->modelSize * (min + 1) - 1; ff++) {
            printf("%f,", solution[ff]);
        }
        printf("%f\n", solution[model->modelSize * (min + 1) - 1]);
        printf("\nfinal f: %.10f", finalFs[min]);
        printf("\nf: %.10f\n", finalFs[min]);
    }

    void persistBestModelTo(Model*model, std::string filename) {
        printf(" persisting best model to %s\n",filename.c_str());
        int min =bestModelIndex();
        std::ofstream output;
        output.open(filename.c_str());
        if (output.is_open()) {
            for (int i=0;i<model->modelSize;i++){
                output<<std::setprecision(17)<<solution[model->modelSize * min+i]<<std::endl;
            }
            output.close();
        } else {
            std::cout << "err: could not open " << filename << std::endl;
            exit(1);
        }
    }
};


#endif //PARALLELLBFGS_MODELMETRICS_CUH
