//
// Created by spaceman on 2023. 02. 18..
//

#ifndef PARALLELLBFGS_SIMULATEDANNEALINGPERTURBHYPERLEVEL_CUH
#define PARALLELLBFGS_SIMULATEDANNEALINGPERTURBHYPERLEVEL_CUH

#include "SimulatedAnnealingHyperLevel.cuh"
//
// Created by spaceman on 2023. 02. 18..
//
class SimulatedAnnealingPerturbHyperLevel: public SimulatedAnnealingHyperLevel {
public:
    SimulatedAnnealingPerturbHyperLevel():SimulatedAnnealingHyperLevel("SA_PERTURB") {
    }

    std::unordered_map<std::string,OperatorParameters*> createOptimizerParameters(int totalBaseLevelEvaluations) override {
        std::cout<<"initializing perturb optimizer parameters"<<std::endl;
        return createSimplePerturbOptimizerParameters(totalBaseLevelEvaluations);
    }

    void mutateParameters(std::unordered_map<std::string,OperatorParameters*> &chainParameters) override {
        std::cout<<"mutating by selection"<<std::endl;
        mutateSelectParametersByEpsilon(chainParameters,{
                                                         "PerturbatorInitializerSimplex",
                                                         "PerturbatorDESimplex",
                                                         "PerturbatorGASimplex",
                                                         "PerturbatorDEOperatorParams",
                                                         "PerturbatorGAOperatorParams"});
    }
};


#endif //PARALLELLBFGS_SIMULATEDANNEALINGPERTURBHYPERLEVEL_CUH
