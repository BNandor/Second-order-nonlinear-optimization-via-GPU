//
// Created by spaceman on 2023. 02. 18..
//

#ifndef PARALLELLBFGS_SIMULATEDANNEALINGREFINEHYPERLEVEL_CUH
#define PARALLELLBFGS_SIMULATEDANNEALINGREFINEHYPERLEVEL_CUH

#include "SimulatedAnnealingHyperLevel.cuh"

class SimulatedAnnealingRefineHyperLevel: public SimulatedAnnealingHyperLevel {
public:
    SimulatedAnnealingRefineHyperLevel():SimulatedAnnealingHyperLevel("SA_REFINE") {
    }

    std::unordered_map<std::string,OperatorParameters*> createOptimizerParameters(int totalBaseLevelEvaluations) override {
        std::cout<<"initializing refine optimizer parameters"<<std::endl;
        return createSimpleLocalSearchOptimizerParameters(totalBaseLevelEvaluations);
    }

    void mutateParameters(std::unordered_map<std::string,OperatorParameters*> &chainParameters) override {
        mutateSelectParametersByEpsilon(chainParameters,{
                "RefinerLBFGSOperatorParams",
                "RefinerGDOperatorParams",
                "RefinerInitializerSimplex",
                "RefinerGDSimplex",
                "RefinerLBFGSSimplex"});

    }
};
#endif //PARALLELLBFGS_SIMULATEDANNEALINGREFINEHYPERLEVEL_CUH
