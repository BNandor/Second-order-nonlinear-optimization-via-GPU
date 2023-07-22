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
            std::list<std::string> mutatedParameters={
                    "PerturbatorInitializerSimplex",
                    "PerturbatorDESimplex",
                    "PerturbatorGASimplex",
                    "PerturbatorDEOperatorParams",
                    "PerturbatorGAOperatorParams"};
#ifdef BASE_PERTURB_EXTRA_OPERATORS
        std::string operators=BASE_PERTURB_EXTRA_OPERATORS;
            std::set<std::string> operatorSet=stringutil::splitString(operators,',');
            std::for_each(operatorSet.begin(),operatorSet.end(),[&chainParameters,operatorSet,&mutatedParameters](std::string op) {
                mutatedParameters.push_back("Perturbator"+op+"Simplex");
                mutatedParameters.push_back("Perturbator"+op+"OperatorParams");
            });
#endif
        mutateSelectParametersByEpsilon(chainParameters,mutatedParameters);
    }
};


#endif //PARALLELLBFGS_SIMULATEDANNEALINGPERTURBHYPERLEVEL_CUH
