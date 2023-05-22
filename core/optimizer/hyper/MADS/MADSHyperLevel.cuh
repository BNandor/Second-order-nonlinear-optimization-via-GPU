//
// Created by spaceman on 2023. 04. 27..
//

#ifndef PARALLELLBFGS_MADSHYPERLEVEL_CUH
#define PARALLELLBFGS_MADSHYPERLEVEL_CUH

#include "../HyperLevel.cuh"
#include <limits>
#include <math.h>
#include <utility>
#include <string.h>
#include <vector>
#include <unordered_set>
#include "Nomad/nomad.hpp"

class MADSHyperLevel: public HyperLevel {
    std::unordered_map<std::string,OperatorParameters*> parameters;
    std::unordered_set<std::string> CMAES_parameters;
    std::vector<double> l;
    std::vector<double> u;
    int totalBaseLevelEvaluations;

public:
    class My_Evaluator : public NOMAD::Evaluator
    {
        MADSHyperLevel* hyperLevel;
    public:
        My_Evaluator(const std::shared_ptr<NOMAD::EvalParameters>& evalParams,MADSHyperLevel * hyperLevel)
                : NOMAD::Evaluator(evalParams, NOMAD::EvalType::BB),hyperLevel(hyperLevel) // Evaluator for true blackbox evaluations only
        {}

        ~My_Evaluator() {}

        bool eval_x(NOMAD::EvalPoint &x, const NOMAD::Double &hMax, bool &countEval) const override
        {
            bool eval_ok = false;
            try
            {
                std::cout<<"Running "<<hyperLevel->hyperLevelId<<" for "<<hyperLevel->totalBaseLevelEvaluations<<" evaluations"<<std::endl;
                std::vector<double> madsPoint;
                for(int i=0;i<x.size();i++){
                    double p=hyperLevel->l[i]+x[i].todouble()*(hyperLevel->u[i]-hyperLevel->l[i]);
                    std::cout<<p<<" ";
                    madsPoint.push_back(p);
                }
                hyperLevel->updateOperatorParams(hyperLevel->parameters,madsPoint.data(),hyperLevel->CMAES_parameters);
                hyperLevel->baseLevel.init(hyperLevel->logJson);
                hyperLevel->baseLevel.loadInitialModel();
                double f=hyperLevel->getPerformanceSampleOfSize(hyperLevel->baseLevelSampleSize,hyperLevel->parameters,hyperLevel->totalBaseLevelEvaluations);
                x.setBBO(to_string(f));
                eval_ok = true;
            }
            catch (std::exception &e)
            {
                std::string err("Exception: ");
                err += e.what();
                throw std::logic_error(err);
            }

            countEval = true;
            return eval_ok;
        }
    };

    void initAllParams(std::shared_ptr<NOMAD::AllParameters> allParams, std::vector<double> initialX,std::vector<double> l,std::vector<double> u,std::vector<double> sigma,int dim,int steps)
    {
        // Parameters creation
        // Number of variables

        allParams->setAttributeValue( "DIMENSION", dim);
        // The algorithm terminates after
        // this number of black-box evaluations
        allParams->setAttributeValue( "MAX_BB_EVAL", steps);
        // Starting point
        allParams->setAttributeValue( "X0", NOMAD::Point(initialX));

        allParams->getPbParams()->setAttributeValue("GRANULARITY", NOMAD::ArrayOfDouble(dim,0.0));

        // Constraints and objective
        NOMAD::BBOutputTypeList bbOutputTypes;
        bbOutputTypes.push_back(NOMAD::BBOutputType::OBJ);    // f
        allParams->setAttributeValue("BB_OUTPUT_TYPE", bbOutputTypes );
        allParams->setAttributeValue("DIRECTION_TYPE", NOMAD::DirectionType::ORTHO_NP1_NEG);
        allParams->setAttributeValue("DISPLAY_DEGREE", 2);
        allParams->setAttributeValue("DISPLAY_UNSUCCESSFUL", false);
        allParams->setAttributeValue("LOWER_BOUND", NOMAD::ArrayOfDouble(dim,0.0));
        allParams->setAttributeValue("UPPER_BOUND", NOMAD::ArrayOfDouble(dim,1.0));

        // Parameters validation
        allParams->checkAndComply();
    }

    MADSHyperLevel():HyperLevel("MADS") {
    }

    MADSHyperLevel(std::string id):HyperLevel(std::move(id)) {
    }

    double hyperOptimize(int totalEvaluations) override {
        parameters=createOptimizerParameters(totalEvaluations);
        CMAES_parameters={"OptimizerChainInitializerSimplex",
                          "OptimizerChainRefinerSimplex",
                          "OptimizerChainPerturbatorSimplex",
                          "PerturbatorDESimplex",
                          "PerturbatorGASimplex",
                          "RefinerLBFGSSimplex",
                          "RefinerGDSimplex"};
        totalBaseLevelEvaluations=totalEvaluations;
        std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,std::vector<double>> serializedParameters=getParameterArrays(parameters,CMAES_parameters);
        int dim = std::get<0>(serializedParameters).size();
        l=std::get<1>(serializedParameters);
        u=std::get<2>(serializedParameters);
        NOMAD::MainStep TheMainStep;

        try
        {
            auto params = std::make_shared<NOMAD::AllParameters>();
            initAllParams(params,std::get<0>(serializedParameters),l,u,std::get<3>(serializedParameters),dim,totalBaseLevelEvaluations);
            TheMainStep.setAllParameters(params);
            auto ev = std::make_unique<My_Evaluator>(params->getEvalParams(),this);
            TheMainStep.addEvaluator(std::move(ev));
            TheMainStep.start();
            TheMainStep.run();
            TheMainStep.end();
        }

        catch(std::exception &e)
        {
            std::cerr << "\nNOMAD has been interrupted (" << e.what() << ")\n\n";
        }
//        printParameters(bestParameters);
//        baseLevel.printCurrentBestGlobalModel();
        return 0;
    };

    virtual std::unordered_map<std::string,OperatorParameters*>  createOptimizerParameters(int totalBaseLevelEvaluations) {
        std::unordered_map<std::string,OperatorParameters*> param=createDefaultOptimizerParameters(totalBaseLevelEvaluations);
//        setRandomUniform(param);
        return param;
    }

    virtual void mutateParameters(std::unordered_map<std::string,OperatorParameters*> &chainParameters) {
        mutateAllParametersByEpsilon(chainParameters);
    }

    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,std::vector<double>> getParameterArrays(
            const std::unordered_map<std::string, OperatorParameters*>& operatorParamsMap,
            const std::unordered_set<std::string>& paramsToInclude
    ) {
        // Initialize the arrays to be returned
        std::vector<double> x, l, u, sigma;

        // Iterate over the operator parameters and add their values, lower bounds, and upper bounds to the arrays
        for (const auto& paramsTuple : operatorParamsMap) {
//            if (paramsToInclude.find(paramsTuple.first) == paramsToInclude.end()) {
//                continue;
//            }
            OperatorParameters* operatorParams = paramsTuple.second;
            for (auto& paramPair : operatorParams->values) {
                BoundedParameter& param = paramPair.second;
                x.push_back((param.value-param.lowerBound)/(param.upperBound-param.lowerBound));
                l.push_back(param.lowerBound);
                u.push_back(param.upperBound);
                sigma.push_back((param.upperBound-param.lowerBound)/5);
            }
        }
        return std::make_tuple(x, l, u,sigma);
    }

    void updateOperatorParams(std::unordered_map<std::string, OperatorParameters*>& operatorParamsMap, const double* x, const std::unordered_set<std::string>& paramsToInclude) {
        size_t index = 0;
        // Iterate over the operator parameters and update the values of the parameters in the set
        for (const auto& paramsTuple : operatorParamsMap) {
//            if (paramsToInclude.find(paramsTuple.first) == paramsToInclude.end()) {
//                continue;
//            }
            OperatorParameters* operatorParams = paramsTuple.second;
            for (auto& paramPair : operatorParams->values) {
                BoundedParameter& param = paramPair.second;
                param.value=x[index];
                ++index;
            }
        }
    }


public:
    ~MADSHyperLevel() {
        freeOperatorParamMap(parameters);
    };
};

#endif //PARALLELLBFGS_MADSHYPERLEVEL_CUH
