//
// Created by spaceman on 2023. 06. 12..
//

#ifndef PARALLELLBFGS_SAMADSHYPERLEVEL_CUH
#define PARALLELLBFGS_SAMADSHYPERLEVEL_CUH

#include "../HyperLevel.cuh"
#include <limits>
#include <math.h>
#include <utility>
#include <string.h>
#include <vector>
#include <unordered_set>
#include "Nomad/nomad.hpp"

class SA_MADSHyperLevel: public HyperLevel {
    std::unordered_map<std::string,OperatorParameters*> parameters;
    std::unordered_set<std::string> MADS_parameters;
    std::vector<double> l;
    std::vector<double> u;
    int totalBaseLevelEvaluations;
    int currentSteps=0;

public:
    class My_Evaluator : public NOMAD::Evaluator
    {
        SA_MADSHyperLevel* hyperLevel;
    public:
        My_Evaluator(const std::shared_ptr<NOMAD::EvalParameters>& evalParams,SA_MADSHyperLevel * hyperLevel)
                : NOMAD::Evaluator(evalParams, NOMAD::EvalType::BB),hyperLevel(hyperLevel) // Evaluator for true blackbox evaluations only
        {}

        ~My_Evaluator() {}

        bool eval_x(NOMAD::EvalPoint &x, const NOMAD::Double &hMax, bool &countEval) const override
        {
            bool eval_ok = false;
            try
            {
                if(hyperLevel->currentSteps>=HH_TRIALS) {
                    return false;
                }
                std::cout<<"Running "<<hyperLevel->hyperLevelId<<" for "<<hyperLevel->totalBaseLevelEvaluations<<" evaluations, step"<<hyperLevel->currentSteps<<std::endl;
                std::vector<double> madsPoint;
                for(int i=0;i<x.size();i++){
                    double p=hyperLevel->l[i]+x[i].todouble()*(hyperLevel->u[i]-hyperLevel->l[i]);
                    std::cout<<p<<" ";
                    madsPoint.push_back(p);
                }
                hyperLevel->updateOperatorParams(hyperLevel->parameters,madsPoint.data(),hyperLevel->MADS_parameters);
                hyperLevel->baseLevel.init(hyperLevel->logJson);
                hyperLevel->baseLevel.loadInitialModel();
                double f=hyperLevel->getPerformanceSampleOfSize(hyperLevel->baseLevelSampleSize,hyperLevel->parameters,hyperLevel->totalBaseLevelEvaluations);
                hyperLevel->currentSteps++;
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

    SA_MADSHyperLevel():HyperLevel("SA-MADS") {
    }

    SA_MADSHyperLevel(std::string id):HyperLevel(std::move(id)) {
    }

    double hyperOptimize(int totalEvaluations) override {
        int totalTrials=HH_TRIALS;
        int SA_trials=totalTrials/3;
#ifdef HH_SA_HYBRID_PERCENTAGE
        SA_trials=std::truncf(((double)totalTrials)*HH_SA_HYBRID_PERCENTAGE);
#endif
        int MADS_trials=totalTrials-SA_trials;
        baseLevel.init(logJson);
        std::cout<<"SA starting"<<std::endl;
        simulatedAnnealingInitialization(totalEvaluations,SA_trials);
        cloneParameters(bestParameters,parameters);
        MADS_parameters={"OptimizerChainInitializerSimplex",
                          "OptimizerChainRefinerSimplex",
                          "OptimizerChainPerturbatorSimplex",
                          "PerturbatorDESimplex",
                          "PerturbatorGASimplex",
                          "RefinerLBFGSSimplex",
                          "RefinerGDSimplex"};
#ifdef BASE_PERTURB_EXTRA_OPERATORS
        std::string operators=BASE_PERTURB_EXTRA_OPERATORS;
        std::set<std::string> operatorSet=stringutil::splitString(operators,',');
        std::for_each(operatorSet.begin(),operatorSet.end(),[this](std::string op){
            MADS_parameters.insert("Perturbator"+op+"Simplex");
            MADS_parameters.insert("Perturbator"+op+"OperatorParams");
        });
#endif
//        parameters=createOptimizerParameters(totalEvaluations);
//        CMAES_parameters={"OptimizerChainInitializerSimplex",
//                          "OptimizerChainRefinerSimplex",
//                          "OptimizerChainPerturbatorSimplex",
//                          "PerturbatorDESimplex",
//                          "PerturbatorDEOperatorParams",
//                          "PerturbatorGASimplex",
//                          "PerturbatorGAOperatorParams",
//                          "RefinerLBFGSSimplex",
//                          "RefinerLBFGSOperatorParams",
//                          "RefinerGDSimplex",
//                          "RefinerGDOperatorParams"};
//#ifdef BASE_PERTURB_EXTRA_OPERATORS
//        std::string operators=BASE_PERTURB_EXTRA_OPERATORS;
//        std::set<std::string> operatorSet=stringutil::splitString(operators,',');
//        std::for_each(operatorSet.begin(),operatorSet.end(),[this](std::string op){
//            CMAES_parameters.insert("Perturbator"+op+"Simplex");
//            CMAES_parameters.insert("Perturbator"+op+"OperatorParams");
//        });
//#endif
        totalBaseLevelEvaluations=totalEvaluations;
        std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,std::vector<double>> serializedParameters=getParameterArrays(parameters,MADS_parameters);
        int dim = std::get<0>(serializedParameters).size();
        l=std::get<1>(serializedParameters);
        u=std::get<2>(serializedParameters);
        NOMAD::MainStep TheMainStep;

        try
        {
            auto params = std::make_shared<NOMAD::AllParameters>();
            initAllParams(params,std::get<0>(serializedParameters),l,u,std::get<3>(serializedParameters),dim,MADS_trials);
            TheMainStep.setAllParameters(params);
            auto ev = std::make_unique<My_Evaluator>(params->getEvalParams(),this);
            TheMainStep.addEvaluator(std::move(ev));
            std::cout<<"MADS starting"<<std::endl;
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

    void simulatedAnnealingInitialization(int totalEvaluations,int trials ){
        std::cout<<"Running simulated annealing with "<<totalEvaluations<<" for"<<trials<<" steps"<<std::endl;
        int totalSABaseLevelEvaluations=totalEvaluations;
        std::mt19937 generator=std::mt19937(std::random_device()());
        std::unordered_map<std::string,OperatorParameters*> defaultParameters=createOptimizerParameters(totalSABaseLevelEvaluations);
        std::unordered_map<std::string,OperatorParameters*> currentParameters=std::unordered_map<std::string,OperatorParameters*>();
        std::unordered_map<std::string,OperatorParameters*> currentMutatedByEpsilonParameters=std::unordered_map<std::string,OperatorParameters*>();
        cloneParameters(defaultParameters,currentParameters);

        baseLevel.loadInitialModel();
        double currentF=getPerformanceSampleOfSize(baseLevelSampleSize,currentParameters,totalSABaseLevelEvaluations);
        currentSteps+=1;
        double min=currentF;
        cloneParameters(currentParameters,bestParameters);

        double temp0=HH_SA_TEMP;
        double temp=temp0;
        double alpha=HH_SA_ALPHA;
        int acceptedWorse=0;
        logJson["SA-temp0"]=temp0;
        logJson["SA-alpha"]=alpha;
        logJson["SA-temps"].push_back(temp);
        for(int i=0; i < trials-1 || temp < 0; i++) {

            printf("[med+iqr]f: %f trial %u \n",currentF, i);
            cloneParameters(currentParameters,currentMutatedByEpsilonParameters);
            mutateParameters(currentMutatedByEpsilonParameters);

            double currentFPrime=getPerformanceSampleOfSize(baseLevelSampleSize,currentMutatedByEpsilonParameters,totalSABaseLevelEvaluations);
            currentSteps+=1;
            printf("[med+iqr]f': %f trial %u \n",currentFPrime, i);
            if(currentFPrime < currentF ) {
                cloneParameters(currentMutatedByEpsilonParameters,currentParameters);
                currentF=currentFPrime;
            }else {
                double r= std::uniform_real_distribution<double>(0,1)(generator);
                if(r<exp((currentF-currentFPrime)/temp)) {
                    cloneParameters(currentMutatedByEpsilonParameters,currentParameters);
                    currentF=currentFPrime;
                    acceptedWorse++;
                    std::cout<<"Accepted worse at "<<i<<"/"<<trials-1<<" at temp: "<<temp<<std::endl;
                }
            }
            temp=temp0/(1+alpha*i);
            logJson["SA-temps"].push_back(temp);
        }
        freeOperatorParamMap(defaultParameters);
        freeOperatorParamMap(currentParameters);
        freeOperatorParamMap(currentMutatedByEpsilonParameters);
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
    ~SA_MADSHyperLevel() {
        freeOperatorParamMap(parameters);
    };
};


#endif //PARALLELLBFGS_SAMADSHYPERLEVEL_CUH
