//
// Created by spaceman on 2022. 11. 18..
//

#ifndef PARALLELLBFGS_SIMPLEXPARAMETERS_CUH
#define PARALLELLBFGS_SIMPLEXPARAMETERS_CUH
#include <utility>
#include <iostream>
#include <limits>
#include "../../../../common/model/BoundedParameter.cuh"
#include <algorithm>
#include <vector>
#include <tuple>
#include <json.hpp>
using json = nlohmann::json;

class SimplexParameters: public OperatorParameters {
public:
    explicit SimplexParameters(std::unordered_map<std::string, BoundedParameter> params): OperatorParameters(std::move(params)) {
        double sum=0;
        std::for_each(values.begin(),values.end(),[&sum]( std::pair<const std::string, BoundedParameter> &parameter) {
            sum+=parameter.second.value;
        });
        if(std::abs(sum - 1.0) >std::numeric_limits<double>::epsilon()) {
            std::cerr<<"invalid simplex chainParameters provided, they add up to "<< sum;
            std::for_each(values.begin(),values.end(),[](std::pair<const std::string, BoundedParameter>  &parameter) {
               std::cerr<<parameter.first<<" "<<parameter.second.value<<" ";
            });
            std::cerr<<std::endl;
            exit(3);
        }
    }

    void setRandomUniform() override {
        int n=values.size();
        std::vector<double> simplexSamples;
        simplexSamples.push_back(0.0);
        for(int i=0;i<n-1;i++) {
            simplexSamples.push_back(std::uniform_real_distribution<double>(0.0, 1.0)(generator));
        }
        std::sort(simplexSamples.begin(),simplexSamples.end());
        simplexSamples.push_back(1.0);
        int sIndex=0;
        std::for_each(values.begin(),values.end(),[&sIndex,&simplexSamples]( std::pair<const std::string, BoundedParameter> & parameter){
            parameter.second.value=simplexSamples[sIndex+1]-simplexSamples[sIndex];
            sIndex++;
        });
    }

    void printParameters() override {
        std::cout<<std::endl;
        std::cout<<"\tsimplex: ";
        std::for_each(values.begin(),values.end(),[](std::pair<const std::string, BoundedParameter>  &parameter) {
            std::cout<<parameter.first<<" "<<parameter.second.value<<" ";
        });
        std::cout<<std::endl;
    }

    SimplexParameters* clone() override {
        SimplexParameters* clonedParameters=new SimplexParameters(values);
        return clonedParameters;
    };

    void mutateByEpsilon() override {
        std::vector<double> simplexSamples;
        double pSum=0;
        simplexSamples.push_back(0.0);
        std::for_each(values.begin(),values.end(),[&pSum,&simplexSamples](const std::pair<std::string, BoundedParameter>  & parameter) {
            pSum+=parameter.second.value;
            simplexSamples.push_back(pSum);
        });
        for(int i=1; i < simplexSamples.size() - 1 ; i++) {
            simplexSamples[i] += std::normal_distribution<double>(0, 0.3)(generator);
            if(simplexSamples[i]<0) {
                simplexSamples[i]=0;
            } else {
                if(simplexSamples[i]>1.0) {
                    simplexSamples[i]=1.0;
                }
            }
        }
        std::sort(simplexSamples.begin(),simplexSamples.end());
        int sIndex=0;
        std::for_each(values.begin(),values.end(),[&sIndex,&simplexSamples,this]( std::pair<const std::string, BoundedParameter> & parameter){
            parameter.second.value=simplexSamples[sIndex+1]-simplexSamples[sIndex];
            sIndex++;
        });
    }
};

#endif //PARALLELLBFGS_SIMPLEXPARAMETERS_CUH
