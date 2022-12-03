//
// Created by spaceman on 2022. 10. 31..
//

#ifndef PARALLELLBFGS_BOUNDEDPARAMETER_CUH
#define PARALLELLBFGS_BOUNDEDPARAMETER_CUH
#include <random>
#include <unordered_map>
#include <utility>
#include <algorithm>

class BoundedParameter {
public:
    BoundedParameter& operator=(BoundedParameter const &  other){
        value=other.value;
        upperBound=other.upperBound;
        lowerBound=other.lowerBound;
        return *this;
    }
    double value;
    double lowerBound;
    double upperBound;
    BoundedParameter(){
    }

    BoundedParameter(double defaultValue,double lowerBound,double upperBound):value(defaultValue),lowerBound(lowerBound),upperBound(upperBound) {
    }

    void mutateByEpsilon(std::mt19937 &generator) {
        value += std::normal_distribution<double>(0, (upperBound-lowerBound)/3)(generator);
        if(value<lowerBound){
            value=lowerBound;
        }else{
            if(value>upperBound){
                value=upperBound;
            }
        }
    }

    void setRandomUniform(std::mt19937 &generator) {
        value = std::uniform_real_distribution<double>(lowerBound, upperBound)(generator);
    }

    bool operator() (const BoundedParameter& lhs, const BoundedParameter& rhs) const
    {
        return lhs.value < rhs.value;
    }
};

class OperatorParameters {
public:
    std::mt19937 generator=std::mt19937(std::random_device()());
    std::unordered_map<std::string, BoundedParameter> values;
    OperatorParameters() {
    }

    OperatorParameters(std::unordered_map<std::string, BoundedParameter> parameters):values(std::move(parameters)){
    }

    virtual void setRandomUniform() {
        std::for_each(values.begin(),values.end(),[this](auto& parameter){
            std::get<1>(parameter).setRandomUniform(this->generator);
        });
    }

    virtual void printParameters() {
        std::cout<<std::endl;
        std::for_each(values.begin(),values.end(),[this](auto& parameter){
            std::cout<<std::get<0>(parameter)<<"="<<std::get<1>(parameter).value<<" ";
        });
        std::cout<<std::endl;
    }

    virtual OperatorParameters* clone() {
        OperatorParameters* clonedParameters=new OperatorParameters(values);
        return clonedParameters;
    };

    virtual void mutateByEpsilon() {
        std::for_each(values.begin(),values.end(),[this](auto& parameter){
            std::get<1>(parameter).mutateByEpsilon(this->generator);
        });
    }

};



#endif //PARALLELLBFGS_BOUNDEDPARAMETER_CUH