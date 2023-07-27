//
// Created by spaceman on 2022. 10. 31..
//

#ifndef PARALLELLBFGS_BOUNDEDPARAMETER_CUH
#define PARALLELLBFGS_BOUNDEDPARAMETER_CUH
#include <random>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include "../../json.hpp"
using json = nlohmann::json;

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

    double clipToBound(double  v){
        if(v<lowerBound){
            return lowerBound;
        }else{
            if(v>upperBound){
                return upperBound;
            }
        }
        return v;
    }

    void mutateByEpsilon(std::mt19937 &generator) {
        value += std::normal_distribution<double>(0, (upperBound-lowerBound)/3)(generator);
        value=clipToBound(value);
    }

    void setRandomUniform(std::mt19937 &generator) {
        value = std::uniform_real_distribution<double>(lowerBound, upperBound)(generator);
    }

    bool operator() (const BoundedParameter& lhs, const BoundedParameter& rhs) const
    {
        return lhs.value < rhs.value;
    }

    json getJson(){
        json jsonValue;
        jsonValue["lowerBound"]=lowerBound;
        jsonValue["upperBound"]=upperBound;
        jsonValue["value"]=value;
        return jsonValue;
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

    virtual json getJson() {
        json parametersJson;
        std::for_each(values.begin(),values.end(),[&parametersJson](auto& parameter){
            parametersJson[std::get<0>(parameter)]=std::get<1>(parameter).getJson();
        });
        return parametersJson;
    }

    void setParameterValue(std::string& parameter, double value) {
        if(values.count(parameter)>0) {
            values[parameter].value=values[parameter].clipToBound(value);
        }else{
            std::cerr<<"cannot set value"<<value <<"for parameter"<<parameter<<std::endl;
            exit(4);
        }
    }
};



#endif //PARALLELLBFGS_BOUNDEDPARAMETER_CUH