//
// Created by spaceman on 2022. 10. 31..
//

#ifndef PARALLELLBFGS_BOUNDEDPARAMETER_CUH
#define PARALLELLBFGS_BOUNDEDPARAMETER_CUH
#include <random>
#include <unordered_map>
#include <utility>

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
};



#endif //PARALLELLBFGS_BOUNDEDPARAMETER_CUH