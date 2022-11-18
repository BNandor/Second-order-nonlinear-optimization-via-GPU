//
// Created by spaceman on 2022. 11. 18..
//

#ifndef PARALLELLBFGS_SIMPLEXPARAMETERS_CUH
#define PARALLELLBFGS_SIMPLEXPARAMETERS_CUH
#include <utility>
#include <limits>
#include "../../../../common/model/BoundedParameter.cuh"
#include <algorithm>

class SimplexParameters: public OperatorParameters {
public:
    explicit SimplexParameters(std::unordered_map<std::string, BoundedParameter> params): OperatorParameters(std::move(params)) {
        double sum=0;
        std::for_each(values.begin(),values.end(),[&sum](auto &parameter) {
            sum+=std::get<1>(parameter).value;
        });
        if(std::abs(sum - 1.0) >std::numeric_limits<double>::epsilon()) {
            std::cerr<<"invalid simplex parameters provided, they add up to "<< sum;
            std::for_each(values.begin(),values.end(),[](auto &parameter) {
               std::cerr<<std::get<0>(parameter)<<" "<<std::get<1>(parameter).value<<" ";
            });
            std::cerr<<std::endl;
            exit(3);
        }
    }



};

#endif //PARALLELLBFGS_SIMPLEXPARAMETERS_CUH
