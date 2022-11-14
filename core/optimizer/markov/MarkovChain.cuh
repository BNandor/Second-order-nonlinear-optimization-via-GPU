//
// Created by spaceman on 2022. 11. 14..
//

#ifndef PARALLELLBFGS_MARKOVCHAIN_CUH
#define PARALLELLBFGS_MARKOVCHAIN_CUH

#include "MarkovNode.cuh"
#include <unordered_map>
#include <algorithm>

class MarkovChain {
protected:
    std::mt19937 generator=std::mt19937(std::random_device()());
    std::unordered_map<std::string, MarkovNode*> nodes;
    MarkovNode* currentNode;
};
#endif //PARALLELLBFGS_MARKOVCHAIN_CUH
