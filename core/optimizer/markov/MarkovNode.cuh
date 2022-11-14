//
// Created by spaceman on 2022. 11. 09..
//

#ifndef PARALLELLBFGS_MARKOVNODE_CUH
#define PARALLELLBFGS_MARKOVNODE_CUH

#include "../Operator.h"
#include <vector>
#include <random>

class MarkovNode {
    std::vector<MarkovNode*> nextNodes;
    std::vector<double> nextProbabilities;

public:
    std::string name;
    explicit MarkovNode(const char* name):name(name) {
        nextNodes=std::vector<MarkovNode*>();
        nextProbabilities=std::vector<double>();
    }

    void addNext(MarkovNode* next, double nextProb) {
        // TODO check probability sums
        nextNodes.push_back(next);
        nextProbabilities.push_back(nextProb);
    }

    MarkovNode* getNext( std::mt19937 &generator) {
      double u=std::uniform_real_distribution<double>(0.0, 1.0)(generator);
      double cSum=0;
      for(int i=0;i<nextProbabilities.size();i++){
          cSum+=nextProbabilities[i];
        if(u<cSum) {
            return nextNodes[i];
        }
      }
      printf("cannot get next \n");
      return this;
    }

};
#endif //PARALLELLBFGS_MARKOVNODE_CUH
