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

    virtual void operate(CUDAMemoryModel* cudaMemoryModel) =0;

    MarkovNode* getNext( std::mt19937 &generator) {
      double u=std::uniform_real_distribution<double>(0.0, 1.0)(generator);
      double cSum=0;
      for(int i=0;i<nextProbabilities.size();i++){
          cSum+=nextProbabilities[i];
        if(u<cSum) {
            return nextNodes[i];
        }
      }
      printf("invalid probabilities configured\n");
      exit(0);
    }

};
#endif //PARALLELLBFGS_MARKOVNODE_CUH
