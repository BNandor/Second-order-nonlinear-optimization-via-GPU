//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_DECONTEXT_H
#define PARALLELLBFGS_DECONTEXT_H

#include "../Perturbator.h"

class DEContext : public Perturbator {
public:
    DEContext() {
        populationSize=POPULATION_SIZE;
    }
    double crossoverRate=CR;
    double force=F;
};

#endif //PARALLELLBFGS_DECONTEXT_H
