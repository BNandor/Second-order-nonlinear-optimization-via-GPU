//
// Created by spaceman on 2022. 01. 16..
//

#ifndef PARALLELLBFGS_DFUNCTION_CUHH
#define PARALLELLBFGS_DFUNCTION_CUHH

#include "DFloat.cuh"
#include "HeapSet.cuh"
#include <assert.h>

struct DFunction {
public:
    unsigned *BFSOrder;
    unsigned BFSSize;
    bool BFSPerformed;
    DFloat **parameterList;
    unsigned parameterCount;

    __device__ __host__ DFunction() : BFSPerformed(false) {
        BFSOrder = nullptr;
    }

    __device__ __host__  void BFS(DFloat &value) {
        assert(BFSOrder == nullptr);
        BFSOrder = (unsigned *) malloc(sizeof(unsigned) * value.operatorTreeSize);
        parameterList = (DFloat **) malloc(sizeof(DFloat *) * (value.index + 1));
        parameterCount = value.index + 1;

        HeapSet orderHeap(value.index + 1);
        parameterList[value.index] = &value;

        orderHeap.insert(value.index);
        int nextIndex = 0;
        while (!orderHeap.empty()) {
            unsigned next = orderHeap.peekNext();
            orderHeap.removeNext();
            BFSOrder[nextIndex] = next;
            assert(nextIndex < parameterCount);
            for (int i = 0; i < parameterList[next]->argumentCount; i++) {
                orderHeap.insert(parameterList[next]->arguments[i]->index);
                parameterList[parameterList[next]->arguments[i]->index] = parameterList[next]->arguments[i];
            }
            nextIndex++;
        }
        BFSSize = nextIndex;
        BFSPerformed = true;
    }

    __device__ __host__ void setPartialDerivatives(DFloat &value) { // BFSSize needs to be set
        if (!BFSPerformed) {
            BFS(value);
        }
        value.derivative = 1.0;
        for (int i = 0; i < BFSSize; i++) {
            propagateDerivative(parameterList[BFSOrder[i]]);
        }
    }

    __device__ __host__ void propagateDerivative(DFloat *node) {
        switch (node->operation) {
            case PLUS: {
                parameterList[node->arguments[0]->index]->derivative += node->derivative;
                parameterList[node->arguments[1]->index]->derivative += node->derivative;
            }
                break;
            case MINUS: {
                parameterList[node->arguments[0]->index]->derivative += node->derivative;
                parameterList[node->arguments[1]->index]->derivative -= node->derivative;
            }
                break;
            case SQUARE: {
                parameterList[node->arguments[0]->index]->derivative +=
                        2.0 * node->derivative * parameterList[node->arguments[0]->index]->value;
            }
                break;
            case SQRT: {
                parameterList[node->arguments[0]->index]->derivative +=
                        node->derivative * 0.5 / std::sqrt(parameterList[node->arguments[0]->index]->value);
            }
                break;
            case MUL: {
                parameterList[node->arguments[0]->index]->derivative +=
                        node->derivative * parameterList[node->arguments[1]->index]->value;
                parameterList[node->arguments[1]->index]->derivative +=
                        node->derivative * parameterList[node->arguments[0]->index]->value;
            }
                break;
            case DIV: {
                parameterList[node->arguments[0]->index]->derivative +=
                        node->derivative * (1.0 / parameterList[node->arguments[1]->index]->value);
                parameterList[node->arguments[1]->index]->derivative +=
                        node->derivative * parameterList[node->arguments[0]->index]->value *
                        (-1 / std::pow(parameterList[node->arguments[1]->index]->value, 2));
            }
                break;
            case COS: {
                parameterList[node->arguments[0]->index]->derivative -=
                        node->derivative * std::sin(parameterList[node->arguments[0]->index]->value);
            }
                break;
            case SIN: {
                parameterList[node->arguments[0]->index]->derivative +=
                        node->derivative * std::cos(parameterList[node->arguments[0]->index]->value);
            }
                break;
            case INVERSE: {
                parameterList[node->arguments[0]->index]->derivative -=
                        node->derivative / std::pow(parameterList[node->arguments[0]->index]->value, 2);
            }
                break;
        }
    }

    __device__ __host__  ~DFunction() {
        if (BFSOrder != nullptr) {
            free(BFSOrder);
        }
        if (parameterList != nullptr) {
            free(parameterList);
        }
    }
};


#endif //PARALLELLBFGS_DFUNCTION_CUHH
