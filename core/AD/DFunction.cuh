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

    __device__ __host__  void BFS(DFloat &value) { // parameterlist needs to be initialized
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
