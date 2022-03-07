//
// Created by spaceman on 2022. 03. 07..
//

#ifndef PARALLELLBFGS_FIFOQUEUE_CUH
#define PARALLELLBFGS_FIFOQUEUE_CUH


class FIFOQueue {
public:
    static const unsigned staticQueueSize = 5;
    const unsigned queueSize = 5;
    double *queue[staticQueueSize] = {};
    unsigned back;
    unsigned size;

    __device__ __host__
    FIFOQueue() : back(0), size(0) {
    }

    __device__ __host__
    void enqueue(double *next) {
        queue[back] = next;
        back = (back + 1) % queueSize;
        size = size + 1 < queueSize ? (size + 1) : queueSize;
#ifdef SAFE
        assert(size <= queueSize);
#endif
    }

    __device__ __host__
    unsigned getIterator() {
        return 0;
    }

    __device__ __host__
    bool hasNext(unsigned iterator) {
        return size - iterator > 0;
    }

    __device__ __host__
    double *next(unsigned &prev) {
        ++prev;
        return queue[(queueSize + back - size - 1 + prev) % queueSize];
    }

    __device__ __host__
    double *reverseNext(unsigned &prev) {
        ++prev;
        return queue[(queueSize + back - prev) % queueSize];
    }
};

#endif //PARALLELLBFGS_FIFOQUEUE_CUH
