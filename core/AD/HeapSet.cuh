//
// Created by spaceman on 2022. 01. 16..
//

#ifndef PARALLELLBFGS_SET_CUH
#define PARALLELLBFGS_SET_CUH

#include <assert.h>

#endif //PARALLELLBFGS_SET_CUH

struct HeapSet {
public:
    unsigned *elements;
    bool *used;
    unsigned n; //current number of elements
    unsigned size; // maximum number of elements


    __host__ __device__ explicit HeapSet(unsigned size) : size(size), n(0) {
        elements = (unsigned *) malloc(sizeof(unsigned) * size);
        used = (bool *) malloc(sizeof(bool) * size);
        for (int i = 0; i < size; i++) { // TODO try with memset
            used[i] = false;
        }
    }

    __host__ __device__ void insert(unsigned element) {
        printf("inserting %d \n", element);
        assert(n < size);
        assert(element < size);
        if (used[element]) {
            return;
        }
        elements[n] = elements[0];
        elements[0] = element;
        used[element] = true;
        n++;
        heapify(0);
    }

    __host__ __device__ bool empty() {
//        printf("empty %d \n", n == 0);
        return n == 0;
    }

    __host__ __device__ unsigned peekNext() {
//        printf("peekNext %d \n", elements[0]);
        assert(n >= 1);
        return elements[0];
    }

    __host__ __device__ void removeNext() {
        printf("removing %d \n", elements[0]);
        assert(n >= 1);
        used[elements[0]] = false;
        elements[0] = elements[n - 1];
        n--;
        heapify(0);
    }


    __host__ __device__ void heapify(unsigned i) {
        unsigned left = 2 * i;
        unsigned right = 2 * i + 1;
        unsigned max = i;
        if (left < n && elements[left] > elements[i]) {
            max = left;
        } else if (right < n && elements[right] > elements[i]) {
            max = right;
        }
        if (max != i) {
            unsigned tmp = elements[i];
            elements[i] = elements[max];
            elements[max] = tmp;
            heapify(max);
        }
    }

    __device__ __host__ ~HeapSet() {
        free(elements);
        free(used);
    }

};