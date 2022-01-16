//
// Created by spaceman on 2021. 12. 01..
//

#ifndef PARALLELLBFGS_DFLOAT_CUH
#define PARALLELLBFGS_DFLOAT_CUH

#include <math.h>
#include <set>

enum DOperation {
    SQUARE, SQRT, PLUS, MINUS, MUL, DIV, ID, COS, SIN, INVERSE
};

struct DFloat {
public:
    float value;
    float derivative;
    unsigned index;
    unsigned *globalIndex;
    DOperation operation;
    DFloat *arguments[2];
    int argumentCount;
    int operatorTreeSize;


    __host__ __device__
    DFloat(float value, unsigned index, unsigned *globalIndex, DOperation operation = ID, int argumentCount = 0) :
            value(value),
            index(index),
            globalIndex(globalIndex),
            operation(operation),
            argumentCount(argumentCount), operatorTreeSize(1), derivative(0.0) {
    }

    __device__ DFloat operator*(DFloat &other) {
        DFloat mul = DFloat(value * other.value, atomicAdd(globalIndex, 1), globalIndex, MUL, 2);
        mul.arguments[0] = this;
        mul.arguments[1] = &other;
        mul.operatorTreeSize = this->operatorTreeSize + other.operatorTreeSize + 1;
        return mul;
    }

    __device__ DFloat operator/(DFloat &other) {
        DFloat div = DFloat(value / other.value, atomicAdd(globalIndex, 1), globalIndex, DIV, 2);
        div.arguments[0] = this;
        div.arguments[1] = &other;
        div.operatorTreeSize = this->operatorTreeSize + other.operatorTreeSize + 1;
        return div;
    }

    __device__ DFloat operator+(DFloat &other) {
        DFloat plus = DFloat(value + other.value, atomicAdd(globalIndex, 1), globalIndex, PLUS, 2);
        plus.arguments[0] = this;
        plus.arguments[1] = &other;
        plus.operatorTreeSize = this->operatorTreeSize + other.operatorTreeSize + 1;
        return plus;
    }

    __device__ DFloat operator-(DFloat &other) {
        DFloat minus = DFloat(value - other.value, atomicAdd(globalIndex, 1), globalIndex, MINUS, 2);
        minus.arguments[0] = this;
        minus.arguments[1] = &other;
        minus.operatorTreeSize = this->operatorTreeSize + other.operatorTreeSize + 1;
        return minus;
    }

    __device__ DFloat square() {
        DFloat square = DFloat(value * value, atomicAdd(globalIndex, 1), globalIndex, SQUARE, 1);
        square.arguments[0] = this;
        square.operatorTreeSize = this->operatorTreeSize + 1;
        return square;
    }

    __device__ DFloat sqrt() {
        DFloat root = DFloat(std::sqrt(value), atomicAdd(globalIndex, 1), globalIndex, SQRT, 1);
        root.arguments[0] = this;
        root.operatorTreeSize = this->operatorTreeSize + 1;
        return root;
    }

    __device__ DFloat inverse() {
        DFloat inverse = DFloat(1.0 / value, atomicAdd(globalIndex, 1), globalIndex, INVERSE, 1);
        inverse.arguments[0] = this;
        inverse.operatorTreeSize = this->operatorTreeSize + 1;
        return inverse;
    }

    __device__ DFloat sin() {
        DFloat sin = DFloat(std::sin(value), atomicAdd(globalIndex, 1), globalIndex, SIN, 1);
        sin.arguments[0] = this;
        sin.operatorTreeSize = this->operatorTreeSize + 1;
        return sin;
    }

    __device__ DFloat cos() {
        DFloat cos = DFloat(std::cos(value), atomicAdd(globalIndex, 1), globalIndex, COS, 1);
        cos.arguments[0] = this;
        cos.operatorTreeSize = this->operatorTreeSize + 1;
        return cos;
    }

    __device__ bool operator<(const DFloat &other) {
        return this->value < other.value;
    }

    __device__ bool operator>(const DFloat &other) {
        return this->value > other.value;
    }
};


#endif //PARALLELLBFGS_DFLOAT_CUH
