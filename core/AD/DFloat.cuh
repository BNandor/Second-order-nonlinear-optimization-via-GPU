//
// Created by spaceman on 2021. 12. 01..
//

#ifndef PARALLELLBFGS_DFLOAT_CUH
#define PARALLELLBFGS_DFLOAT_CUH

#include <math.h>
#include <stdlib.h>

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
        ++(*globalIndex);
    }

    __host__ __device__
    DFloat() : value(0),
               index(0),
               globalIndex(nullptr),
               operation(ID),
               argumentCount(0), operatorTreeSize(1), derivative(0.0) {
    }

    __host__ __device__ DFloat operator*(DFloat &other) {
        DFloat mul = DFloat(value * other.value, *globalIndex, globalIndex, MUL, 2);
        mul.arguments[0] = this;
        mul.arguments[1] = &other;
        mul.operatorTreeSize = this->operatorTreeSize + other.operatorTreeSize + 1;
        return mul;
    }

    __host__ __device__ DFloat operator/(DFloat &other) {
        DFloat div = DFloat(value / other.value, *globalIndex, globalIndex, DIV, 2);
        div.arguments[0] = this;
        div.arguments[1] = &other;
        div.operatorTreeSize = this->operatorTreeSize + other.operatorTreeSize + 1;
        return div;
    }

    __host__ __device__ DFloat operator+(DFloat &other) {
        DFloat plus = DFloat(value + other.value, *globalIndex, globalIndex, PLUS, 2);
        plus.arguments[0] = this;
        plus.arguments[1] = &other;
        plus.operatorTreeSize = this->operatorTreeSize + other.operatorTreeSize + 1;
        return plus;
    }

    __host__ __device__ DFloat operator-(DFloat &other) {
        DFloat minus = DFloat(value - other.value, *globalIndex, globalIndex, MINUS, 2);
        minus.arguments[0] = this;
        minus.arguments[1] = &other;
        minus.operatorTreeSize = this->operatorTreeSize + other.operatorTreeSize + 1;
        return minus;
    }

    __host__ __device__ DFloat square() {
        DFloat square = DFloat(value * value, *globalIndex, globalIndex, SQUARE, 1);

        square.arguments[0] = this;
        square.operatorTreeSize = this->operatorTreeSize + 1;
        return square;
    }

    __host__ __device__ DFloat sqrt() {
        DFloat root = DFloat(std::sqrt(value), *globalIndex, globalIndex, SQRT, 1);
        root.arguments[0] = this;
        root.operatorTreeSize = this->operatorTreeSize + 1;
        return root;
    }

    __host__ __device__ DFloat inverse() {
        DFloat inverse = DFloat(1.0 / value, *globalIndex, globalIndex, INVERSE, 1);
        inverse.arguments[0] = this;
        inverse.operatorTreeSize = this->operatorTreeSize + 1;
        return inverse;
    }

    __host__ __device__ DFloat sin() {
        DFloat sin = DFloat(std::sin(value), *globalIndex, globalIndex, SIN, 1);
        sin.arguments[0] = this;
        sin.operatorTreeSize = this->operatorTreeSize + 1;
        return sin;
    }

    __host__ __device__ DFloat cos() {
        DFloat cos = DFloat(std::cos(value), *globalIndex, globalIndex, COS, 1);
        cos.arguments[0] = this;
        cos.operatorTreeSize = this->operatorTreeSize + 1;
        return cos;
    }

    __host__ __device__ bool operator<(const DFloat &other) {
        return this->value < other.value;
    }

    __host__ __device__ bool operator>(const DFloat &other) {
        return this->value > other.value;
    }

    __host__ __device__  void
    setPartialDerivatives(DFloat *parameterList) {
        derivative = 1.0;
        for (int i = index; i >= 0; i--) {
            propagateDerivative(parameterList[i], parameterList);
        }
    }

    __host__ __device__ void propagateDerivative(DFloat &node, DFloat *parameterList) { // TODO optimize dereferencing
        switch (node.operation) {
            case PLUS: {
                parameterList[node.arguments[0]->index].derivative += node.derivative;
                parameterList[node.arguments[1]->index].derivative += node.derivative;
            }
                break;
            case MINUS: {
                parameterList[node.arguments[0]->index].derivative += node.derivative;
                parameterList[node.arguments[1]->index].derivative -= node.derivative;
            }
                break;
            case SQUARE: {
                parameterList[node.arguments[0]->index].derivative +=
                        2.0 * node.derivative * parameterList[node.arguments[0]->index].value;
            }
                break;
            case SQRT: {
                parameterList[node.arguments[0]->index].derivative +=
                        node.derivative * 0.5 / std::sqrt(parameterList[node.arguments[0]->index].value);
            }
                break;
            case MUL: {
                parameterList[node.arguments[0]->index].derivative +=
                        node.derivative * parameterList[node.arguments[1]->index].value;
                parameterList[node.arguments[1]->index].derivative +=
                        node.derivative * parameterList[node.arguments[0]->index].value;
            }
                break;
            case DIV: {
                parameterList[node.arguments[0]->index].derivative +=
                        node.derivative * (1.0 / parameterList[node.arguments[1]->index].value);
                parameterList[node.arguments[1]->index].derivative +=
                        node.derivative * parameterList[node.arguments[0]->index].value *
                        (-1 / std::pow(parameterList[node.arguments[1]->index].value, 2));
            }
                break;
            case COS: {
                parameterList[node.arguments[0]->index].derivative -=
                        node.derivative * std::sin(parameterList[node.arguments[0]->index].value);
            }
                break;
            case SIN: {
                parameterList[node.arguments[0]->index].derivative +=
                        node.derivative * std::cos(parameterList[node.arguments[0]->index].value);
            }
                break;
            case INVERSE: {
                parameterList[node.arguments[0]->index].derivative -=
                        node.derivative / std::pow(parameterList[node.arguments[0]->index].value, 2);
            }
                break;
        }
    }
};


#endif //PARALLELLBFGS_DFLOAT_CUH
