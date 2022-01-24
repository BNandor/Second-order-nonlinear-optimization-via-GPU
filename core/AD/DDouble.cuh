//
// Created by spaceman on 2021. 12. 01..
//

#ifndef PARALLELLBFGS_DDOUBLE_CUH
#define PARALLELLBFGS_DDOUBLE_CUH

#include <math.h>
#include <stdlib.h>

enum DOperation {
    SQUARE, SQRT, PLUS, MINUS, MUL, DIV, ID, COS, SIN, INVERSE, CONST
};

struct DDouble {
public:
    double value;
    double derivative;
    unsigned index;
    unsigned *globalIndex;
    DOperation operation;
    DDouble *arguments[2];
    int argumentCount;
    int operatorTreeSize;
    bool constant;

    __host__ __device__
    DDouble(double value, unsigned index, unsigned *globalIndex, DOperation operation = ID,
            int argumentCount = 0) :
            value(value),
            index(index),
            globalIndex(globalIndex),
            operation(operation),
            argumentCount(argumentCount),
            operatorTreeSize(1),
            derivative(0.0) {
        ++(*globalIndex);
    }

    __host__ __device__
    DDouble() : value(0),
                index(-1),
                globalIndex(nullptr),
                operation(CONST),
                argumentCount(0),
                operatorTreeSize(1),
                derivative(0.0),
                constant(true) {
    }

    __host__ __device__ DDouble operator*(DDouble &other) {
        DDouble mul = DDouble(value * other.value, *globalIndex, globalIndex, MUL, 2);
        mul.arguments[0] = this;
        mul.arguments[1] = &other;
        mul.operatorTreeSize = this->operatorTreeSize + other.operatorTreeSize + 1;
        return mul;
    }

    __host__ __device__ DDouble operator/(DDouble &other) {
        DDouble div = DDouble(value / other.value, *globalIndex, globalIndex, DIV, 2);
        div.arguments[0] = this;
        div.arguments[1] = &other;
        div.operatorTreeSize = this->operatorTreeSize + other.operatorTreeSize + 1;
        return div;
    }

    __host__ __device__ DDouble operator+(DDouble &other) {
        DDouble plus = DDouble(value + other.value, *globalIndex, globalIndex, PLUS, 2);
        plus.arguments[0] = this;
        plus.arguments[1] = &other;
        plus.operatorTreeSize = this->operatorTreeSize + other.operatorTreeSize + 1;
        return plus;
    }

    __host__ __device__ DDouble operator-(DDouble &other) {
        DDouble minus = DDouble(value - other.value, *globalIndex, globalIndex, MINUS, 2);
        minus.arguments[0] = this;
        minus.arguments[1] = &other;
        minus.operatorTreeSize = this->operatorTreeSize + other.operatorTreeSize + 1;
        return minus;
    }

    __host__ __device__ DDouble square() {
        DDouble square = DDouble(value * value, *globalIndex, globalIndex, SQUARE, 1);

        square.arguments[0] = this;
        square.operatorTreeSize = this->operatorTreeSize + 1;
        return square;
    }

    __host__ __device__ DDouble sqrt() {
        DDouble root = DDouble(std::sqrt(value), *globalIndex, globalIndex, SQRT, 1);
        root.arguments[0] = this;
        root.operatorTreeSize = this->operatorTreeSize + 1;
        return root;
    }

    __host__ __device__ DDouble inverse() {
        DDouble inverse = DDouble(1.0 / value, *globalIndex, globalIndex, INVERSE, 1);
        inverse.arguments[0] = this;
        inverse.operatorTreeSize = this->operatorTreeSize + 1;
        return inverse;
    }

    __host__ __device__ DDouble sin() {
        DDouble sin = DDouble(std::sin(value), *globalIndex, globalIndex, SIN, 1);
        sin.arguments[0] = this;
        sin.operatorTreeSize = this->operatorTreeSize + 1;
        return sin;
    }

    __host__ __device__ DDouble cos() {
        DDouble cos = DDouble(std::cos(value), *globalIndex, globalIndex, COS, 1);
        cos.arguments[0] = this;
        cos.operatorTreeSize = this->operatorTreeSize + 1;
        return cos;
    }

    __host__ __device__ bool operator<(const DDouble &other) {
        return this->value < other.value;
    }

    __host__ __device__ bool operator>(const DDouble &other) {
        return this->value > other.value;
    }

    __host__ __device__  void
    setPartialDerivatives(DDouble *parameterList) {
        derivative = 1.0;
        for (int i = index; i >= 0; i--) {
            propagateDerivative(parameterList[i], parameterList);
        }
    }

    __host__ __device__ void propagateDerivative(DDouble &node, DDouble *parameterList) { // TODO optimize dereferencing
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


#endif //PARALLELLBFGS_DDOUBLE_CUH
