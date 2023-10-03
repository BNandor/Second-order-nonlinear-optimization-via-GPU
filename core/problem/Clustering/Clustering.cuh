//
// Created by spaceman on 2023. 08. 16..
//

#ifndef PARALLELLBFGS_CLUSTERING_CUH
#define PARALLELLBFGS_CLUSTERING_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"
#include <math.h>
#include <limits>

class Clustering : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 5 * PROBLEM_CLUSTERING_POINT_DIM ;
    static const unsigned ThisParameterSize =  PROBLEM_CLUSTERING_POINT_DIM;
    static const unsigned ThisConstantSize = PROBLEM_CLUSTERING_POINT_DIM;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    Clustering() {

        operatorTreeSize = ThisOperatorTreeSize;
        parameterSize = ThisParameterSize;
        constantSize = ThisConstantSize;
        operatorTree = ThisOperatorTree;
        jacobianIndices = ThisJacobianIndices;
        initIndex();
    }

    __device__ __host__
    void setConstants(double *constants, unsigned constantsSize) {
        initConst(constants, constantsSize);
    }

    __device__ __host__
    DDouble *eval(double *x, unsigned xSize) {
        //Calculate closest point
        double minDist=std::numeric_limits<double>::max();
        int minIndex=0;
        double dist;
        double d;
        for(int i=0;i<PROBLEM_CLUSTERING_K;i++) {
            dist=0;
            for(int j=0;j<PROBLEM_CLUSTERING_POINT_DIM;j++){
                d=(x[i*PROBLEM_CLUSTERING_POINT_DIM + j]-operatorTree[j].value);
                dist+=d*d;
            }
//            dist=sqrt(dist);
            if(dist<minDist){
                minIndex=i;
                minDist=dist;
            }
        }
        // (c(1)-x(0))^2 // c(0) index of x(0)
        initOperatorTreePartially(x, minIndex*PROBLEM_CLUSTERING_POINT_DIM ,PROBLEM_CLUSTERING_POINT_DIM , 0);
        for(int i=0;i<PROBLEM_CLUSTERING_POINT_DIM;i++){
            operatorTree[2*PROBLEM_CLUSTERING_POINT_DIM+i]=operatorTree[PROBLEM_CLUSTERING_POINT_DIM+i]-operatorTree[i];
            operatorTree[3*PROBLEM_CLUSTERING_POINT_DIM+i]=operatorTree[2*PROBLEM_CLUSTERING_POINT_DIM+i].square();
        }
        operatorTree[4*PROBLEM_CLUSTERING_POINT_DIM]=operatorTree[3*PROBLEM_CLUSTERING_POINT_DIM]+operatorTree[3*PROBLEM_CLUSTERING_POINT_DIM+1];
        for(int i=2;i<PROBLEM_CLUSTERING_POINT_DIM;i++){
            operatorTree[4*PROBLEM_CLUSTERING_POINT_DIM+i-1]=operatorTree[4*PROBLEM_CLUSTERING_POINT_DIM+i-2]+operatorTree[3*PROBLEM_CLUSTERING_POINT_DIM+i];
        }
        operatorTree[5*PROBLEM_CLUSTERING_POINT_DIM-1]=operatorTree[5*PROBLEM_CLUSTERING_POINT_DIM-2].sqrt();
        return &operatorTree[5*PROBLEM_CLUSTERING_POINT_DIM-1];
    }
};

#endif //PARALLELLBFGS_CLUSTERING_CUH
