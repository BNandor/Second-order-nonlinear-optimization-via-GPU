//
// Created by spaceman on 2022. 02. 12..
//

#ifndef PARALLELLBFGS_GD_CUH
#define PARALLELLBFGS_GD_CUH

#include "../AD/DDouble.cuh"
#include "../AD/function/DFunction.cuh"
#include "../AD/function/DPlusFunction.cuh"
#include "../AD/function/DMultiplicationFunction.cuh"
#include "../AD/function/DMinusFunction.cuh"
#include "../AD/function/DIDFunction.cuh"
#include "../AD/function/DSquareFunction.cuh"
#include "../AD/function/Operations.cuh"
#include "../problem/F1.cuh"
#include "../problem/PlaneFitting.cuh"
#include <stdio.h>

__global__ void testPlaneFitting(double *globalX, double *globalDX, double *globalF,
                                 double *data) { // use shared memory instead of global memory


}

#endif //PARALLELLBFGS_GD_CUH
