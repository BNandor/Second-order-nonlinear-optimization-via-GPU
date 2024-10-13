import sys
import os

sys.path.insert(0, '../')
sys.path.insert(0, '../..')

import numpy as np
import json
import subprocess
import itertools
from skopt.space import Real, Integer,Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize


def pyNMHHHyperParametersToGP(paramConfig):
    gpParamconfig=[]
    for (key,value) in paramConfig.items():
            if isinstance(value[0], int):
                gpParamconfig.append(Integer(value[0],value[1],name=key))
            elif isinstance(value[0], float):
                gpParamconfig.append(Real(value[0],value[1],name=key))
            else:
                gpParamconfig.append(Categorical(value,name=key))
    return gpParamconfig

def toGPParams(func):
    gpParamconfig=[]
    for (i,(lower,upper,type)) in enumerate(zip(func.lowerbounds,func.upperbounds,func.xtypes)):
            if type == 'continuous':
                gpParamconfig.append(Real(lower,upper,name=str(i)))
            elif type == 'discrete':
                gpParamconfig.append(Integer(lower,upper,name=str(i)))
            else:
                print(f'Invalid func config {func}')
                exit 
    return gpParamconfig

def snapToType(params,func):
    listparams=[]
    for (i,(lower,upper,type)) in enumerate(zip(func.lowerbounds,func.upperbounds,func.xtypes)):
          if type == 'discrete':
               listparams.append(int(np.round(params[i])))
          else:
               listparams.append(params[i])
    return listparams

def unflatten(flatParams,paramConfig):
    unflattened=dict()
    for (flatValue,(key,value)) in zip(flatParams,paramConfig.items()):
            if isinstance(value[0], int):
                unflattened[key]=int(flatValue)
            elif isinstance(value[0], float):
                unflattened[key]=float(flatValue)
            else:
                unflattened[key]=flatValue
    return unflattened

def bayesGP(hist,func):    
    histsize=len(hist.population_history)
    X = []
    Y = []
    for pop in hist.population_history[-(min(histsize,150)):]:
        for ind in pop:
            X.append(snapToType(ind,func))
    for popvalues in hist.population_values_history[-(min(histsize,150)):]:
        for value in popvalues:
            Y.append(value)
    gpParams=toGPParams(func)
    popsize=len(hist.population_history[0])
    newpointcount=popsize
    res_gp = gp_minimize(func, gpParams,x0=X,y0=Y, n_calls=newpointcount, n_initial_points=0,random_state=0,verbose=True,n_jobs=-1)
    return res_gp.x_iters[-newpointcount:],res_gp.func_vals[-newpointcount:] 