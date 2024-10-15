import sys
import os

sys.path.insert(0, '../')
sys.path.insert(0, '../..')

import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold


# def objective(params):
#     params = {
#         "optimizer":str(params['optimizer']),
#         "activation":str(params['activation']),
#         'batch_size': abs(int(params['batch_size'])),
#         'neurons': abs(int(params['neurons'])),
#         'epochs': abs(int(params['epochs'])),
#         'patience': abs(int(params['patience']))
#     }
#     clf = KerasClassifier(build_fn=ANN,**params, verbose=0)
#     score = -np.mean(cross_val_score(clf, X, y, cv=3, 
#                                     scoring="accuracy"))

#     return {'loss':score, 'status': STATUS_OK }


def add_pre_evaluated_point(trials,tid, params, loss):
    trial = {
        'tid': tid,
        'result': {'status': STATUS_OK, 'loss': loss},
        'misc': {
            'tid': tid,
            'cmd': ('domain_attachment', 'FMinIter_Domain'),
            'workdir': None,
            'idxs': {k: [tid] for k in params},
            'vals': {k: [v] for k, v in params.items()}
        },
        'spec': None,
        'state': 2,
        'owner': None,
        'book_time': None,
        'refresh_time': None,
        'exp_key': None
    }
    trials.insert_trial_doc(trial)
    return trials

def generate_trials_to_calculate(points,results):
    """
    Function that generates trials to be evaluated from list of points

    :param points: List of points to be inserted in trials object in form of
        dictionary with variable names as keys and variable values as dict
        values. Example value:
        [{'x': 0.0, 'y': 0.0}, {'x': 1.0, 'y': 1.0}]

    :return: object of class base.Trials() with points which will be calculated
        before optimisation start if passed to fmin().
    """
    trials = Trials()
    [add_pre_evaluated_point(trials,tid,x,y) for tid, (x,y) in enumerate(zip(points,results))]
    return trials

def snapToType(params,func):
    listparams=[]
    for (i,(lower,upper,type)) in enumerate(zip(func.lowerbounds,func.upperbounds,func.xtypes)):
          if type == 'discrete':
               listparams.append(int(np.round(params[i])))
          else:
               listparams.append(params[i])
    return listparams

def toTPEParams(func):
    tpeParamconfig={}
    for (i,(lower,upper,type)) in enumerate(zip(func.lowerbounds,func.upperbounds,func.xtypes)):
            if type == 'continuous':
                tpeParamconfig[str(i)]=(hp.uniform(str(i),float(lower),float(upper)))
            elif type == 'discrete':
                tpeParamconfig[str(i)]=(hp.quniform(str(i),int(lower),int(upper),1))
            else:
                print(f'Invalid func config {func}')
                exit 
    return tpeParamconfig

def unflatten(flatParams):
    unflattened={}
    for i,val in enumerate(flatParams):
        unflattened[str(i)]=val
    return unflattened

def wrappedFunc(params,func):
    flattened=[]
    for i in range(len(params.keys())):
        flattened.append(params[str(i)])
    return func(flattened)

def trialsToXY(trials,lastn):
    X=[ [ misc['vals'][str(keyIndex)][0] for  keyIndex in range(len(misc['vals'].keys()))  ]  for misc in trials.miscs[-lastn:]]
    Y=[ result['loss'] for result in trials.results[-lastn:]]
    return X,Y

def bayesTPE(hist,func):    
    histsize=len(hist.population_history)
    X = []
    Y = []
    for pop in hist.population_history[-(min(histsize,150)):]:
        for ind in pop:
            X.append(snapToType(ind,func))
    for popvalues in hist.population_values_history[-(min(histsize,150)):]:
        for value in popvalues:
            Y.append(value)
    tpeParams=toTPEParams(func)

    if len(hist.population_history)>1:
        trials=generate_trials_to_calculate([unflatten(x) for x in X],Y)
    else:
        trials=Trials()
    best = fmin(fn=lambda params: wrappedFunc(params,func),
            space=tpeParams,
            algo=tpe.suggest,
            max_evals=len(hist.population_history[0]),
            trials=trials)
    # print(best)
    newX,newY=trialsToXY(trials,len(hist.population_history[0]))
    return newX,newY
    # popsize=len(hist.population_history[0])
    # newpointcount=popsize
    # res_gp = fmin(lambda params: wrappedFunc(params,func), tpeParams,x0=X,y0=Y, n_calls=newpointcount, n_initial_points=0,random_state=0,verbose=True,n_jobs=-1)
    # return res_gp.x_iters[-newpointcount:],res_gp.func_vals[-newpointcount:] 