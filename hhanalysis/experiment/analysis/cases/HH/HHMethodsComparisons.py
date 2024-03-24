import sys
import os
import re
sys.path.insert(0, '../..')

from commonAnalysis import *
from importData import *
from dict_hash import sha256
from commonPlots import *
from common import *
import tabloo
import numpy as np
from loadHHData import *
from  commonHHAnalysis import *

colorMap=[
    (set(['CRO','PSO','BRO','ArchOA','AEO','SMA']),'orange'),
    (set(['NMHH']),'blue'),
    (set(['CUSTOMHyS']),'black'),
    (set(['SA-MADS','SA-CMA-ES']),'green')
]

def mapBarplotColors(optimizer):
    for  colors in colorMap:
        for category in colors[0]:
            if category in optimizer:
                # if 'GWO' in optimizer:
                #     return f'dark {colors[1]}'
                return colors[1]
    print(f'could not map {optimizer} to a valid color, check colormap!')
    return 'purple'

def optimizerPrettify(optimizer):
    noPathInName=optimizer.split('/')[-1]
    noNumbers=re.sub(r'\d+', '',noPathInName)
    return noNumbers[1:] if noNumbers.startswith('-') else noNumbers

def compare(methodExperiments,problems,dimensions):
    metadata={
            "minMetricColumn":'minAvg',
            "metricsAggregation":{'minAvg':'min'},
            "mergeOn":mergeOnAvg,
            # "minMetricColumn":'minMedIQR',
            # "metricsAggregation":{'minMedIQR':'min'},
            # "mergeOn":mergeOnMinMedIQR,
            'optimizers':list(),
            "saveMetadataColumns":["minMetricColumn",'optimizers','baselevelIterations'],
            "baselevelIterations": [100],
            "problems":problems,
            "modelSize":dimensions,
            "datasets":{}
        }
    hhdata=loadDataMap()
    all=pd.concat([hhdata[methodAndExperiment[0]](metadata,methodAndExperiment[1]) for methodAndExperiment in methodExperiments])

    all=dropIrrelevantColumns(all,set(['modelSize','problemName','hyperLevel-id','baselevelIterations','minAvg','minStd','minMedIQR','samples']))
    all=all[selectAllMatchAtLeastOne(all,[('baselevelIterations',metadata["baselevelIterations"]),('modelSize',metadata["modelSize"]),('problemName',metadata["problems"])])]
    all=all.sort_values(by=['modelSize',"problemName",metadata["minMetricColumn"]])
    # all=all[['problemName','modelSize','hyperLevel-id',metadata["minMetricColumn"],'minStd','samples']]

    print(f"Problems: {problems} \n Dimensions: {dimensions}")
    methodsComparison(all,metadata,True,mapBarplotColors,optimizerPrettify)

allmethods=[
            ('nmhh2','/'),
            ('nmhh2','/sprt-seq-t'),
            # ('saGWOGroup','/'),
            # ('sacmaesGWOGroup','/'),
            # ('bigsacmaesGWOGroup','/'),
            # ('bigsacmaesGroup','/'),
            # ('sacmaesGroup','/'),   
            # ('cmaesGWOGroup','/'),
            # ('cmaesGroup','/'),
            # ('bigsamadsGroup','/'),
            # ('bigsamadsGWOGroup','/'),
            # ('saMadsGWOGroup','/'),
            # ('madsGWOGroup','/'),
            # ('madsGroup','/'),
            # ('saMadsGroup','/'),

            # ('nmhh2','/bigDim/100/bigPop/100'),
            # ('saGWOGroup','/bigDim/100/bigPop/100'),
            # ('sacmaesGWOGroup','/bigDim/100/bigPop/100'),
            # ('bigsacmaesGWOGroup','/bigDim/100/bigPop/100'),
            # ('bigsacmaesGroup','/bigDim/100/bigPop/100'),
            # ('sacmaesGroup','/bigDim/100/bigPop/100'),   
            # ('cmaesGWOGroup','/bigDim/100/bigPop/100'),
            # ('cmaesGroup','/bigDim/100/bigPop/100'),
            # ('bigsamadsGroup','/bigDim/100/bigPop/100'),
            # ('bigsamadsGWOGroup','/bigDim/100/bigPop/100'),
            # ('saMadsGWOGroup','/bigDim/100/bigPop/100'),
            # ('madsGWOGroup','/bigDim/100/bigPop/100'),
            # ('madsGroup','/bigDim/100/bigPop/100'),
            # ('saMadsGroup','/bigDim/100/bigPop/100'),

            # ('customhys2','/'),
            # ('mealpy','/'),
            # ('mealpy','/benchmarks/dim/2_100/pop/30')
            ]

convexUnimodal=['PROBLEM_SCHWEFEL223','PROBLEM_TRID','PROBLEM_SPHERE','PROBLEM_SUMSQUARES']
nonconvexMultimodal=['PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_ROSENBROCK',
                     'PROBLEM_MICHALEWICZ','PROBLEM_DIXONPRICE','PROBLEM_LEVY','PROBLEM_SCHWEFEL']
nonSeparable=['PROBLEM_ROSENBROCK','PROBLEM_TRID','PROBLEM_DIXONPRICE','PROBLEM_LEVY']
separable=['PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_SCHWEFEL223','PROBLEM_MICHALEWICZ','PROBLEM_SCHWEFEL','PROBLEM_SPHERE','PROBLEM_SUMSQUARES']

initialproblems=['PROBLEM_SCHWEFEL223','PROBLEM_TRID','PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_ROSENBROCK']
extraProblems=['PROBLEM_MICHALEWICZ',
               'PROBLEM_DIXONPRICE','PROBLEM_LEVY','PROBLEM_SCHWEFEL','PROBLEM_SUMSQUARES','PROBLEM_SPHERE']
allproblems=initialproblems+extraProblems

#next benchmark functions
#http://www.sfu.ca/~ssurjano/michal.html
#http://www.sfu.ca/~ssurjano/dixonpr.html
#http://www.sfu.ca/~ssurjano/zakharov.html
#http://www.sfu.ca/~ssurjano/sumsqu.html
#http://www.sfu.ca/~ssurjano/sumpow.html
#http://www.sfu.ca/~ssurjano/spheref.html
#http://www.sfu.ca/~ssurjano/schwef.html
#http://www.sfu.ca/~ssurjano/levy.html

# Real optimization problems
# https://raw.githubusercontent.com/P-N-Suganthan/CEC-2011--Real_World_Problems/master/Tech-Rep.pdf
# http://www.sfu.ca/~ssurjano/borehole.html
# http://www.sfu.ca/~ssurjano/piston.html
# http://www.sfu.ca/~ssurjano/robot.html
# http://www.sfu.ca/~ssurjano/wingweight.html
# http://www.sfu.ca/~ssurjano/environ.html

initialdimensions=[5,50,100,500]
# alldimensions=[2,3,4,5,6,7,8,9,10,15,30,50,100,500,750]
alldimensions=[5,6,7,8,9,10,15,30,50,100,500,750]
# alldimensions=[100]
highdimensions=[10,15,30,50,100,500,750]
lowerdimensions=[2,3,4,5,6,7,8,9] 
singleDim=[2]
compare(allmethods,allproblems,[5])
# compare(allmethods,initialproblems,initialdimensions)


# print("All-Lower dimensions")
# compare(allmethods,allproblems,lowerdimensions)
# print("ConvexUnimodal-Lower dimensions")
# compare(allmethods,convexUnimodal,lowerdimensions)
# print("NonconvexMultimodal-Lower dimensions")
# compare(allmethods,nonconvexMultimodal,lowerdimensions)
# print("Separable-Lower dimensions")
# compare(allmethods,separable,lowerdimensions)
# print("NonSeparable-Lower dimensions")
# compare(allmethods,nonSeparable,lowerdimensions)

# print("All-High dimensions")
# compare(allmethods,allproblems,highdimensions)
# print("ConvexUnimodal-High dimensions")
# compare(allmethods,convexUnimodal,highdimensions)
# print("NonconvexMultimodal-High dimensions")
# compare(allmethods,nonconvexMultimodal,highdimensions)
# print("Separable-High dimensions")
# compare(allmethods,separable,highdimensions)
# print("NonSeparable-High dimensions")
# compare(allmethods,nonSeparable,highdimensions)