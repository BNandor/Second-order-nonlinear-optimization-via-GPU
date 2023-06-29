import sys
import os
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

def compare(methods,problems,dimensions):
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
            # "modelSize":[1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750] 
            "modelSize":dimensions,
            "datasets":{}
            # "modelSize":[1,2,3,4,5,6,7,8,9] 
            # "modelSize":[10,15] 
        }
    hhdata=loadDataMap()
    all=pd.concat([hhdata[method](metadata) for method in methods])

    all=dropIrrelevantColumns(all,set(['modelSize','problemName','hyperLevel-id','baselevelIterations','minAvg','minStd','minMedIQR','samples']))
    all=all[selectAllMatchAtLeastOne(all,[('baselevelIterations',metadata["baselevelIterations"]),('modelSize',metadata["modelSize"]),('problemName',metadata["problems"])])]
    all=all.sort_values(by=['modelSize',"problemName",metadata["minMetricColumn"]])
    # all=all[['problemName','modelSize','hyperLevel-id',metadata["minMetricColumn"],'minStd','samples']]

    print(f"Problems: {problems} \n Dimensions: {dimensions}")
    methodsComparison(all,metadata,True)

allmethods=['nmhh2','saGWOGroup','bigsacmaesGroup','bigsacmaesGWOGroup','sacmaesGWOGroup','sacmaesGroup','cmaesGWOGroup','cmaesGroup','bigsamadsGroup','bigsamadsGWOGroup','saMadsGWOGroup','madsGWOGroup','madsGroup','saMadsGroup']

convexUnimodal=['PROBLEM_SCHWEFEL223','PROBLEM_TRID']
nonconvexMultimodal=['PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_ROSENBROCK']
nonSeparable=['PROBLEM_ROSENBROCK','PROBLEM_TRID']
separable=['PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_SCHWEFEL223']
allproblems=['PROBLEM_SCHWEFEL223','PROBLEM_TRID','PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_ROSENBROCK']

alldimensions=[1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750]
highdimensions=[30,50,100,500,750]
lowerdimensions=[1,2,3,4,5,6]

# compare(allmethods,allproblems,alldimensions)

# compare(allmethods,allproblems,lowerdimensions)
# compare(allmethods,allproblems,highdimensions)

# compare(allmethods,convexUnimodal,lowerdimensions)
# compare(allmethods,convexUnimodal,highdimensions)

compare(allmethods,nonconvexMultimodal,lowerdimensions)
compare(allmethods,nonconvexMultimodal,highdimensions)

# compare(allmethods,separable,lowerdimensions)
# compare(allmethods,separable,highdimensions)

# compare(allmethods,nonSeparable,lowerdimensions)
# compare(allmethods,nonSeparable,highdimensions)

