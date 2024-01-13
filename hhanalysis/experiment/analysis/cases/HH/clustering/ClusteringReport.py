import sys
import os

sys.path.insert(1, '../../..')
sys.path.insert(2, '..')
# sys.path.insert(0, '../../..')

from commonAnalysis import *
from importData import *
from dict_hash import sha256
from commonPlots import *
from common import *
import tabloo
import numpy as np
from loadHHData import *
from  commonHHAnalysis import *

def compare(methodExperiments):
    metadata={
            "minMetricColumn":'minAvg',
            "metricsAggregation":{'minAvg':'min'},
            "mergeOn":mergeOnAvg,
            # "minMetricColumn":'minMedIQR',
            # "metricsAggregation":{'minMedIQR':'min'},
            # "mergeOn":mergeOnMinMedIQR,
            'optimizers':list(),
            "saveMetadataColumns":["minMetricColumn",'optimizers','baselevelIterations'],
            "datasets":{}
        }
    hhdata=loadDataMap()
    all=pd.concat([hhdata[methodAndExperiment[0]](metadata,methodAndExperiment[1]) for methodAndExperiment in methodExperiments])
    all=dropIrrelevantColumns(all,set(['populationSize','modelSize','problemName','hyperLevel-id','baselevelIterations','minAvg','minStd','minMedIQR','samples']))
    tabloo.show(all)
    # methodsComparison(all,metadata,True)

allmethods=[('clustering','/cmc/smalliter'),
            ('clustering','/glass/smalliter'),
            ('clustering','/iris/constrained'),
            ('clustering','/wine/constrained')]
compare(allmethods)