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

def calculateSpeedup(threadsSecDF):
    sequentialTime=threadsSecDF[threadsSecDF['threads']==1].iloc[0]['sec']
    threadsSecDF['seqTime']=sequentialTime
    threadsSecDF=threadsSecDF.rename(columns={'sec':'parTime'})
    threadsSecDF['speedup']=threadsSecDF['seqTime']/threadsSecDF['parTime']
    return threadsSecDF

def ScalabilityAnalysis():
    scalabilityView=createTestGroupView(
                                    recordsPath=SCALABILITY_EXPERIMENT_RECORDS_PATH,
                                    getMetricsAndId=(noMetric,"hashSHA256"),
                                    mapRecordToExperiment=recordToScalabilityExperiment,
                                    explanatoryColumns=set(),
                                    responseColumns=set(["sec"]),
                                    metricsAggregation={'sec':'mean'},
                                    enrichAndFilter=justAggregations)
    scalabilityView=scalabilityView.drop(['problemPath'],axis=1)                                    
    scalabilityView=scalabilityView[['hyperLevelMethod','modelSize','threads','sec']]
    scalabilityView=scalabilityView.sort_values(by=['hyperLevelMethod','modelSize','threads'])
    scalabilityView=scalabilityView.groupby(['modelSize','hyperLevelMethod']).apply(calculateSpeedup)
    scalabilityView=scalabilityView[scalabilityView['threads']>1]
    print(scalabilityView.to_latex(index=False))

ScalabilityAnalysis()