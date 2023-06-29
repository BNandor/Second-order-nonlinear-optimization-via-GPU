import sys
import os
sys.path.insert(0, '../../../')

from commonAnalysis import *
from importData import *
from dict_hash import sha256
from commonPlots import *
from common import *
import tabloo
import numpy as np

def SATempAnalysis():
    # (experimentColumns,minF,HH-SA-Temp,HH-SA-alpha)
    # {
    tempView=createTestGroupView(SA_EXPERIMENT_RECORDS_PATH,
                                    (filterMetricPropertiesMinMedIQR,"hashSHA256"),
                                    recordToExperiment,
                                    set(['HH-SA-temp','HH-SA-alpha','trialStepCount']),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    enrichAndFilterSATemp)
    # }
    # tempView=createTestGroupView(SA_EXPERIMENT_RECORDS_PATH,
    #                                 (filterMetricPropertiesMinMedIQR,"hashSHA256"),
    #                                 recordToExperiment,
    #                                 set(),
    #                                 set(["minMedIQR"]),
    #                                 {'minMedIQR':'min'},
    #                                 enrichAndFilterSATemp)
    
    tempView=tempView.drop(['problemName','populationSize','problemPath','trialSampleSizes'],axis=1)
    # tempView=tempView.reset_index(drop=True)
    # print(tempView)
    testSet=[("baselevelIterations",[100,5000]),('modelSize',[50,100,500]),('trialStepCount',[50,100,200]),('HH-SA-temp',[1000,10000]),("HH-SA-alpha",[5,50])]
    tempView=tempView[  selectAllMatchAtLeastOne(tempView,testSet) ]
    tempView=tempView.reset_index(drop=True)
    # print(tempView)
    tempView=tempView[['modelSize','baselevelIterations','minMedIQR','trialStepCount','HH-SA-temp',"HH-SA-alpha"]]
    tempView=tempView.sort_values(by=['modelSize','baselevelIterations'])
    tempView=tempView.reset_index(drop=True)
    # print(tempView)
    print(tempView.to_latex(index=False))
    # { 
    tempRankView=tempView.groupby(['HH-SA-temp']).count().reset_index()[['HH-SA-temp','HH-SA-alpha','trialStepCount','modelSize']]
    tempRankView=tempRankView.rename({'modelSize':'bestCount'})
    print(tempRankView)
    # print(tempRankView.to_html())
    # }

SATempAnalysis()