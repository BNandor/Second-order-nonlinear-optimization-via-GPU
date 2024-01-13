import sys
import os
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from commonAnalysis import *
from importData import *
# from loadCustomHYS import *
from customhys.loadCustomHYS import *
import customhys.loadCustomHyS2 
import loadHHData  
from dict_hash import sha256
from commonPlots import *
from common import *
import tabloo
import numpy as np
import mealpyExp.loadMealpy
import loadHHData

mealpyCRODF=loadHHData.loadDataMap()['mealpyCRO']( {
                                                    "minMetricColumn":'minAvg',
                                                    "metricsAggregation":{'minAvg':'min'},
                                                    "mergeOn":mergeOnAvg,
                                                    "datasets":{}})
mealpyCRODF=mealpyExp.loadMealpy.expandTrials(mealpyCRODF)
methods=[     
                  # (BIGSA_MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,"BIGSA-MADS-GWO"),
                  # (SA_MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,"SA_MADS"),
                  # (MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,"MADS"),
                  # (SA_CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,"SA_CMAES"),
                  # (BIGSA_CMA_ES_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH,"bigSA_CMAES"),
                  # (CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,"CMAES"),
                  (SA_GA_DE_GD_LBFGS_RECORDS_PATH,"NMHH",{}),
                  (RANDOM_CONTROL_GROUP_EXPERIMENT_RECORDS_PATH,"Random-NMHH",{}),
                  # (SAPERTURBGWO_EXPERIMENT_RECORDS_PATH,"SA-PERTURB",{}),
                  # (RANDOM_GA_EXPERIMENT_RECORDS_PATH,"RANDOM-GA",{}),
                  # (RANDOM_DE_EXPERIMENT_RECORDS_PATH,"RANDOM-DE",{}),
                  (CUSTOMHYS2_RESULTS_PATH,"CustomhyS",{"customDF":customhys.loadCustomHyS2.loadCustomHysDF(),
                                                       'problemFilter':customhys.loadCustomHyS2.selectBasedOnRecordsMapping}),
                  # (MEALPY_CRO_EXPERIMENT_RECORDS_PATH,"CRO",{"customDF":mealpyCRODF,
                  #                                            'problemFilter':mealpyExp.loadMealpy.selectBasedOnRecordsMapping})
                  ]
problems=[    
            # ('rosenbrock.json','log'),
              # ('rastrigin.json','log'),
              ('styblinskitang.json','linear'),
              # ('trid.json','linear'),
              # ('schwefel223.json','log'),
              # ('qing.json','log')
              ]
# dimensions=[30,50,100,500,750]
dimensions=[5,50,100,500]
performanceMapping=lambda trial: trial['med_+_iqr']
filter=[('baseLevelEvals',[100]),('baseLevel-xDim',dimensions)]

createMethodsCostEvolutionPlots(methods,problems,performanceMapping,filter,filename=f'{ROOT}/plots/peerj/styblinskitang.svg')