import sys
import os
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from commonAnalysis import *
from importData import *
from hhanalysis.experiment.analysis.cases.customhys.loadCustomHYS import *
from dict_hash import sha256
from commonPlots import *
from common import *
import tabloo
import numpy as np

methods=[     
                #   (BIGSA_MADS_NMHH_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH,"bigSA_MADS"),
                #   (SA_MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,"SA_MADS"),
                #   (MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,"MADS"),
                  (SA_CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,"SA_CMAES"),
                  (BIGSA_CMA_ES_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH,"bigSA_CMAES"),
                  (CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,"CMAES"),
                  (SA_GA_DE_GD_LBFGS_RECORDS_PATH,"SA-NMHH")
                  ]
problems=[('rosenbrock.json','log'),
              ('rastrigin.json','log'),
              ('styblinskitang.json','linear'),('trid.json','linear'),
              ('schwefel223.json','log'),
              ('qing.json','log')]
dimensions=[5,50,100,500,750]
performanceMapping=lambda trial: trial['med_+_iqr']
filter=[('baseLevelEvals',[100]),('baseLevel-xDim',dimensions)]
createMethodsCostEvolutionPlots(methods,problems,performanceMapping,filter)