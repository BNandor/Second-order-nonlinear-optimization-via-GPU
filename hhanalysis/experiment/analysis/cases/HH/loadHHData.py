import sys
import os
sys.path.insert(0, '../..')
sys.path.insert(0, '../')
sys.path.insert(0, '../customhys')

from commonAnalysis import *
from importData import *
from dict_hash import sha256
from commonPlots import *
from common import *
import tabloo
import numpy as np
from customhys.loadCustomHYS import *
from customhys.loadCustomHyS2 import *
import  mealpyExp.loadMealpy 

def customhys(metadata):
    if "customhys" not in metadata['datasets']:
            metadata['datasets']["customhys"]=getCustomHySControlGroupDF()
            metadata['datasets']["customhys"]['hyperLevel-id']='CUSTOMHyS'
    return metadata['datasets']["customhys"]

def customhys2(metadata):
    if "customhys2" not in metadata['datasets']:
        metadata['datasets']["customhys2"]=createTestGroupView(CUSTOMHYS2_RESULTS_PATH,
                                    (None,"hashSHA256"),
                                    customhys2RecordToExperiment,
                                    set(),
                                    set(["minMedIQR","minAvg","minStd","samples"]),
                                    metadata['metricsAggregation'],
                                    metadata['mergeOn'],enrichWithMetrics=False)
        metadata['datasets']["customhys2"]['hyperLevel-id']='CustomHyS'
    return metadata['datasets']["customhys2"]

def mealpy(metadata):
    if "mealpy" not in metadata['datasets']:
        metadata['datasets']["mealpy"]=createTestGroupView(MEALPY_EXPERIMENT_RECORDS_PATH,
                                    (None,"hashSHA256"),
                                    mealpyRecordToExperiment,
                                    set(),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    enrichAndFilterMealpy,enrichWithMetrics=False)
        # metadata['datasets']["mealpy"]['hyperLevel-id']='MEALPY'
    return metadata['datasets']["mealpy"]

def mealpyCRO(metadata):
    if "mealpyCRO" not in metadata['datasets']:
        metadata['datasets']["mealpyCRO"]=createTestGroupView(MEALPY_CRO_EXPERIMENT_RECORDS_PATH,
                                    (None,"hashSHA256"),
                                    mealpyExp.loadMealpy.recordToExperiment,
                                    set(),
                                    set(["minMedIQR","minAvg","minStd","samples"]),
                                    metadata['metricsAggregation'],
                                    metadata['mergeOn'],enrichWithMetrics=False)
        metadata['datasets']["mealpyCRO"]=pd.DataFrame(metadata['datasets']["mealpyCRO"][metadata['datasets']["mealpyCRO"]['hyperLevel-id'] == 'CRO'])
        # metadata['datasets']["mealpy"]['hyperLevel-id']='MEALPY'
    return metadata['datasets']["mealpyCRO"]

def nmhh(metadata):
    if "nmhh" not in metadata['datasets']:
        metadata['datasets']["nmhh"]=createTestGroupView(SA_EXPERIMENT_RECORDS_PATH,
                                    (filterMetricPropertiesMinMedIQR,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    enrichAndFilterSA)
        metadata['datasets']["nmhh"]['hyperLevel-id']='NMHH'
    return metadata['datasets']["nmhh"]

def nmhh2(metadata):
    if "nmhh2" not in metadata['datasets']:
        metadata['datasets']["nmhh2"]=createTestGroupView(SA_GA_DE_GD_LBFGS_RECORDS_PATH,
                                    (filterMetricPropertiesAverageAndMedIQR,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR","minAvg","minStd","samples"]),
                                    metadata['metricsAggregation'],
                                    metadata['mergeOn']) 
        metadata['datasets']["nmhh2"]['hyperLevel-id']='NMHH'
    return metadata['datasets']["nmhh2"]

def sarefineGroup(metadata):
    if "sarefineGroup" not in metadata['datasets']:
        metadata['datasets']["sarefineGroup"] = createTestGroupView(
            SAREFINE_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesMinMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR"]),
            {'minMedIQR': 'min'},
            enrichAndFilterSA)
        metadata['datasets']["sarefineGroup"]['hyperLevel-id']='LBFGS'
    return metadata['datasets']["sarefineGroup"]

def lbfgsGroup(metadata):
    if "lbfgsGroup" not in metadata['datasets']:
        metadata['datasets']["lbfgsGroup"] = createTestGroupView(
            LBFGS_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesMinMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR"]),
            {'minMedIQR': 'min'},
            justAggregations
        )
        metadata['datasets']["lbfgsGroup"]['hyperLevel-id']='LBFGS'
    return metadata['datasets']["lbfgsGroup"]

def gdGroup(metadata):
    if "gdGroup" not in metadata['datasets']:
        metadata['datasets']["gdGroup"] = createTestGroupView(
            GD_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesMinMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR"]),
            {'minMedIQR': 'min'},
            justAggregations)
        metadata['datasets']["gdGroup"]['hyperLevel-id']='GD'
    return metadata['datasets']["gdGroup"]

def saperturbGroup(metadata):
    if "saperturbGroup" not in metadata['datasets']:
        metadata['datasets']["saperturbGroup"] = createTestGroupView(
            SAPERTURB_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["saperturbGroup"]['hyperLevel-id']='SA-PERTURB'
    return metadata['datasets']["saperturbGroup"]

def saperturbGWOGroup(metadata):
    if "saperturbGWOGroup" not in metadata['datasets']:
        metadata['datasets']["saperturbGWOGroup"] = createTestGroupView(
            SAPERTURBGWO_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["saperturbGWOGroup"]['hyperLevel-id']='SA-PERTURB'
    return metadata['datasets']["saperturbGWOGroup"]

def saperturbMultiOperatorGroup(metadata):
    if "saperturbMultiOperatorGroup" not in metadata['datasets']:
        metadata['datasets']["saperturbMultiOperatorGroup"] = createTestGroupView(
            SAPERTURBMULTIOPERATORS_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["saperturbMultiOperatorGroup"]['hyperLevel-id']='SA-PERTURBMultiOperator'
    return metadata['datasets']["saperturbMultiOperatorGroup"]

def gaGroup(metadata):
    if "gaGroup" not in metadata['datasets']:
        metadata['datasets']["gaGroup"] = createTestGroupView(
            GA_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["gaGroup"]['hyperLevel-id']='GA'
    return metadata['datasets']["gaGroup"]
def deGroup(metadata):
    if "deGroup" not in metadata['datasets']:
        metadata['datasets']["deGroup"] = createTestGroupView(
            DE_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["deGroup"]['hyperLevel-id'] = 'DE'
    return metadata['datasets']["deGroup"]

def randomgaGroup(metadata):
    if "randomgaGroup" not in metadata['datasets']:
        metadata['datasets']["randomgaGroup"] = createTestGroupView(
            RANDOM_GA_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["randomgaGroup"]['hyperLevel-id'] = 'RANDOM-GA'
    return metadata['datasets']["randomgaGroup"]

def randomdeGroup(metadata):
    if "randomdeGroup" not in metadata['datasets']:
        metadata['datasets']["randomdeGroup"] = createTestGroupView(
            RANDOM_DE_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["randomdeGroup"]['hyperLevel-id'] = 'RANDOM-DE'
    return metadata['datasets']["randomdeGroup"]

def saGWOGroup(metadata):
    if "saGWOGroup" not in metadata['datasets']:
        metadata['datasets']["saGWOGroup"] = createTestGroupView(
            RANDOM_SA_GWO_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn'],
            True
        )
        metadata['datasets']["saGWOGroup"]['hyperLevel-id'] = 'SA-GWO'
    return metadata['datasets']["saGWOGroup"]

def madsGWOGroup(metadata):
    if "madsGWOGroup" not in metadata['datasets']:
        metadata['datasets']["madsGWOGroup"] = createTestGroupView(
            MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["madsGWOGroup"]['hyperLevel-id'] = 'MADS-GWO'
    return metadata['datasets']["madsGWOGroup"]

def madsGroup(metadata):
    if "madsGroup" not in metadata['datasets']:
        metadata['datasets']["madsGroup"] = createTestGroupView(
            MADS_NMHH_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["madsGroup"]['hyperLevel-id'] = 'MADS'
    return metadata['datasets']["madsGroup"]

def sacmaesGWOGroup(metadata):
    if "sacmaesGWOGroup" not in metadata['datasets']:
        metadata['datasets']["sacmaesGWOGroup"] = createTestGroupView(
            SA_CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["sacmaesGWOGroup"]['hyperLevel-id'] = 'SA-CMA-ES-GWO'
    return metadata['datasets']["sacmaesGWOGroup"]

# Repeat the same pattern for other functions...

def saMadsGWOGroup(metadata):
    if "saMadsGWOGroup" not in metadata['datasets']:
        metadata['datasets']["saMadsGWOGroup"] = createTestGroupView(
            SA_MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["saMadsGWOGroup"]['hyperLevel-id'] = 'SA-MADS-GWO'
    return metadata['datasets']["saMadsGWOGroup"]

def saMadsGroup(metadata):
    if "saMadsGroup" not in metadata['datasets']:
        metadata['datasets']["saMadsGroup"] = createTestGroupView(
            SA_MADS_NMHH_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["saMadsGroup"]['hyperLevel-id'] = 'SA-MADS'
    return metadata['datasets']["saMadsGroup"]

def cmaesGWOGroup(metadata):
    if "cmaesGWOGroup" not in metadata['datasets']:
        metadata['datasets']["cmaesGWOGroup"] = createTestGroupView(
            CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["cmaesGWOGroup"]['hyperLevel-id'] = 'CMA-ES-GWO'
    return metadata['datasets']["cmaesGWOGroup"]

def cmaesGroup(metadata):
    if "cmaesGroup" not in metadata['datasets']:
        metadata['datasets']["cmaesGroup"] = createTestGroupView(
            CMA_ES_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["cmaesGroup"]['hyperLevel-id'] = 'CMA-ES'
    return metadata['datasets']["cmaesGroup"]

def bigsamadsGroup(metadata):
    if "bigsamadsGroup" not in metadata['datasets']:
        metadata['datasets']["bigsamadsGroup"] = createTestGroupView(
            BIGSA_MADS_NMHH_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["bigsamadsGroup"]['hyperLevel-id'] = 'BIGSA-MADS'
    return metadata['datasets']["bigsamadsGroup"]

def bigsamadsGWOGroup(metadata):
    if "bigsamadsGWOGroup" not in metadata['datasets']:
        metadata['datasets']["bigsamadsGWOGroup"] = createTestGroupView(
            BIGSA_MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["bigsamadsGWOGroup"]['hyperLevel-id'] = 'BIGSA-MADS-GWO'
    return metadata['datasets']["bigsamadsGWOGroup"]

def bigsacmaesGroup(metadata):
    if "bigsacmaesGroup" not in metadata['datasets']:
        metadata['datasets']["bigsacmaesGroup"] = createTestGroupView(
            BIGSA_CMA_ES_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["bigsacmaesGroup"]['hyperLevel-id'] = 'BIGSA-CMA-ES'
    return metadata['datasets']["bigsacmaesGroup"]


def bigsacmaesGWOGroup(metadata):
    if "bigsacmaesGWOGroup" not in metadata['datasets']:
        metadata['datasets']["bigsacmaesGWOGroup"] = createTestGroupView(
            BIGSA_CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["bigsacmaesGWOGroup"]['hyperLevel-id'] = 'BIGSA-CMA-ES-GWO'
    return metadata['datasets']["bigsacmaesGWOGroup"]


def sacmaesGroup(metadata):
    if "sacmaesGroup" not in metadata['datasets']:
        metadata['datasets']["sacmaesGroup"] = createTestGroupView(
            SA_CMA_ES_EXPERIMENT_RECORDS_PATH,
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets']["sacmaesGroup"]['hyperLevel-id'] = 'SA-CMA-ES'
    return metadata['datasets']["sacmaesGroup"]

def loadDataMap():
    return {
    "customhys":customhys,
    "customhys2":customhys2,
    "mealpy": mealpy,
    "mealpyCRO": mealpyCRO,
    "nmhh": nmhh,
    "nmhh2": nmhh2,
    "sarefineGroup": sarefineGroup,
    "lbfgsGroup": lbfgsGroup,
    "gdGroup": gdGroup,
    "saperturbGroup": saperturbGroup,
    "saperturbGWOGroup":saperturbGWOGroup,
    "saperturbMultiOperatorGroup":saperturbMultiOperatorGroup,
    "gaGroup": gaGroup,
    "deGroup": deGroup,
    "randomgaGroup": randomgaGroup,
    "randomdeGroup": randomdeGroup,
    "saGWOGroup": saGWOGroup,
    "madsGWOGroup": madsGWOGroup,
    "madsGroup": madsGroup,
    "sacmaesGWOGroup": sacmaesGWOGroup,
    "bigsacmaesGroup": bigsacmaesGroup,
    "bigsacmaesGWOGroup": bigsacmaesGWOGroup,
    "sacmaesGroup": sacmaesGroup,
    "bigsamadsGroup": bigsamadsGroup,
    "bigsamadsGWOGroup": bigsamadsGWOGroup,
    "cmaesGWOGroup": cmaesGWOGroup,
    "cmaesGroup": cmaesGroup,
    "saMadsGWOGroup": saMadsGWOGroup,
    "saMadsGroup": saMadsGroup
}