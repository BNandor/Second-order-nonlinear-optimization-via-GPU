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

def customhys(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-CUSTOMHyS'
    if dataId not in metadata['datasets']:
            metadata['datasets'][dataId]=getCustomHySControlGroupDF()
            metadata['datasets'][dataId]['hyperLevel-id']=dataId
    return metadata['datasets'][dataId]

def customhys2(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-CUSTOMHyS'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId]=createTestGroupView(f"{CUSTOMHYS2_RESULTS_PATH}{experimentPath}/records.json",
                                    (None,"hashSHA256"),
                                    customhys2RecordToExperiment,
                                    set(),
                                    set(["minMedIQR","minAvg","minStd","samples"]),
                                    metadata['metricsAggregation'],
                                    metadata['mergeOn'],enrichWithMetrics=False)
        metadata['datasets'][dataId]['hyperLevel-id']=dataId
    return metadata['datasets'][dataId]

def mealpy(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-MEALPY'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId]=createTestGroupView(f"{MEALPY_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
                                    (None,"hashSHA256"),
                                    mealpyRecordToExperiment,
                                    set(),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    enrichAndFilterMealpy,enrichWithMetrics=False)
        # metadata['datasets'][dataId]['hyperLevel-id']=dataId
    return metadata['datasets'][dataId]

def mealpyCRO(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-MEALPY'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId]=createTestGroupView(f"{MEALPY_CRO_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
                                    (None,"hashSHA256"),
                                    mealpyExp.loadMealpy.recordToExperiment,
                                    set(),
                                    set(["minMedIQR","minAvg","minStd","samples"]),
                                    metadata['metricsAggregation'],
                                    metadata['mergeOn'],enrichWithMetrics=False)
        metadata['datasets'][dataId]=pd.DataFrame(metadata['datasets'][dataId][metadata['datasets'][dataId]['hyperLevel-id'] =='CRO'])
        # metadata['datasets']["mealpy"]['hyperLevel-id']=dataId
    return metadata['datasets'][dataId]


def nmhh(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-NMHH'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId]=createTestGroupView(f"{SA_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
                                    (filterMetricPropertiesMinMedIQR,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    enrichAndFilterSA)
        metadata['datasets'][dataId]['hyperLevel-id']=dataId
    return metadata['datasets'][dataId]


def nmhh2(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-NMHH'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId]=createTestGroupView(f"{SA_GA_DE_GD_LBFGS_RECORDS_PATH}{experimentPath}/records.json",
                                    (filterMetricPropertiesAverageAndMedIQR,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR","minAvg","minStd","samples"]),
                                    metadata['metricsAggregation'],
                                    metadata['mergeOn']) 
        metadata['datasets'][dataId]['hyperLevel-id']=dataId
    return metadata['datasets'][dataId]

def sarefineGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-LBFGS'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{SAREFINE_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesMinMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR"]),
            {'minMedIQR': 'min'},
            enrichAndFilterSA)
        metadata['datasets'][dataId]['hyperLevel-id']=dataId
    return metadata['datasets'][dataId]

def lbfgsGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-LBFGS'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{LBFGS_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesMinMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR"]),
            {'minMedIQR': 'min'},
            justAggregations
        )
        metadata['datasets'][dataId]['hyperLevel-id']=dataId
    return metadata['datasets'][dataId]


def gdGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-GD'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{GD_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesMinMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR"]),
            {'minMedIQR': 'min'},
            justAggregations)
        metadata['datasets'][dataId]['hyperLevel-id']=dataId
    return metadata['datasets'][dataId]

def saperturbGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-SA-PERTURB'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{SAPERTURB_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id']=dataId
    return metadata['datasets'][dataId]

def saperturbGWOGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-SA-PERTURB'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{SAPERTURBGWO_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id']=dataId
    return metadata['datasets'][dataId]

def saperturbMultiOperatorGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-SA-PERTURBMultiOperator'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{SAPERTURBMULTIOPERATORS_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id']=dataId
    return metadata['datasets'][dataId]

def gaGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-GA'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{GA_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id']=dataId
    return metadata['datasets'][dataId]
def deGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-DE'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{DE_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]

def randomgaGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-RANDOM-GA'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{RANDOM_GA_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]

def randomdeGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-RANDOM-DE'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{RANDOM_DE_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]

def saGWOGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-SA-GWO'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{RANDOM_SA_GWO_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn'],
            True
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]

def madsGWOGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-MADS-GWO'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]

def madsGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-MADS'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{MADS_NMHH_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]

def sacmaesGWOGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-SA-CMA-ES-GWO'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{SA_CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]

# Repeat the same pattern for other functions...

def saMadsGWOGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-SA-MADS-GWO'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{SA_MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]

def saMadsGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-SA-MADS'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{SA_MADS_NMHH_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]

def cmaesGWOGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-CMA-ES-GWO'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]

def cmaesGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-CMA-ES'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{CMA_ES_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]

def bigsamadsGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-BIGSA-MADS'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{BIGSA_MADS_NMHH_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]

def bigsamadsGWOGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-BIGSA-MADS-GWO'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{BIGSA_MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]

def bigsacmaesGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-BIGSA-CMA-ES'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{BIGSA_CMA_ES_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]


def bigsacmaesGWOGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-BIGSA-CMA-ES-GWO'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{BIGSA_CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]


def sacmaesGroup(metadata,experimentPath='/'):
    dataId=f'{experimentPath}-SA-CMA-ES'
    if dataId not in metadata['datasets']:
        metadata['datasets'][dataId] = createTestGroupView(f"{SA_CMA_ES_EXPERIMENT_RECORDS_PATH}{experimentPath}/records.json",
            (filterMetricPropertiesAverageAndMedIQR, "hashSHA256"),
            recordToExperiment,
            set(),
            set(["minMedIQR", "minAvg", "minStd", "samples"]),
            metadata['metricsAggregation'],
            metadata['mergeOn']
        )
        metadata['datasets'][dataId]['hyperLevel-id'] = dataId
    return metadata['datasets'][dataId]

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