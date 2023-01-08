import pandas as pd
import concepts
import matplotlib.pyplot as mp
import seaborn as sns
import scipy.stats as stats
from common import *
from loadCustomHYS import *
from fca import *

# recordsPath -> experimentDF(index: hashSHA256, columns:*experiments( baselevelIterations',
#                                                         'populationSize', 
#                                                         'modelSize',
#                                                         'trialSampleSizes', 
#                                                         'trialStepCount', 
#                                                         'HH-SA-temp', 
#                                                         'HH-SA-alpha', 
#                                                         'problemName', 
#                                                         'problemPath) )
def loadRecordsFrom(path,indexColumn,mapRecordToExperiment):
    recordsDF=pd.DataFrame(loadJsonFrom(path)["experiments"]).transpose()
    recordsDF[indexColumn]=recordsDF.index
    experimentsDF=pd.concat(recordsDF.apply(mapRecordToExperiment,axis=1).map(lambda ex:pd.DataFrame(ex,index=[0])).to_list())
    experimentsDF[indexColumn]=recordsDF.index
    return experimentsDF.set_index(indexColumn)

def convertMetricsPathToThisDir(path):
    return "../../../"+path

# experimentDF(index: hashSHA256, columns: *experiments(baselevelIterations', ... 'problemName','problemPath()))
# -> 
# metrics(index: hashSHA256, minMedIQR)
def enrichRecordsWithMetrics(records,getMetricsAndId):
    mapMetrics,metricId=getMetricsAndId
    metricsPaths=records['problemPath'].unique()
    metricsDataframe=pd.concat(list(map(lambda path: pd.DataFrame(loadJsonFrom(convertMetricsPathToThisDir(path))),metricsPaths)))
    metricsDataframe=pd.concat(metricsDataframe["experiments"].map(mapMetrics).map(lambda metric:pd.DataFrame(metric,index=[0])).to_list())
    metricsDataframe=metricsDataframe.set_index(metricId)
    return metricsDataframe

# A list of columns that are used to group the different experiments
# Each group contains a set of experiments where the explanatoryVars and responseVars vary i.e the experiment results.

# experimentDF(index: hashSHA256, columns: *experiments(baselevelIterations', ... 'problemName','problemPath()),
#                                          *metrics(minMedIQR)),
# explanatoryVars, responseVars
# -> 
# (*experiments U *metrics) - (explanatoryVars U responseVars)
def experimentDifferentiatingColumns(recordsAndMetricsDF,explanatoryVars,responseVars):
    allColumns=set(recordsAndMetricsDF.columns.to_list())
    experimentColumns=allColumns.difference(explanatoryVars.union(responseVars))
    return list(experimentColumns)

# Groups records according to different experiments(experimentDifferentiatingColumns) 
# the varying columns in the groups are the explanatory and response variables
def groupByExperiments(recordsAndMetricsDF,experimentGroupColumns):
    return recordsAndMetricsDF.groupby(experimentGroupColumns,as_index=False)

# Loads the records from recordsPath and fetches all the relevant metrics of each record
# recordsPath -> (experimentId, experimentColumns,metricsColumns)
def joinRecordsAndMetrics(recordsPath,getMetricsAndId,mapRecordToExperiment):
    _,indexColumn=getMetricsAndId

    # records -> (experimentId, experimentColumns)
    # recordsPath,experimentId,(record->experiment) -> experimentDF(index: hashSHA256, columns:*experiments( baselevelIterations',... 'problemPath) )
    records=loadRecordsFrom(recordsPath,indexColumn,mapRecordToExperiment)
    { 
    # print(f"Records{records.columns}")
    # print(records)
    # print(records[(records['problemName']=="PROBLEM_ROSENBROCK") & (records['trialStepCount']==100) & (records['HH-SA-temp'] == 10000) & (records['HH-SA-alpha'] == 50)])
    }
    
    # (experimentId, experimentColumns), (allExperimentMetrics -> relevantMetrics,experimentId) 
    # -> (experimentId, metricsColumns)
    #
    # experimentDF(index: hashSHA256, columns:*experiments( baselevelIterations',... 'problemPath) )
    # ->   metrics(index: hashSHA256, minMedIQR)
    enriched=enrichRecordsWithMetrics(records,getMetricsAndId)
    {
    # print(f"enrichedRecordsWithMetrics{enriched.columns}")
    # print(f"enrichedRecordsWithMetrics{enriched.index}")
    }
    
    # (experimentId, experimentColumns) -> (experimentId, experimentColumns+metricsColumns)
    # experimentDF(index: hashSHA256, columns: *experiments(baselevelIterations', ... 'problemName','problemPath()),
    #                                          *metrics(minMedIQR)),
    recordsWithMetrics=records.join(enriched)
    {
    # print(f"recordsWithMetrics{recordsWithMetrics.columns}")
    # print(recordsWithMetrics)
    # print(recordsWithMetrics[(recordsWithMetrics['problemName']=="PROBLEM_ROSENBROCK") & (recordsWithMetrics['trialStepCount']==100) & (recordsWithMetrics['HH-SA-temp'] == 10000) & (recordsWithMetrics['HH-SA-alpha'] == 50)])
    }
    return recordsWithMetrics

# (experimentId, experimentColumns,metricsColumns)
# explanatoryColumns, responseColumns
# -> 
#  (
#   Experiments(*experimentColumns U *metricsColumns) - (explanatoryVars U responseVars)
#   +
#   AggregationsOf(explanatoryVars U responseVars)
def groupByExperimentsAndAggregate(recordsWithMetrics,explanatoryColumns,responseColumns,metricsAggregation):
    experimentColumns=experimentDifferentiatingColumns(
                        recordsAndMetricsDF=recordsWithMetrics,
                        explanatoryVars=explanatoryColumns,
                        responseVars=responseColumns)
    groupedExperiments=groupByExperiments(recordsAndMetricsDF=recordsWithMetrics,experimentGroupColumns= experimentColumns)
    aggregation=groupedExperiments.agg(metricsAggregation)
    {
    # print(f"aggregation{aggregation.columns}")
    # print(aggregation)
    # print(aggregation[(aggregation['problemName']=="PROBLEM_ROSENBROCK") & (aggregation['trialStepCount']==100) & (aggregation['HH-SA-temp'] == 10000) & (aggregation['HH-SA-alpha'] == 50)])
    }
    return (experimentColumns,aggregation)

def enrichAggregations(recordsAndMetrics,aggregations,joinColumns):
    return pd.merge(aggregations, recordsAndMetrics, on=joinColumns,how='left').dropna(axis='columns')


def createTestGroupView(recordsPath,getMetricsAndId,mapRecordToExperiment,explanatoryColumns,responseColumns,metricsAggregation,enrichAndFilter):
    recordsWithMetrics=joinRecordsAndMetrics(recordsPath,getMetricsAndId,mapRecordToExperiment)
    experimentColumns,aggregations=groupByExperimentsAndAggregate(recordsWithMetrics,explanatoryColumns,responseColumns,metricsAggregation)
    testGroupDF=enrichAndFilter(recordsWithMetrics,aggregations,experimentColumns)
    return testGroupDF

def dropIrrelevantColumns(testGroupDF,controlGroupColumns):
    columnDifference=set(testGroupDF.columns).difference(controlGroupColumns)
    testGroupDF=testGroupDF.drop(columnDifference,axis=1)
    assert set(testGroupDF.columns.to_list()) == controlGroupColumns
    return testGroupDF

def createComparisonDF(controlDF, testDF, independetColumns):
    return pd.merge(controlDF, testDF, on=independetColumns, how="inner")
