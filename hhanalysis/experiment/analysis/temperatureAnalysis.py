import pandas as pd 
import concepts
import matplotlib.pyplot as mp
import seaborn as sns
import scipy.stats as stats
import json

EXPERIMENT_RECORDS_PATH="../../logs/records.json"

def loadJsonFrom(path):
    f = open(path)
    return json.load(f)

def experimentToDFExperiment(experiment):
    experiment["problemName"]=experiment["problems"][0]
    experiment["problemPath"]=experiment["problems"][1]
    experiment.pop("problems")
    return experiment

def loadRecordsFrom(path):
    recordsDF=pd.DataFrame(loadJsonFrom(path=path)["experiments"]).transpose()
    experimentsSeriesDF=pd.DataFrame(recordsDF["experiment"])
    experimentsSeriesDF["hashSHA256"]=experimentsSeriesDF.index
    experimentsDF=pd.concat(experimentsSeriesDF["experiment"].map(experimentToDFExperiment).map(lambda ex:pd.DataFrame(ex,index=[0])).to_list())
    experimentsDF["hashSHA256"]=experimentsSeriesDF["hashSHA256"].index
    experimentsDF["hashSHA256"]=experimentsSeriesDF["hashSHA256"].index
    return experimentsDF.set_index("hashSHA256")

def convertMetricsPathToThisDir(path):
    return "../../../"+path

def filterMetricProperties(metric):
    return {
        "minMedIQR":metric["minBaseLevelStatistic"],
        "hashSHA256":metric["experimentHashSha256"]
    }
def enrichRecordsWithMetrics(records):
    metricsPaths=records['problemPath'].unique()
    metricsDataframe=pd.concat(list(map(lambda path: pd.DataFrame(loadJsonFrom(convertMetricsPathToThisDir(path))),metricsPaths)))
    metricsDataframe=pd.concat(metricsDataframe["experiments"].map(filterMetricProperties).map(lambda metric:pd.DataFrame(metric,index=[0])).to_list())
    metricsDataframe=metricsDataframe.set_index("hashSHA256")
    return metricsDataframe

def experimentsGroupByColumns(recordsAndMetricsDF,explanatoryVars,responseVars):
    allColumns=set(recordsAndMetricsDF.columns.to_list())
    experimentColumns=allColumns.difference(explanatoryVars.union(responseVars))
    return list(experimentColumns)

def groupByExperiments(recordsAndMetricsDF,experimentGroupColumns):
    return recordsAndMetricsDF.groupby(experimentGroupColumns,as_index=False)

records=loadRecordsFrom(EXPERIMENT_RECORDS_PATH)
recordsWithMetrics=records.join(enrichRecordsWithMetrics(records))
experimentColumns=experimentsGroupByColumns(recordsAndMetricsDF=recordsWithMetrics,
                    explanatoryVars=set(["HH-SA-temp","HH-SA-alpha"]),
                    responseVars=set(["minMedIQR"]))
groupedExperiments=groupByExperiments(recordsAndMetricsDF=recordsWithMetrics,experimentGroupColumns= experimentColumns)
minMedIqrView=groupedExperiments.agg({'minMedIQR':'min'})
expandedMinMedIqrView=pd.merge(minMedIqrView,recordsWithMetrics,on=experimentColumns+["minMedIQR"],how='left').dropna(axis='columns')
expandedMinMedIqrView=expandedMinMedIqrView.drop(["problemPath"],axis=1)
print(expandedMinMedIqrView.to_html())
# print(minMedIqrView.columns)

# print(expandedMinMedIqrView)
# print(expandedMinMedIqrView.columns)
