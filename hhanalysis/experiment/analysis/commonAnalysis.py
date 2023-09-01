import pandas as pd
import concepts
import matplotlib.pyplot as mp
import seaborn as sns
import scipy.stats as stats
from common import *
from fca import *
import os

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
    return f"{os.path.dirname(os.path.abspath(__file__))}/../../../"+path

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
def joinRecordsAndMetrics(records,getMetricsAndId):
    _,indexColumn=getMetricsAndId
    
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
    if(metricsAggregation != {}):
        aggregation=groupedExperiments.agg(metricsAggregation)
    else:
        aggregation=recordsWithMetrics

    {
    # print(f"aggregation{aggregation.columns}")
    # print(aggregation)
    # print(aggregation[(aggregation['problemName']=="PROBLEM_ROSENBROCK") & (aggregation['trialStepCount']==100) & (aggregation['HH-SA-temp'] == 10000) & (aggregation['HH-SA-alpha'] == 50)])
    }
    return (experimentColumns,aggregation)

def mergeOn(recordsAndMetrics,aggregations,joinColumns):
    return pd.merge(aggregations, recordsAndMetrics, on=joinColumns,how='left').dropna(axis='columns')

def checkAggregationCorrectness(aggregations):
    rows_with_nan = aggregations[aggregations.isna().any(axis=1)]
    if rows_with_nan.shape[0]:
        print("Incorrect experiments:")
        print(rows_with_nan)

def createTestGroupView(recordsPath,getMetricsAndId,mapRecordToExperiment,explanatoryColumns,responseColumns,metricsAggregation,enrichAndFilter,enrichWithMetrics=True):
    _,indexColumn=getMetricsAndId
    # records -> (experimentId, experimentColumns)
    # recordsPath,experimentId,(record->experiment) -> experimentDF(index: hashSHA256, columns:*experiments( baselevelIterations',... 'problemPath) )
    records=loadRecordsFrom(recordsPath,indexColumn,mapRecordToExperiment)
    { 
    # print(f"Records{records.columns}")
    # print(records)
    # print(records[(records['problemName']=="PROBLEM_ROSENBROCK") & (records['trialStepCount']==100) & (records['HH-SA-temp'] == 10000) & (records['HH-SA-alpha'] == 50)])
    }
    if enrichWithMetrics:
        recordsWithMetrics=joinRecordsAndMetrics(records,getMetricsAndId)
    else:
        recordsWithMetrics=records
    experimentColumns,aggregations=groupByExperimentsAndAggregate(recordsWithMetrics,explanatoryColumns,responseColumns,metricsAggregation)
    checkAggregationCorrectness(aggregations)
    testGroupDF=enrichAndFilter(recordsWithMetrics,aggregations,experimentColumns)
    return testGroupDF

def dropIrrelevantColumns(testGroupDF,controlGroupColumns):
    columnDifference=set(testGroupDF.columns).difference(controlGroupColumns)
    testGroupDF=testGroupDF.drop(columnDifference,axis=1)
    assert set(testGroupDF.columns.to_list()) == controlGroupColumns
    return testGroupDF

def createComparisonDF(controlDF, testDF, independetColumns):
    return pd.merge(controlDF, testDF, on=independetColumns, how="inner")

def printMinResultEachRow(df,experimentCols,columns):
    print("Win statistics:")
    minStatistic={}
    for column in columns:
        minStatistic[column]=0.0
    for  index,row in df.iterrows():
        minResult=min([row[column] for column in columns])
        methodsHavingMinResult=[column for column in columns if row[column] == minResult]
        for minMehtod in methodsHavingMinResult:
            minStatistic[minMehtod]=minStatistic[minMehtod]+1
        # print(f"{[row[column] for column in experimentCols]}{methodsHavingMinResult}->{minResult}")
    for optimizer,wins in sorted(minStatistic.items(), key=lambda x: -x[1]):
        print(f" {wins/df.shape[0]:.3f} - {optimizer}")

def printLatexMinAvgStd(df, optimizers):
    std_suffix = "-std"
    print("\\begin{table}[b]")
    print("\\resizebox{\columnwidth}{!}{")
    print("\\begin{tabular}{lc" + "".join(["l"]*len(optimizers)) + "}")
    print("\\hline")
    print("problem  & dimension & " + " & ".join([f"{opt}" for opt in optimizers]) + " \\\\")
    print("\\hline")
    current_problem = None
    for index, row in df.iterrows():
        if row["problemName"] != current_problem:
            if current_problem is not None:
                print("\\hline") 
            print(f"\\multirow{{{len(df[df['problemName'] == row['problemName']])}}}{{*}}{{{row['problemName'].replace('PROBLEM_','').capitalize()}}}")
            current_problem = row["problemName"]
        # find the minimal value among optimizers
        min_value = min(row[opt]+row[opt+std_suffix] for opt in optimizers)
        # create a list of formatted values with bold for the minimal one
        values_str_list = []
        for opt in optimizers:
            value_str = f"{row[opt]:.2e} $\\pm$ {row[opt + std_suffix]:.2e}"
            if row[opt]+row[opt+std_suffix] == min_value:
                value_str = f"\\textbf{{{value_str}}}" # add bold command
            values_str_list.append(value_str)
        # join the list with &
        values_str = " & ".join(values_str_list)
        print(f"&{int(row['modelSize'])}& {values_str} \\\\")
        
    print("\\hline")
    print("\\end{tabular}}")
    print("\\end{table}")

def printLatexMinEachRow(df, optimizers):
    print("\\begin{table}[b]")
    print("\\resizebox{\columnwidth}{!}{")
    print("\\begin{tabular}{lc" + "".join(["l"]*len(optimizers)) + "}")
    print("\\hline")
    print("problem  & dimension & " + " & ".join([f"{opt}" for opt in optimizers]) + " \\\\")
    print("\\hline")
    current_problem = None
    for index, row in df.iterrows():
        if row["problemName"] != current_problem:
            if current_problem is not None:
                print("\\hline") 
            print(f"\\multirow{{{len(df[df['problemName'] == row['problemName']])}}}{{*}}{{{row['problemName'].replace('PROBLEM_','').capitalize()}}}")
            current_problem = row["problemName"]
        # find the minimal value among optimizers
        min_value = min(row[opt]+row[opt] for opt in optimizers)
        # create a list of formatted values with bold for the minimal one
        values_str_list = []
        for opt in optimizers:
            value_str = f"{row[opt]:.2e}"
            if row[opt]+row[opt] == min_value:
                value_str = f"\\textbf{{{value_str}}}" # add bold command
            values_str_list.append(value_str)
        # join the list with &
        values_str = " & ".join(values_str_list)
        print(f"&{int(row['modelSize'])}& {values_str} \\\\")
        
    print("\\hline")
    print("\\end{tabular}}")
    print("\\end{table}")


def printMinAvgStdHighlighWilcoxRanksums(df, optimizers):
    std_suffix = "-std"
    print("\\begin{table}[b]")
    print("\\resizebox{\columnwidth}{!}{")
    print("\\begin{tabular}{lc" + "".join(["l"]*len(optimizers)) + "}")
    print("\\hline")
    print("problem  & dimension & " + " & ".join([f"{opt}" for opt in optimizers]) + " \\\\")
    print("\\hline")
    
    current_problem = None
    for index, row in df.iterrows():
        wilcoxRanksum = pickle.loads(json.loads(row['wilcoxRanksums']).encode('latin-1'))
        wilcoxRanksumIndexOrder = json.loads(row['wilcoxRanksumsIndexOrder'])
        bestOptimizers=set(optimizers)
        for i in range(len(wilcoxRanksumIndexOrder)):
            for j in range(len(wilcoxRanksumIndexOrder)):
                if wilcoxRanksum[i][j]==0:
                    if wilcoxRanksumIndexOrder[j] in bestOptimizers:
                        bestOptimizers.remove(wilcoxRanksumIndexOrder[j])
        
        if row["problemName"] != current_problem:
            if current_problem is not None:
                print("\\hline") 
            print(f"\\multirow{{{len(df[df['problemName'] == row['problemName']])}}}{{*}}{{{row['problemName'].replace('PROBLEM_','').capitalize()}}}")
            current_problem = row["problemName"]
        # find the minimal value among optimizers
        min_value = min(row[opt]+row[opt+std_suffix] for opt in optimizers)
        # create a list of formatted values with bold for the minimal one
        values_str_list = []
        for opt in optimizers:
            value_str = f"{row[opt]:.2e} $\\pm$ {row[opt + std_suffix]:.2e}"
            if opt in bestOptimizers:
                value_str = f"\\textbf{{{value_str}}}" # add bold command
            values_str_list.append(value_str)
        # join the list with &
        values_str = " & ".join(values_str_list)
        print(f"&{int(row['modelSize'])}& {values_str} \\\\")
        
    print("\\hline")
    print("\\end{tabular}}")
    print("\\end{table}")

def printStatisticsOfWilcoxRanksums(df, optimizers):
    print("Wilcoxon win statistics")
    bestStatistics={}
    for opt in optimizers:
        bestStatistics[opt]=0.0
    for index, row in df.iterrows():
        wilcoxRanksum = pickle.loads(json.loads(row['wilcoxRanksums']).encode('latin-1'))
        wilcoxRanksumIndexOrder = json.loads(row['wilcoxRanksumsIndexOrder'])
        bestOptimizers=set(optimizers)
        for i in range(len(wilcoxRanksumIndexOrder)):
            for j in range(len(wilcoxRanksumIndexOrder)):
                if wilcoxRanksum[i][j]==0:
                    if wilcoxRanksumIndexOrder[j] in bestOptimizers:
                        bestOptimizers.remove(wilcoxRanksumIndexOrder[j])
        for opt in bestOptimizers:
            bestStatistics[opt]=bestStatistics[opt]+1
    for optimizer,wins in sorted(bestStatistics.items(), key=lambda x: -x[1]):
        print(f" {wins/df.shape[0]:.3f} - {optimizer}")

def calculateWilcoxRanksumStatisticsForEachDimension(df, optimizers):
    statisticsforDimension={}
    grouped=df.groupby(['modelSize'])
    for (modelSize,groupIndex) in grouped:
        bestStatistics={}
        for opt in optimizers:
            bestStatistics[opt]=0.0
        for index, row in groupIndex.iterrows():
            wilcoxRanksum = pickle.loads(json.loads(row['wilcoxRanksums']).encode('latin-1'))
            wilcoxRanksumIndexOrder = json.loads(row['wilcoxRanksumsIndexOrder'])
            bestOptimizers=set(optimizers)
            for i in range(len(wilcoxRanksumIndexOrder)):
                for j in range(len(wilcoxRanksumIndexOrder)):
                    if wilcoxRanksum[i][j]==0:
                        if wilcoxRanksumIndexOrder[j] in bestOptimizers:
                            bestOptimizers.remove(wilcoxRanksumIndexOrder[j])
            for opt in bestOptimizers:
                bestStatistics[opt]=bestStatistics[opt]+1

        for optimizer,wins in sorted(bestStatistics.items(), key=lambda x: -x[1]):
            bestStatistics[optimizer]=wins/groupIndex.shape[0]
        statisticsforDimension[modelSize]=bestStatistics
    return statisticsforDimension

def printStatisticsOfWilcoxRanksumsForEachDimension(statisticsforDimension):
    for modelSize,statistics in statisticsforDimension.items():
        for optimizer,wins in sorted(statistics.items(), key=lambda x: -x[1]):
            print(f"Dim {modelSize} - {wins:.3f} - {optimizer}")

def wilcoxRanksum(samples):
    sampleCount=len(samples)
    comparisonMatrix=np.zeros((sampleCount,sampleCount))
    for i in range(sampleCount):
        for j in range(sampleCount):
            (statistic,pvalue)=sp.stats.ranksums(samples[i], samples[j], alternative='less')
            # (statistic,pvalue)=sp.stats.ranksums(samples[i], samples[j])
            comparisonMatrix[i][j]=1-(pvalue<0.05)
    return comparisonMatrix

def wilcoxRanksumForRow(sampleColumns,row):
    samples=[row[sampleColumn] for sampleColumn in sampleColumns]
    return json.dumps(pickle.dumps(wilcoxRanksum(samples)).decode('latin-1'))

def addWilcoxRankSumResultToEachRow(df,experimentCols,sampleColumns):
    # print(f"Experiment order: {sampleColumns}")
    matrices=[]
    df['wilcoxRanksums'] = df.apply(partial(wilcoxRanksumForRow,sampleColumns), axis=1)
    df['wilcoxRanksumsIndexOrder'] = json.dumps(list(map(lambda column:column.replace('-samples',''), sampleColumns)))
    return matrices