import json
import json5
from dict_hash import sha256
import itertools
import scipy as sp
import numpy as np
from functools import partial
import pickle

def loadJsonFrom(path,ignoreTrailingCommas=False):
    f = open(path)
    if ignoreTrailingCommas:
        return json5.load(f)
    else:
        return json.load(f)

def stringifyList(list):
    return [str(item) for item in list]


def zipWithProperty(list,property):
    print([property]*len(list))
    return zip([property]*len(list),list)

def mapExperimentListToDict(experiment):
    paramsDict={}
    for param in experiment:
        if param[1] != None:
            paramsDict[param[0]]=param[1]
    return json.loads(json.dumps(paramsDict))

def hashOfExperiment(experiment):
    return sha256(experiment)

def possibleExperimentIds(experimentParams):
    ids=[]
    variations=list(itertools.product(*list(experimentParams.values())))
    for experiment in variations:
        experimentDict=mapExperimentListToDict(experiment=experiment)
        ids.append(hashOfExperiment(experimentDict))
    return ids

def matchOneIdInIndex(index,ids):
    mathcingRandomIs=list(filter(lambda id: id in index,ids))
    assert len(mathcingRandomIs) == 1
    return mathcingRandomIs[0]

def printMinResultEachRow(df,experimentCols,columns):
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

def plotDataForWilcoxRanksumsComparisonPlot(statisticsforDimension,optimizers):
    categories=[]
    subcategories=optimizers
    values=[]
    for modelSize,statistics in statisticsforDimension.items():
        categories.append(str(modelSize))
        theseValues=[]
        for optimizer in optimizers:
            theseValues.append(statisticsforDimension[modelSize][optimizer])
        values.append(theseValues)
    return (categories,subcategories,values)

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

        