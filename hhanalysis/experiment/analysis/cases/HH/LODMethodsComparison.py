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
from loadHHData import *
from  commonHHAnalysis import *

def methodsComparison(all,metadata,optimizersToCompare=None):
    metadata["baselevelIterations"]=all['baselevelIterations'].iloc[0]
    # all=all[['problemName','modelSize','hyperLevel-id',metadata["minMetricColumn"]]]
    all=all.groupby(['problemName','modelSize'])
    transpose=pd.DataFrame()
    optimizersSet=set()
    for (group,groupIndex) in all:
        transposedRow={}
        transposedRow['problemName']=group[0]
        transposedRow['modelSize']=group[1]
        for index,row in groupIndex.iterrows():
            transposedRow[row["hyperLevel-id"]]=row[metadata["minMetricColumn"]]
            transposedRow[f'{row["hyperLevel-id"]}-std']=row['minStd']
            transposedRow[f'{row["hyperLevel-id"]}-samples']=json.loads(row["samples"])
            if not row["hyperLevel-id"] in optimizersSet:
                metadata['optimizers'].append(row["hyperLevel-id"])
                optimizersSet.add(row["hyperLevel-id"])
        transpose=transpose.append(transposedRow,ignore_index=True)
        
    # if(optimizersToCompare!=None):
    #     printMinResultEachRow(transpose,['problemName','modelSize'],optimizersToCompare)
    # else:
    #     printMinResultEachRow(transpose,['problemName','modelSize'],optimizersSet)

    addWilcoxRankSumResultToEachRow(transpose,['problemName','modelSize'],[f'{column}-samples' for column in metadata['optimizers']])
    statisticsforDimension=calculateWilcoxRanksumStatisticsForEachDimension(transpose,metadata['optimizers'])
    printStatisticsOfWilcoxRanksums(transpose,metadata['optimizers'])

    plotWilcoxRanksums(transpose,6,len(metadata["modelSize"]),
                       list(map(lambda name:name.replace('-BIG',''),metadata['optimizers'])),
                       filename=f"{ROOT}/plots/WILCOX_{[metadata[savecol] for savecol in metadata['saveMetadataColumns']]}.svg",
                    #    filename=None,
                       figsize=(8,8),blockPlot=True)
    printMinAvgStdHighlighWilcoxRanksums(transpose,metadata['optimizers'])
    # tabloo.show(transpose)
def loadData(metadata,methods):
    hhdata=loadDataMap()
    # c=hhdata['customhys2'](metadata)
    return pd.concat([hhdata[method](metadata) for method in methods])

def compare(methods,problems,dimensions):
    metadata={
            "minMetricColumn":'minAvg',
            "metricsAggregation":{'minAvg':'min'},
            "mergeOn":mergeOnAvg,
            # "minMetricColumn":'minMedIQR',
            # "metricsAggregation":{'minMedIQR':'min'},
            # "mergeOn":mergeOnMinMedIQR,
            'optimizers':list(),
            "saveMetadataColumns":["minMetricColumn",'optimizers','baselevelIterations'],
            "baselevelIterations": [100],
            "problems":problems,
            # "modelSize":[1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750] 
            "modelSize":dimensions,
            "datasets":{}
            # "modelSize":[1,2,3,4,5,6,7,8,9] 
            # "modelSize":[10,15] 
        }
    all=loadData(metadata,methods)
    all=dropIrrelevantColumns(all,set(['modelSize','problemName','hyperLevel-id','baselevelIterations','minAvg','minStd','minMedIQR','samples','elapsedTimeSec']))
    all=all[selectAllMatchAtLeastOne(all,[('baselevelIterations',metadata["baselevelIterations"]),('modelSize',metadata["modelSize"]),('problemName',metadata["problems"])])]
    # all=all.sort_values(by=['modelSize',"problemName",metadata["minMetricColumn"]])
    # all=all[['problemName','modelSize','hyperLevel-id',metadata["minMetricColumn"],'minStd','samples']]

    print(f"Problems: {problems} \n Dimensions: {dimensions}")
    # methodsComparison(all,metadata,set(['SA-PERTURBMulti','PSO','CRO','SMA']))
    methodsComparison(all,metadata)

def compareElapsedTimes(methods,problems,dimensions,optimizersToCompare=None):
    metadata={
            "minMetricColumn":'elapsedTimeSec',
            "metricsAggregation":{'elapsedTimeSec':'min'},
            "mergeOn":mergeOnElapsedTime,
            'optimizers':list(),
            "saveMetadataColumns":["minMetricColumn",'optimizers','baselevelIterations'],
            "baselevelIterations": [100],
            "problems":problems,
            "modelSize":dimensions,
            "datasets":{}
        }
    all=loadData(metadata,methods)
    all=dropIrrelevantColumns(all,set(['baselevelIterations','modelSize','problemName','hyperLevel-id','elapsedTimeSec']))
    all=all[selectAllMatchAtLeastOne(all,[('baselevelIterations',metadata["baselevelIterations"]),('modelSize',metadata["modelSize"]),('problemName',metadata["problems"])])]
    all=all.groupby(['problemName','modelSize'])
    transpose=pd.DataFrame()
    optimizersSet=set()
    for (group,groupIndex) in all:
        transposedRow={}
        transposedRow['problemName']=group[0]
        transposedRow['modelSize']=group[1]
        for index,row in groupIndex.iterrows():
            transposedRow[row["hyperLevel-id"]]=row[metadata["minMetricColumn"]]
            if not row["hyperLevel-id"] in optimizersSet:
                metadata['optimizers'].append(row["hyperLevel-id"])
                optimizersSet.add(row["hyperLevel-id"])
        transpose=transpose.append(transposedRow,ignore_index=True)
    if optimizersToCompare !=None:
        printLatexMinEachRow(transpose,optimizers=optimizersToCompare)
    else:
        printLatexMinEachRow(transpose,optimizers=optimizersSet)
    tabloo.show(transpose)

# allmethods=['saperturbGWOGroup','customhys2','randomgaGroup','randomdeGroup','mealpyCRO']
allmethods=['mealpyCRO','saperturbGWOGroup']
# allmethods=['saperturbGWOGroup','customhys2']
# allmethods=['saperturbGWOGroup','mealpyCRO']
# allmethods=['saperturbGWOGroup','randomgaGroup']
# allmethods=['saperturbGWOGroup','randomdeGroup']
# allmethods=['saperturbMultiOperatorGroup','randomgaGroup','randomdeGroup']
# allmethods=['saperturbMultiOperatorGroup','saperturbGWOGroup','randomgaGroup','randomdeGroup']
allproblems=['PROBLEM_SCHWEFEL223','PROBLEM_STYBLINSKITANG','PROBLEM_TRID','PROBLEM_RASTRIGIN','PROBLEM_QING','PROBLEM_ROSENBROCK']
alldimensions=[5,50,100]

# compare(allmethods,allproblems,alldimensions)
compareElapsedTimes(allmethods,allproblems,alldimensions,['SA-PERTURB','CRO'])

# compare(allmethods,allproblems,lowerdimensions)
# compare(allmethods,allproblems,highdimensions)

# compare(allmethods,convexUnimodal,lowerdimensions)
# compare(allmethods,convexUnimodal,highdimensions)

# compare(allmethods,nonconvexMultimodal,lowerdimensions)
# compare(allmethods,nonconvexMultimodal,highdimensions)

# compare(allmethods,separable,lowerdimensions)
# compare(allmethods,separable,highdimensions)

# compare(allmethods,nonSeparable,lowerdimensions)
# compare(allmethods,nonSeparable,highdimensions)