from commonAnalysis import *
from dict_hash import sha256
from commonPlots import *

def enrichAndFilterSA(recordsWithMetrics,aggregations,experimentColumns):
    HHView=mergeOn(recordsWithMetrics,aggregations,experimentColumns+["minMedIQR"])
    return HHView[ (HHView['trialStepCount']==100) & (HHView['HH-SA-temp'] == 10000) & (HHView['HH-SA-alpha'] == 50)]

def enrichAndFilterSATemp(recordsWithMetrics,aggregations,experimentColumns):
    HHView=mergeOn(recordsWithMetrics,aggregations,experimentColumns+["minMedIQR"])
    return HHView[ (HHView['problemName']=="PROBLEM_ROSENBROCK") ]

def justAggregations(recordsWithMetrics,aggregations,experimentColumns):
    return aggregations


# record(experiment(problems(name,path),..),metadata) -> experiment(problemName,problemPath,..)
def recordToExperiment(record):
    experiment=record['experiment']
    experiment["problemName"]=experiment["problems"][0]
    experiment["problemPath"]=experiment["problems"][1]
    experiment.pop("problems")
    return experiment

#  "problems": [
#                     "PROBLEM_ROSENBROCK",
#                     "hhanalysis/logs/scalabilityTests/rosenbrock.json"
#                 ],
#                 "baselevelIterations": 100,
#                 "populationSize": 30,
#                 "modelSize": 128,
#                 "trialSampleSizes": 30,
#                 "trialStepCount": 10,
#                 "HH-SA-temp": 10000,
#                 "HH-SA-alpha": 50,
#                 "hyperLevelMethod": "PERTURB"
#             },
#             "metadata": {
#                 "elapsedTimeSec": 840.7598524409987,
#                 "threads": 1
#             }
# record(experiment(problems(name,path),..),metadata) -> experiment(problemPath,modelSize...sec)
def recordToScalabilityExperiment(record):
    experiment={}
    experiment["problemPath"]=record['experiment']["problems"][1]
    experiment["modelSize"]=record['experiment']["modelSize"]
    experiment['hyperLevelMethod']=record['experiment']["hyperLevelMethod"]
    experiment["threads"]=record['metadata']['threads']
    experiment['sec']=record['metadata']["elapsedTimeSec"]
    return experiment

# Filter the metrics data to the relevant metrics

 # SA-alpha,SA-temp0,SA-temps[],
 # baseLevel-popSize,baseLevel-problemId,baseLevel-sampleSize,baseLevel-xDim,baseLevelEvals,
 # bestParameters{},
 # experimentHashSha256,
 # hyperLevel-id,
 # minBaseLevelStatistic,
 # trialCount,
 # trials[{atEval,med_+_iqr}]
 # -> 
 # {minMedIQR,hashSHA256}
def filterMetricPropertiesSA(metric):
    return {
        "minMedIQR":metric["minBaseLevelStatistic"],
        "hashSHA256":metric["experimentHashSha256"]
    }

def noMetric(metric):
    return {
        "hashSHA256":metric["experimentHashSha256"]
    }

def SAStepsMetric(metric):
    return {
        "minMedIQR":metric["minBaseLevelStatistic"],
        "steps":json.dumps(metric["trials"]),
        "hashSHA256":metric["experimentHashSha256"]
    }

def getControlGroupDF():
    return pd.DataFrame(stepsToResults(loadResultsSteps(CustomHYSPath)))
    
def getGroups(recordsPath):
    controlGroup=getControlGroupDF()
    testGroupDF=createTestGroupView(recordsPath,
                                    (filterMetricPropertiesSA,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    enrichAndFilterSA)
    testGroupDF=dropIrrelevantColumns(testGroupDF,set(controlGroup.columns))
    return (controlGroup,testGroupDF)

def getComparison(controlGroup,testGroupDF,independentColumns):
    comparisonDF=createComparisonDF(controlGroup, testGroupDF,independentColumns)
    comparisonDF['HHBetter']=comparisonDF['minMedIQR_x']>comparisonDF['minMedIQR_y']
    return comparisonDF

def createAndSaveContext(comparisonDF,independentColumns):
    ctx=createContext(comparisonDF,independentColumns+['HHBetter'])
    saveContext(ctx,'comparison.cxt')

SA_EXPERIMENT_RECORDS_PATH="../../logs/records.json"
def SAAnalysis():
    controlGroup,testGroupDF=getGroups(SA_EXPERIMENT_RECORDS_PATH)
    independentColumns=["baselevelIterations","modelSize","problemName"]
    comparisonDF=getComparison(controlGroup,testGroupDF,independentColumns)
    # print(comparisonDF)
    comparisonDF=comparisonDF.sort_values(by=["problemName",'modelSize','baselevelIterations'])
    createAndSaveContext(comparisonDF,independentColumns)
    comparisonDF=comparisonDF.drop(['HHBetter'],axis=1)
    print(comparisonDF.to_latex(index=False))
    
def SATempAnalysis():
    # (experimentColumns,minF,HH-SA-Temp,HH-SA-alpha)
    # {
    tempView=createTestGroupView(SA_EXPERIMENT_RECORDS_PATH,
                                    (filterMetricPropertiesSA,"hashSHA256"),
                                    recordToExperiment,
                                    set(['HH-SA-temp','HH-SA-alpha','trialStepCount']),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    enrichAndFilterSATemp)
    # }
    # tempView=createTestGroupView(SA_EXPERIMENT_RECORDS_PATH,
    #                                 (filterMetricPropertiesSA,"hashSHA256"),
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

SCALABILITY_EXPERIMENT_RECORDS_PATH="../../logs/scalabilityTests/records.json"
def calculateSpeedup(threadsSecDF):
    sequentialTime=threadsSecDF[threadsSecDF['threads']==1].iloc[0]['sec']
    threadsSecDF['seqTime']=sequentialTime
    threadsSecDF=threadsSecDF.rename(columns={'sec':'parTime'})
    threadsSecDF['speedup']=threadsSecDF['seqTime']/threadsSecDF['parTime']
    return threadsSecDF

def ScalabilityAnalysis():
    scalabilityView=createTestGroupView(
                                    recordsPath=SCALABILITY_EXPERIMENT_RECORDS_PATH,
                                    getMetricsAndId=(noMetric,"hashSHA256"),
                                    mapRecordToExperiment=recordToScalabilityExperiment,
                                    explanatoryColumns=set(),
                                    responseColumns=set(["sec"]),
                                    metricsAggregation={'sec':'mean'},
                                    enrichAndFilter=justAggregations)
    scalabilityView=scalabilityView.drop(['problemPath'],axis=1)                                    
    scalabilityView=scalabilityView[['hyperLevelMethod','modelSize','threads','sec']]
    scalabilityView=scalabilityView.sort_values(by=['hyperLevelMethod','modelSize','threads'])
    scalabilityView=scalabilityView.groupby(['modelSize','hyperLevelMethod']).apply(calculateSpeedup)
    scalabilityView=scalabilityView[scalabilityView['threads']>1]
    print(scalabilityView.to_latex(index=False))

def fillStepsMinValue(steps,til):
    filled=[]
    min=steps[0][1]
    i=0
    # (0,10)
    # (3,11)
    # (5,1)
    for (step,value) in steps:
        if min > value:
            min=value
        for j in range(i,step+1):
            filled.append(min)
        i=step+1
    for j in range(i,til):
        filled.append(min)
    return filled
        
RANDOM_CONTROL_GROUP_EXPERIMENT_RECORDS_PATH="../../logs/randomHH/records.json"
def createSAPlots():
    random=createTestGroupView(RANDOM_CONTROL_GROUP_EXPERIMENT_RECORDS_PATH,
                                    (SAStepsMetric,"hashSHA256"),
                                    recordToExperiment,
                                    set([]),
                                    set([]),
                                    {},
                                    justAggregations)
    # random=random.reset_index(drop=True)
    random=random.drop(['problemPath'],axis=1)
    print(random.to_latex())
    sa=createTestGroupView(SA_EXPERIMENT_RECORDS_PATH,
                                    (SAStepsMetric,"hashSHA256"),
                                    recordToExperiment,
                                    set([]),
                                    set([]),
                                    {},
                                    justAggregations)
    customHYS=pd.DataFrame(onlySteps(loadResultsSteps(CustomHYSPath)))
    rosenbrockCHYSResults=customHYS[(customHYS['problemName'] == 'PROBLEM_ROSENBROCK') & (customHYS['baselevelIterations'] == int(100))]

    RANDOM_5D=list(map(lambda at: at['med_+_iqr'],json.loads(random.loc['e9272933d50d87930781c5afb72b76f472c13d05278107d6181a6b51eeb0eb5b']['steps'])))
    RANDOM_5D_MIN=fillStepsMinValue(list(zip(range(0,len(RANDOM_5D)),RANDOM_5D)),len(RANDOM_5D))
    SA_5D=list(map(lambda at: at['med_+_iqr'],json.loads(sa.loc[ '481b0deaadf28552e7958e2c61feab8e36456416ec3c3aea512f854970783b22']['steps'])))
    SA_5D_MIN=fillStepsMinValue(list(zip(range(0,len(SA_5D)),SA_5D)),len(SA_5D))
    CUSTOMHYS_5D=fillStepsMinValue(list(map(lambda at: (int(at['step'].split('-')[0]),at['med_iqr']),json.loads(rosenbrockCHYSResults[rosenbrockCHYSResults['modelSize']==5]['steps'].to_list()[0]))),len(RANDOM_5D))
    data_series_5D = [
    # (range(0, len(SA_5D)), SA_5D, 'SA'),
    # (range(0, len(RANDOM_5D)), RANDOM_5D, 'RANDOM'),
    (range(0, len(RANDOM_5D_MIN)), RANDOM_5D_MIN, 'RANDOM_MIN'),
    (range(0, len(SA_5D_MIN)), SA_5D_MIN, 'SA_MIN'),
    (range(0, len(CUSTOMHYS_5D)), CUSTOMHYS_5D, 'CUSTOMHYS'),
    ]
    plot_series(data_series_5D, title='5D comparison',scale='log')

    RANDOM_50D=list(map(lambda at: at['med_+_iqr'],json.loads(random.loc['97142ed9a3d719cb637d12cfc10911779f738eacbe00f93df6269ee331ad408b']['steps'])))
    RANDOM_50D_MIN=fillStepsMinValue(list(zip(range(0,len(RANDOM_50D)),RANDOM_50D)),len(RANDOM_50D))
    SA_50D=list(map(lambda at: at['med_+_iqr'],json.loads(sa.loc[ 'fa229fb84688dde5cd4721f4fa64de2fec91a2ba82f3e67a9920a71fe0510606']['steps'])))
    SA_50D_MIN=fillStepsMinValue(list(zip(range(0,len(SA_50D)),SA_50D)),len(SA_50D))
    CUSTOMHYS_50D=fillStepsMinValue(list(map(lambda at: (int(at['step'].split('-')[0]),at['med_iqr']),json.loads(rosenbrockCHYSResults[rosenbrockCHYSResults['modelSize']==50]['steps'].to_list()[0]))),len(RANDOM_50D))
    data_series_50D = [
    # (range(0, len(SA_50D)), SA_50D, 'SA'),
    # (range(0, len(RANDOM_50D)), RANDOM_50D, 'RANDOM'),
    (range(0, len(RANDOM_50D_MIN)), RANDOM_50D_MIN, 'RANDOM_MIN'),
    (range(0, len(SA_50D_MIN)), SA_50D_MIN, 'SA_MIN'),
    (range(0, len(CUSTOMHYS_50D)), CUSTOMHYS_50D, 'CUSTOMHYS')]
    plot_series(data_series_50D, title='50D comparison',scale='log')

# SAAnalysis()
#SATempAnalysis()
# ScalabilityAnalysis()
createSAPlots()