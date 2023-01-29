from commonAnalysis import *
from dict_hash import sha256
from commonPlots import *
from common import *

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
def createSA_RANDOM_CUSTOMHYS_PlotSeries(customHYSFilterToOneResult,ids):
    random=createTestGroupView(RANDOM_CONTROL_GROUP_EXPERIMENT_RECORDS_PATH,
                                    (SAStepsMetric,"hashSHA256"),
                                    recordToExperiment,
                                    set([]),
                                    set([]),
                                    {},
                                    justAggregations)
    randomSteps=json.loads(random.loc[matchOneIdInIndex(random.index,ids)]['steps'])
    sa=createTestGroupView(SA_EXPERIMENT_RECORDS_PATH,
                                    (SAStepsMetric,"hashSHA256"),
                                    recordToExperiment,
                                    set([]),
                                    set([]),
                                    {},
                                    justAggregations)
    saSteps=json.loads(sa.loc[matchOneIdInIndex(sa.index,ids)]['steps'])
    customHYS=pd.DataFrame(onlySteps(loadResultsSteps(CustomHYSPath)))
    rosenbrockCHYSResults=customHYS[selectAllMatchAtLeastOne(customHYS,customHYSFilterToOneResult)]
    RANDOM_STEPS=list(map(lambda at: at['med_+_iqr'],randomSteps))
    RANDOM_STEPS_MIN=fillStepsMinValue(list(zip(range(0,len(RANDOM_STEPS)),RANDOM_STEPS)),len(RANDOM_STEPS))
    SA_STEPS=list(map(lambda at: at['med_+_iqr'],saSteps))
    SA_STEPS_MIN=fillStepsMinValue(list(zip(range(0,len(SA_STEPS)),SA_STEPS)),len(SA_STEPS))
    CUSTOMHYS_5D=fillStepsMinValue(list(map(lambda at: (int(at['step'].split('-')[0]),at['med_iqr']),json.loads(rosenbrockCHYSResults['steps'].to_list()[0]))),len(RANDOM_STEPS))
    data_series = [
    # (range(0, len(SA_STEPS)), SA_STEPS, 'SA'),
    # (range(0, len(RANDOM_STEPS)), RANDOM_STEPS, 'RANDOM'),
    (range(0, len(RANDOM_STEPS_MIN)), RANDOM_STEPS_MIN, 'Random NMHH'),
    (range(0, len(SA_STEPS_MIN)), SA_STEPS_MIN, 'SA-NMHH'),
    (range(0, len(CUSTOMHYS_5D)), CUSTOMHYS_5D, 'CUSTOMHyS'),
    ]
    return data_series

def comparisonSeriesFor(baselevelIterations,modelSize,problemName):
    params={}
    params["problems"]=zipWithProperty([
              (problemName,"hhanalysis/logs/rosenbrock.json"),
              (problemName,"hhanalysis/logs/randomHH/rosenbrock.json"),
              (problemName,"hhanalysis/logs/schwefel223.json"),
              (problemName,"hhanalysis/logs/randomHH/schwefel223.json"),
              (problemName,"hhanalysis/logs/qing.json"),
              (problemName,"hhanalysis/logs/randomHH/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([baselevelIterations],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([modelSize],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["RANDOM",None],"hyperLevelMethod")
    customHYSFilterToOneResult=[("baselevelIterations",[baselevelIterations]),('modelSize',[modelSize]),('problemName',[problemName])]
    return createSA_RANDOM_CUSTOMHYS_PlotSeries(customHYSFilterToOneResult,possibleExperimentIds(experimentParams=params))
    
def optimizerMethodsComparisonPlot():
    performances=[comparisonSeriesFor(100,dim,"PROBLEM_ROSENBROCK") for dim in [5,50,100,500]]
    titles=[f"Rosenbrock {dim} dimensions" for dim in [5,50,100,500]]
    plot_series(performances,titles, x_label='steps', y_label=' fitness (log)',scale='log',file_name=f"plots/HH_SA_RAND_CUSTOMHYS_Rosenbrock.svg")

    # performances=[comparisonSeriesFor(100,dim,"PROBLEM_SCHWEFEL223") for dim in [5,50,100,500]]
    # titles=[f"Schwefel 2.23 {dim} dimensions" for dim in [5,50,100,500]]
    # plot_series(performances,titles, x_label='steps', y_label='log cost',scale='log',file_name=f"plots/HH_SA_RAND_CUSTOMHYS_Schwefel.svg")

    # performances=[comparisonSeriesFor(100,dim,"PROBLEM_QING") for dim in [5,50,100,500]]
    # titles=[f"Qing {dim} dimensions" for dim in [5,50,100,500]]
    # plot_series(performances,titles, x_label='steps', y_label='log cost',scale='log',file_name=f"plots/HH_SA_RAND_CUSTOMHYS_QING.svg")

# SAAnalysis()
#SATempAnalysis()
# ScalabilityAnalysis()
optimizerMethodsComparisonPlot()