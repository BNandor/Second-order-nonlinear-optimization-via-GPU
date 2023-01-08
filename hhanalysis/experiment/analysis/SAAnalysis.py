from commonAnalysis import *
from dict_hash import sha256

def enrichAndFilterSA(recordsWithMetrics,aggregations,experimentColumns):
    HHView=enrichAggregations(recordsWithMetrics,aggregations,experimentColumns+["minMedIQR"])
    return HHView[ (HHView['trialStepCount']==100) & (HHView['HH-SA-temp'] == 10000) & (HHView['HH-SA-alpha'] == 50)]

def enrichAndFilterSATemp(recordsWithMetrics,aggregations,experimentColumns):
    HHView=enrichAggregations(recordsWithMetrics,aggregations,experimentColumns+["minMedIQR"])
    return HHView[ (HHView['problemName']=="PROBLEM_ROSENBROCK") ]


# experiment(problems(name,path),..) -> experiment(problemName,problemPath,..)
def experimentToDFExperimentSA(experimentAndMetrics):
    experiment=experimentAndMetrics['experiment']
    experiment["problemName"]=experiment["problems"][0]
    experiment["problemPath"]=experiment["problems"][1]
    experiment.pop("problems")
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

def getControlGroupDF():
    return pd.DataFrame(stepsToResults(loadResultsSteps(CustomHYSPath)))
    
def getGroups(recordsPath):
    controlGroup=getControlGroupDF()
    testGroupDF=createTestGroupView(recordsPath,
                                    (filterMetricPropertiesSA,"hashSHA256"),
                                    experimentToDFExperimentSA,
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
    print(comparisonDF.to_latex(index=False))
    print(sha256(comparisonDF.to_dict()))
    createAndSaveContext(comparisonDF,independentColumns)

def SATempAnalysis():
    controlGroup=getControlGroupDF()
    # (experimentColumns,minF,HH-SA-Temp,HH-SA-alpha)
    tempView=createTestGroupView(SA_EXPERIMENT_RECORDS_PATH,
                                    (filterMetricPropertiesSA,"hashSHA256"),
                                    experimentToDFExperimentSA,
                                    set(['HH-SA-temp','HH-SA-alpha','trialStepCount']),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    enrichAndFilterSATemp)
    tempView=tempView.drop(['problemName','populationSize','problemPath','trialSampleSizes'],axis=1)
    tempView=tempView[tempView['modelSize'] > 5]
    tempView=tempView[['modelSize','baselevelIterations','minMedIQR','trialStepCount','HH-SA-temp','HH-SA-alpha']]
    tempView.sort_values(by=['modelSize','baselevelIterations'])


    print(tempView.to_latex(index=False))
    
    # tempRankView=tempView.groupby(['HH-SA-temp','HH-SA-alpha','trialStepCount']).count().reset_index()[['HH-SA-temp','HH-SA-alpha','trialStepCount','modelSize']]
    # tempRankView=tempRankView.rename({'modelSize':'bestCount'})
    # print(tempRankView.to_latex())
    # print(tempRankView.to_html())
SAAnalysis()
# SATempAnalysis()