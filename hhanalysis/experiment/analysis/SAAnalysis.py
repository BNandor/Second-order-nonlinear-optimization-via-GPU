from commonAnalysis import *
from dict_hash import sha256
from commonPlots import *
from common import *
import tabloo
import numpy as np

def enrichAndFilterSA(recordsWithMetrics,aggregations,experimentColumns):
    HHView=mergeOn(recordsWithMetrics,aggregations,experimentColumns+["minMedIQR"])
    return HHView[ (HHView['trialStepCount']==100) & (HHView['HH-SA-temp'] == 10000) & (HHView['HH-SA-alpha'] == 50)]

def enrichAndFilterSATemp(recordsWithMetrics,aggregations,experimentColumns):
    HHView=mergeOn(recordsWithMetrics,aggregations,experimentColumns+["minMedIQR"])
    return HHView[ (HHView['problemName']=="PROBLEM_ROSENBROCK") ]

def enrichAndFilterMealpy(recordsWithMetrics,aggregations,experimentColumns):
    HHView=mergeOn(recordsWithMetrics,aggregations,experimentColumns+["minMedIQR"])
    return HHView[ (HHView['trialStepCount']==100) ]

def justAggregations(recordsWithMetrics,aggregations,experimentColumns):
    return aggregations


# record(experiment(problems(name,path),..),metadata) -> experiment(problemName,problemPath,..)
def recordToExperiment(record):
    experiment=record['experiment']
    experiment["problemName"]=experiment["problems"][0]
    experiment["problemPath"]=experiment["problems"][1]
    experiment.pop("problems")
    return experiment

def mealpyRecordToExperiment(record):
    experiment=record['experiment']
    experiment["problemName"]=experiment["problems"][0]
    experiment["minMedIQR"]=record['metadata']['med_iqr']
    experiment["steps"]=json.dumps(record['metadata']['trials'])
    experiment["trialStepCount"]=experiment["hhsteps"]
    experiment["hyperLevel-id"]=experiment["optimizer"]
    experiment.pop("problems")
    experiment.pop("hhsteps")
    experiment.pop("optimizer")
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
def minTrialAverage(trials):
    minAvg=float('inf')
    minStd=0
    for trial in trials:
        trialAverage=np.average(trial['performanceSamples'])
        if(trialAverage<minAvg):
            minAvg=trialAverage
            minStd=np.std(trial['performanceSamples'])
    return (minAvg,minStd)

def filterMetricPropertiesAverageAndMedIQR(metric):
    (minAvg,minStd)=minTrialAverage(metric['trials'])
    return {
        "minMedIQR":metric["minBaseLevelStatistic"],
        "minAvg":minAvg,
        "minStd":minStd,
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

def categoryTransitionMetric(metric):
    return {
        "init->perturb":metric["bestParameters"]["OptimizerChainInitializerSimplex"]["perturbator"]["value"],
        "init->refiner":metric["bestParameters"]["OptimizerChainInitializerSimplex"]["refiner"]["value"],
        "perturb->refiner":metric["bestParameters"]["OptimizerChainPerturbatorSimplex"]["refiner"]["value"],
        "perturb->selector":metric["bestParameters"]["OptimizerChainPerturbatorSimplex"]["selector"]["value"],
        "refiner->refiner":metric["bestParameters"]["OptimizerChainRefinerSimplex"]["refiner"]["value"],
        "refiner->selector":metric["bestParameters"]["OptimizerChainRefinerSimplex"]["selector"]["value"],
        "selector->perturb":metric["bestParameters"]["OptimizerChainSelectorSimplex"]["perturbator"]["value"],
        "GD_ITER":metric["bestParameters"]["RefinerGDOperatorParams"]["GD_FEVALS"]["value"],
        "LBFGS_ITER":metric["bestParameters"]["RefinerLBFGSOperatorParams"]["LBFGS_FEVALS"]["value"],
        "minMedIQR":metric["minBaseLevelStatistic"],
        "hashSHA256":metric["experimentHashSha256"]
    }

def getCustomHySControlGroupDF():
    return pd.DataFrame(stepsToResults(loadResultsSteps(CustomHYSPath)))
def getGroups(recordsPath):
    controlGroup=getCustomHySControlGroupDF()
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
MEALPY_EXPERIMENT_RECORDS_PATH="../../logs/mealpyPerf/records.json"
def createSA_RANDOM_CUSTOMHYS_PlotSeries(customHYSFilterToOneResult,mealpyFilterToOneResult,ids):
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
    # mealpy=createTestGroupView(MEALPY_EXPERIMENT_RECORDS_PATH,
    #                                 (noMetric,"hashSHA256"),
    #                                 mealpyRecordToExperiment,
    #                                 set([]),
    #                                 set([]),
    #                                 {},
    #                                 justAggregations,enrichWithMetrics=False)
    # mealpySteps=mealpy[selectAllMatchAtLeastOne(mealpy,mealpyFilterToOneResult)]['steps'].to_list()
    # print(mealpy)
    # print(mealpyFilterToOneResult)
    # assert len(mealpySteps)==1
    # mealpySteps=json.loads(mealpySteps[0])
    RANDOM_STEPS=list(map(lambda at: at['med_+_iqr'],randomSteps))
    RANDOM_STEPS_MIN=fillStepsMinValue(list(zip(range(0,len(RANDOM_STEPS)),RANDOM_STEPS)),len(RANDOM_STEPS))
    SA_STEPS=list(map(lambda at: at['med_+_iqr'],saSteps))
    SA_STEPS_MIN=fillStepsMinValue(list(zip(range(0,len(SA_STEPS)),SA_STEPS)),len(SA_STEPS))
    CUSTOMHYS_5D=fillStepsMinValue(list(map(lambda at: (int(at['step'].split('-')[0]),at['med_iqr']),json.loads(rosenbrockCHYSResults['steps'].to_list()[0]))),len(RANDOM_STEPS))
    # MEALPY_STEPS=fillStepsMinValue(list(zip(range(0,len(mealpySteps)),mealpySteps)),len(RANDOM_STEPS))
    data_series = [
    # (range(0, len(SA_STEPS)), SA_STEPS, 'SA'),
    # (range(0, len(RANDOM_STEPS)), RANDOM_STEPS, 'RANDOM'),
    (range(0, len(RANDOM_STEPS_MIN)), RANDOM_STEPS_MIN, 'Random NMHH'),
    (range(0, len(SA_STEPS_MIN)), SA_STEPS_MIN, 'NMHH'),
    (range(0, len(CUSTOMHYS_5D)), CUSTOMHYS_5D, 'CUSTOMHyS'),
    # (range(0, len(MEALPY_STEPS)), MEALPY_STEPS, 'MEALPY'),
    ]
    return data_series

def comparisonSeriesFor(baselevelIterations,modelSize,problemName,optimizers):
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
    params["hyperLevelMethod"]=zipWithProperty(optimizers,"hyperLevelMethod")
    customHYSFilterToOneResult=[("baselevelIterations",[baselevelIterations]),('modelSize',[modelSize]),('problemName',[problemName])]
    mealpyFilterToOneResult=[("baselevelIterations",[baselevelIterations]),('modelSize',[modelSize]),('problemName',[problemName]),('hyperLevel-id',optimizers)]
    return createSA_RANDOM_CUSTOMHYS_PlotSeries(customHYSFilterToOneResult,mealpyFilterToOneResult,possibleExperimentIds(experimentParams=params))
    
def optimizerMethodsComparisonPlot():
    optimizers=["RANDOM",None]
    performances=[comparisonSeriesFor(100,dim,"PROBLEM_ROSENBROCK",optimizers) for dim in [5,50,100,500]]
    
    titles=[f"Rosenbrock {dim} dimensions" for dim in [5,50,100,500]]
    plot_series(performances,titles, x_label='steps', y_label=' best fitness (log)',scale='log',file_name=f"plots/HH_SA_RAND_CUSTOMHYS_Rosenbrock.svg")
{
    # performances=[comparisonSeriesFor(100,dim,"PROBLEM_SCHWEFEL223") for dim in [5,50,100,500]]
    # titles=[f"Schwefel 2.23 {dim} dimensions" for dim in [5,50,100,500]]
    # plot_series(performances,titles, x_label='steps', y_label='log cost',scale='log',file_name=f"plots/HH_SA_RAND_CUSTOMHYS_Schwefel.svg")

    # performances=[comparisonSeriesFor(100,dim,"PROBLEM_QING") for dim in [5,50,100,500]]
    # titles=[f"Qing {dim} dimensions" for dim in [5,50,100,500]]
    # plot_series(performances,titles, x_label='steps', y_label='log cost',scale='log',file_name=f"plots/HH_SA_RAND_CUSTOMHYS_QING.svg")
}

MEALPY_EXPERIMENT_RECORDS_PATH="../../logs/mealpyPerf/records.json"
SAPERTURB_EXPERIMENT_RECORDS_PATH="../../logs/SAPerturb/records.json"
GA_EXPERIMENT_RECORDS_PATH="../../logs/GA/records.json"
DE_EXPERIMENT_RECORDS_PATH="../../logs/DE/records.json"
RANDOM_GA_EXPERIMENT_RECORDS_PATH="../../logs/RANDOM-GA/records.json"
RANDOM_DE_EXPERIMENT_RECORDS_PATH="../../logs/RANDOM-DE/records.json"
SAREFINE_EXPERIMENT_RECORDS_PATH="../../logs/SARefine/records.json"
LBFGS_EXPERIMENT_RECORDS_PATH="../../logs/LBFGS/records.json"
GD_EXPERIMENT_RECORDS_PATH="../../logs/GD/records.json"
def methodsComparison():
    controlGroup=createTestGroupView(MEALPY_EXPERIMENT_RECORDS_PATH,
                                    (None,"hashSHA256"),
                                    mealpyRecordToExperiment,
                                    set(),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    enrichAndFilterMealpy,enrichWithMetrics=False)
    testGroupDF=createTestGroupView(SA_EXPERIMENT_RECORDS_PATH,
                                    (filterMetricPropertiesSA,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    enrichAndFilterSA)
    sarefineGroup=createTestGroupView(SAREFINE_EXPERIMENT_RECORDS_PATH,
                                    (filterMetricPropertiesSA,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    enrichAndFilterSA)
    lbfgsGroup=createTestGroupView(LBFGS_EXPERIMENT_RECORDS_PATH,
                                    (filterMetricPropertiesSA,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    justAggregations)            
    gdGroup=createTestGroupView(GD_EXPERIMENT_RECORDS_PATH,
                                    (filterMetricPropertiesSA,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    justAggregations)      
    saperturbGroup=createTestGroupView(SAPERTURB_EXPERIMENT_RECORDS_PATH,
                                    (filterMetricPropertiesAverageAndMedIQR,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR","minAvg","minStd"]),
                                    {'minAvg':'min'},
                                    justAggregations)
          
    gaGroup=createTestGroupView(GA_EXPERIMENT_RECORDS_PATH,
                                  (filterMetricPropertiesSA,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    justAggregations)
    deGroup=createTestGroupView(DE_EXPERIMENT_RECORDS_PATH,
                                    (filterMetricPropertiesSA,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    justAggregations)     
    randomgaGroup=createTestGroupView(RANDOM_GA_EXPERIMENT_RECORDS_PATH,
                                   (filterMetricPropertiesAverageAndMedIQR,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR","minAvg","minStd"]),
                                    {'minAvg':'min'},
                                    justAggregations)      
     
    randomdeGroup=createTestGroupView(RANDOM_DE_EXPERIMENT_RECORDS_PATH,
                                   (filterMetricPropertiesAverageAndMedIQR,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR","minAvg","minStd"]),
                                    {'minAvg':'min'},
                                    justAggregations)                           
                                  
    customhysDF=getCustomHySControlGroupDF()
    customhysDF['hyperLevel-id']='CUSTOMHyS'
    customhysDF=dropIrrelevantColumns(customhysDF,set(['hyperLevel-id','modelSize','problemName','baselevelIterations','minMedIQR']))
    controlGroup=dropIrrelevantColumns(controlGroup,set(['hyperLevel-id','modelSize','problemName','baselevelIterations','minMedIQR']))
    testGroupDF=dropIrrelevantColumns(testGroupDF,set(['modelSize','problemName','baselevelIterations','minMedIQR']))
    testGroupDF=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('baselevelIterations',[100]),('modelSize',[5,50,100,500])])]
    customhysDF=customhysDF[selectAllMatchAtLeastOne(customhysDF,[('baselevelIterations',[100]),('modelSize',[5,50,100,500])])]
    testGroupDF['hyperLevel-id']='NMHH'
    sarefineGroup=dropIrrelevantColumns(sarefineGroup,set(['modelSize','problemName','baselevelIterations','minMedIQR']))
    sarefineGroup['hyperLevel-id']='SA-REFINE-NMHH'
    sarefineGroup=sarefineGroup[selectAllMatchAtLeastOne(sarefineGroup,[('baselevelIterations',[100]),('modelSize',[5,50,100,500])])]
    lbfgsGroup=dropIrrelevantColumns(lbfgsGroup,set(['modelSize','problemName','baselevelIterations','minMedIQR']))
    lbfgsGroup['hyperLevel-id']='LBFGS'
    lbfgsGroup=lbfgsGroup[selectAllMatchAtLeastOne(lbfgsGroup,[('baselevelIterations',[100]),('modelSize',[5,50,100,500])])]
    gdGroup=dropIrrelevantColumns(gdGroup,set(['modelSize','problemName','baselevelIterations','minMedIQR']))
    gdGroup['hyperLevel-id']='GD'
    gdGroup=gdGroup[selectAllMatchAtLeastOne(gdGroup,[('baselevelIterations',[100]),('modelSize',[5,50,100,500])])]
    saperturbGroup=dropIrrelevantColumns(saperturbGroup,set(['modelSize','problemName','baselevelIterations','minAvg']))
    saperturbGroup['hyperLevel-id']='SA-PERTURB'
    saperturbGroupBig=saperturbGroup.copy()
    saperturbGroupBig['hyperLevel-id']='SA-PERTURB-BIG'
    saperturbGroup=saperturbGroup[selectAllMatchAtLeastOne(saperturbGroup,[('baselevelIterations',[100]),('modelSize',[5,50,100,500])])]
    saperturbGroupBig=saperturbGroupBig[selectAllMatchAtLeastOne(saperturbGroupBig,[('baselevelIterations',[1000]),('modelSize',[5,50,100,500])])]
    gaGroup=dropIrrelevantColumns(gaGroup,set(['modelSize','problemName','baselevelIterations','minMedIQR']))
    gaGroup['hyperLevel-id']='GA'
    gaGroupBig=gaGroup.copy()
    gaGroupBig['hyperLevel-id']='GA-BIG'
    gaGroup=gaGroup[selectAllMatchAtLeastOne(gaGroup,[('baselevelIterations',[100]),('modelSize',[5,50,100,500])])]
    gaGroupBig=gaGroupBig[selectAllMatchAtLeastOne(gaGroupBig,[('baselevelIterations',[1000]),('modelSize',[5,50,100,500])])]
    deGroup=dropIrrelevantColumns(deGroup,set(['modelSize','problemName','baselevelIterations','minMedIQR']))
    deGroup['hyperLevel-id']='DE'
    deGroupBig=deGroup.copy()
    deGroupBig['hyperLevel-id']='DE-BIG'
    deGroup=deGroup[selectAllMatchAtLeastOne(deGroup,[('baselevelIterations',[100]),('modelSize',[5,50,100,500])])]
    deGroupBig=deGroupBig[selectAllMatchAtLeastOne(deGroupBig,[('baselevelIterations',[1000]),('modelSize',[5,50,100,500])])]
    randomgaGroup=dropIrrelevantColumns(randomgaGroup,set(['modelSize','problemName','baselevelIterations','minAvg']))
    randomgaGroup['hyperLevel-id']='RANDOM-GA'
    randomgaGroupBig=randomgaGroup.copy()
    randomgaGroupBig['hyperLevel-id']='RANDOM-GA-BIG'
    randomgaGroup=randomgaGroup[selectAllMatchAtLeastOne(randomgaGroup,[('baselevelIterations',[100]),('modelSize',[5,50,100,500])])]
    randomgaGroupBig=randomgaGroupBig[selectAllMatchAtLeastOne(randomgaGroupBig,[('baselevelIterations',[1000]),('modelSize',[5,50,100,500])])]
    randomdeGroup=dropIrrelevantColumns(randomdeGroup,set(['modelSize','problemName','baselevelIterations','minAvg']))
    randomdeGroup['hyperLevel-id']='RANDOM-DE'
    randomdeGroupBig=randomdeGroup.copy()
    randomdeGroupBig['hyperLevel-id']='RANDOM-DE-BIG'
    randomdeGroup=randomdeGroup[selectAllMatchAtLeastOne(randomdeGroup,[('baselevelIterations',[100]),('modelSize',[5,50,100,500])])]
    randomdeGroupBig=randomdeGroupBig[selectAllMatchAtLeastOne(randomdeGroupBig,[('baselevelIterations',[1000]),('modelSize',[5,50,100,500])])]
    # all=pd.concat([customhysDF,controlGroup,testGroupDF,sarefineGroup,lbfgsGroup,gdGroup,saperturbGroup,gaGroup,deGroup])
    # all=pd.concat([sarefineGroup,lbfgsGroup,gdGroup])
    # all=pd.concat([saperturbGroup,saperturbGroupBig,gaGroup,deGroup,randomgaGroup,randomdeGroup])
    all=pd.concat([saperturbGroupBig,randomgaGroupBig,randomdeGroupBig])
    
    all=all.drop(['baselevelIterations'],axis=1)
    all=all.sort_values(by=['modelSize',"problemName",'minAvg'])
    all=all[['problemName','modelSize','hyperLevel-id','minAvg']]
    all=all.groupby(['problemName','modelSize'])
    transpose=pd.DataFrame()
    optimizers=set()
    for (group,groupIndex) in all:
        transposedRow={}
        transposedRow['problemName']=group[0]
        transposedRow['modelSize']=group[1]
        for index,row in groupIndex.iterrows():
            transposedRow[row["hyperLevel-id"]]=row["minAvg"]
            optimizers.add(row["hyperLevel-id"])
        transpose=transpose.append(transposedRow,ignore_index=True)
    print(printMinResultEachRow(transpose,['problemName','modelSize'],optimizers))
    tabloo.show(transpose)
    # tabloo.show(all)
    # print(all.to_latex(index=False))
    # export the styled dataframe to LaTeX
    print(transpose.to_latex(index=False))
    # return (controlGroup,testGroupDF)

def all5000IterationResults():
    testGroupDF=createTestGroupView(SA_EXPERIMENT_RECORDS_PATH,
                                    (filterMetricPropertiesSA,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    enrichAndFilterSA)
    testGroupDF=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('baselevelIterations',[5000])])]
    groupedByModelSize=testGroupDF.groupby(['modelSize'])
    transpose=pd.DataFrame()

    for (group,groupIndex) in groupedByModelSize:
        transposedRow={}
        for  index,row in groupIndex.iterrows():    
            transposedRow['modelSize']=row['modelSize']
            transposedRow[row['problemName']]=row["minMedIQR"]
        transpose=transpose.append(transposedRow,ignore_index=True)
    
    # tabloo.show(transpose)
    print(transpose.to_latex(index=False))    

def createCategoryTransitionHeatMapsAt(path,baselevelIterations=100):
    testGroupDF=createTestGroupView(path,
                                    (categoryTransitionMetric,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    enrichAndFilterSA)
    testGroupDF=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('baselevelIterations',[baselevelIterations])])]                                    
    testGroupDF['GD_ITER_%']=testGroupDF['GD_ITER']/testGroupDF['baselevelIterations']
    testGroupDF['LBFGS_ITER_%']=testGroupDF['LBFGS_ITER']/testGroupDF['baselevelIterations']
    testGroupDF=testGroupDF.drop(['baselevelIterations','GD_ITER','LBFGS_ITER','HH-SA-alpha','populationSize','trialStepCount','problemPath','HH-SA-temp','trialSampleSizes'],axis=1)                                    
    testGroupDF=testGroupDF.sort_values(by=['refiner->refiner'])
    testGroupDF['P']=testGroupDF.apply(lambda row: json.dumps([
       [0.0,row['perturb->refiner'],row['perturb->selector']],#perturb
       [0.0,row['refiner->refiner'],row['refiner->selector']],#refine
       [row['selector->perturb'],0.0,0.0],#select
    ]),axis=1)
    # tabloo.show(testGroupDF)
    return testGroupDF 
def getProbabilityTransitionsMatchOne(df,dim):
    assert(len(df[df['modelSize']==dim]['P'].to_list())==1)
    return np.array(json.loads(df[df['modelSize']==dim]['P'].to_list()[0]))
def getInitialDistributionsMatchOne(df,dim):
    iperturb=df[df['modelSize']==dim]['init->perturb'].to_list()
    irefiner=df[df['modelSize']==dim]['init->refiner'].to_list()
    assert(len(iperturb)==1)
    assert(len(irefiner)==1)
    return np.array([[iperturb[0]],\
                     [irefiner[0]],\
                     [0.0]])
def createTransitionProbabilityHeatMap():
    baselevelIterations=100
    testGroupDF=createCategoryTransitionHeatMapsAt(SA_EXPERIMENT_RECORDS_PATH,baselevelIterations=baselevelIterations)
    rastrigin=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('problemName',['PROBLEM_RASTRIGIN'])])]                                    
    rosenbrock=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('problemName',['PROBLEM_ROSENBROCK'])])]                                  
    qing=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('problemName',['PROBLEM_QING'])])]                                    
    trid=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('problemName',['PROBLEM_TRID'])])]                                    
    schwefel=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('problemName',['PROBLEM_SCHWEFEL223'])])]                                    
    styblinskitang=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('problemName',['PROBLEM_STYBLINSKITANG'])])]                                  
    Ps=[]
    xticks=[]
    yticks=[]
    states=['Perturb','Refine','Select']
    titles=[]
    figuretitles=[]
    xlabelTitles=[]
    ylabelTitles=[]
    dims=[5,50,100,500]
    # problems=['Rastrigin','Rosenbrock','Qing','Trid','Schwefel223','Styblinksi Tang']
    # problemdfs=[rastrigin,rosenbrock,qing,trid,schwefel,styblinskitang]
    problems=['Styblinksi Tang','Rosenbrock']
    problemdfs=[styblinskitang,rosenbrock]
    for problem,df in zip(problems,problemdfs):
        plotRow=[]
        rowxTicks=[]
        rowyTicks=[]
        rowTitles=[]
        figureTitleRow=[]
        xlabelTitleRow=[]
        ylabelTitleRow=[]
        for dim in dims:
            plotRow.append(getInitialDistributionsMatchOne(df,dim))
            rowxTicks.append('')
            rowyTicks.append(states)
            rowTitles.append('')
            xlabelTitleRow.append('')
            ylabelTitleRow.append('Initial distribution')
            plotRow.append(getProbabilityTransitionsMatchOne(df,dim))
            rowxTicks.append(states)
            rowyTicks.append(states)
            rowTitles.append('Transition probabilities')
            figureTitleRow.append(f"{problem} {dim}")   
            xlabelTitleRow.append('Next category')
            ylabelTitleRow.append('Current category') 
        Ps.append(plotRow)
        xticks.append(rowxTicks)
        yticks.append(rowyTicks)
        titles.append(rowTitles)
        figuretitles.append(figureTitleRow)
        xlabelTitles.append(xlabelTitleRow)
        ylabelTitles.append(ylabelTitleRow)

    width_ratios=[1,3]
    height_ratios=[1]
    rowsize=3
    plotHeatmap(Ps,len(problems),len(dims)*2,xticks,yticks,titles,xlabelTitles,ylabelTitles,figuretitles,
                        width_ratios=width_ratios,height_ratios=height_ratios,subfigdim=(len(problems),len(dims)) ,figsize=(17,len(problems)*rowsize),
                        filename=f"plots/P_{problems}_{dims}_{baselevelIterations}.svg")

# SAAnalysis()
# SATempAnalysis()
# ScalabilityAnalysis()
# optimizerMethodsComparisonPlot()
methodsComparison()
# all5000IterationResults()
# createTransitionProbabilityHeatMap()