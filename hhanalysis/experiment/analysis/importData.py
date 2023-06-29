from commonAnalysis import *
ROOT=f"{os.path.dirname(os.path.abspath(__file__))}"
LOGS_ROOT=f"{ROOT}/../../logs"

SA_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/records.json"
SCALABILITY_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/scalabilityTests/records.json"
SA_GA_DE_GD_LBFGS_RECORDS_PATH=f"{LOGS_ROOT}/SA-NMHH/GA_DE_GD_LBFGS/records.json"
RANDOM_CONTROL_GROUP_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/randomHH/records.json"
MEALPY_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/mealpyPerf/records.json"
MEALPY_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/mealpyPerf/records.json"
SAPERTURB_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SAPerturb/records.json"
GA_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/GA/records.json"
DE_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/DE/records.json"
RANDOM_GA_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/RANDOM-GA/records.json"
RANDOM_DE_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/RANDOM-DE/records.json"
RANDOM_SA_GWO_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-NMHH/GWO/records.json"
SA_CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-CMA-ES-NMHH/GWO/records.json"
SA_CMA_ES_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-CMA-ES-NMHH/records.json"
BIGSA_CMA_ES_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-CMA-ES-NMHH/GA_DE_GD_LBFGS/bigSA/records.json"
BIGSA_CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-CMA-ES-NMHH/GWO/bigSA/records.json"
CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/CMA-ES/GWO/records.json"
CMA_ES_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/CMA-ES/records.json"
MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/MADS-NMHH/GA_DE_GD_LBFGS_GWO/records.json"
MADS_NMHH_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/MADS-NMHH/GA_DE_GD_LBFGS/records.json"                          
SA_MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-MADS-NMHH/GA_DE_GD_LBFGS_GWO/records.json"
SA_MADS_NMHH_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-MADS-NMHH/GA_DE_GD_LBFGS/records.json"
BIGSA_MADS_NMHH_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-MADS-NMHH/GA_DE_GD_LBFGS/bigSA/records.json"
BIGSA_MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-MADS-NMHH/GA_DE_GD_LBFGS_GWO/bigSA/records.json"
SAREFINE_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SARefine/records.json"
LBFGS_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/LBFGS/records.json"
GD_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/GD/records.json"


def enrichAndFilterSA(recordsWithMetrics,aggregations,experimentColumns):
    HHView=mergeOn(recordsWithMetrics,aggregations,experimentColumns+["minMedIQR"])
    return HHView[ (HHView['trialStepCount']==100) & (HHView['HH-SA-temp'] == 10000) & (HHView['HH-SA-alpha'] == 50)]

def enrichAndFilterSATemp(recordsWithMetrics,aggregations,experimentColumns):
    HHView=mergeOn(recordsWithMetrics,aggregations,experimentColumns+["minMedIQR"])
    return HHView[ (HHView['problemName']=="PROBLEM_ROSENBROCK") ]

def enrichAndFilterMealpy(recordsWithMetrics,aggregations,experimentColumns):
    HHView=mergeOn(recordsWithMetrics,aggregations,experimentColumns+["minMedIQR"])
    return HHView[ (HHView['trialStepCount']==100) ]

def mergeOnAvg(recordsWithMetrics,aggregations,experimentColumns):
    return mergeOn(recordsWithMetrics,aggregations,experimentColumns+["minAvg"])

def mergeOnMinMedIQR(recordsWithMetrics,aggregations,experimentColumns):
    return mergeOn(recordsWithMetrics,aggregations,experimentColumns+["minMedIQR"])
    
def justAggregations(recordsWithMetrics,aggregations,experimentColumns):
    return aggregations

# RECORD -> EXPERIMENT

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

# record(experiment(problems(name,path),..),metadata) -> experiment(problemPath,modelSize...sec)
def recordToScalabilityExperiment(record):
    experiment={}
    experiment["problemPath"]=record['experiment']["problems"][1]
    experiment["modelSize"]=record['experiment']["modelSize"]
    experiment['hyperLevelMethod']=record['experiment']["hyperLevelMethod"]
    experiment["threads"]=record['metadata']['threads']
    experiment['sec']=record['metadata']["elapsedTimeSec"]
    return experiment

# METRIC FILTERS

def filterMetricPropertiesMinMedIQR(metric):
    return {
        "minMedIQR":metric["minBaseLevelStatistic"],
        "hashSHA256":metric["experimentHashSha256"]
    }


def minTrialAverage(trials):
    minAvg=float('inf')
    minStd=0
    samples=[]
    for trial in trials:
        # +np.std(trial['performanceSamples'])
        trialAverage=np.average(trial['performanceSamples'])
        if(trialAverage<minAvg):
            minAvg=trialAverage
            minStd=np.std(trial['performanceSamples'])
            samples=trial['performanceSamples']
    return (minAvg,minStd,samples)
def filterMetricPropertiesAverageAndMedIQR(metric):
    (minAvg,minStd,samples)=minTrialAverage(metric['trials'])
    return {
        "minMedIQR":metric["minBaseLevelStatistic"],
        "minAvg":minAvg,
        "minStd":minStd,
        "samples":json.dumps(samples),
        "hashSHA256":metric["experimentHashSha256"]
    }


def noMetric(metric):
    return {
        "hashSHA256":metric["experimentHashSha256"]
    }

def SAStepsWithMinMedIQRMetric(metric):
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