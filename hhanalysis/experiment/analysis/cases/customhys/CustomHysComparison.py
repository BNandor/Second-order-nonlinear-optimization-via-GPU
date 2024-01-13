import sys
import os
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from commonAnalysis import *
from importData import *
from loadCustomHYS import *
from dict_hash import sha256
from commonPlots import *
from common import *
import tabloo
import numpy as np

def createSA_RANDOM_CUSTOMHYS_PlotSeries(customHYSFilterToOneResult,mealpyFilterToOneResult,ids):
    random=createTestGroupView(f"{RANDOM_CONTROL_GROUP_EXPERIMENT_RECORDS_PATH}/records.json",
                                    (SAStepsWithMinMedIQRMetric,"hashSHA256"),
                                    recordToExperiment,
                                    set([]),
                                    set([]),
                                    {},
                                    justAggregations)
    randomSteps=json.loads(random.loc[matchOneIdInIndex(random.index,ids)]['steps'])
    sa=createTestGroupView(f"{SA_EXPERIMENT_RECORDS_PATH}/records.json",
                                    (SAStepsWithMinMedIQRMetric,"hashSHA256"),
                                    recordToExperiment,
                                    set([]),
                                    set([]),
                                    {},
                                    justAggregations)
    saSteps=json.loads(sa.loc[matchOneIdInIndex(sa.index,ids)]['steps'])
    # customHYS=pd.DataFrame(onlySteps(loadResultsSteps(CustomHYSPath)))
    # rosenbrockCHYSResults=customHYS[selectAllMatchAtLeastOne(customHYS,customHYSFilterToOneResult)]
    
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
    # CUSTOMHYS_5D=fillStepsMinValue(list(map(lambda at: (int(at['step'].split('-')[0]),at['med_iqr']),json.loads(rosenbrockCHYSResults['steps'].to_list()[0]))),len(RANDOM_STEPS))
    # MEALPY_STEPS=fillStepsMinValue(list(zip(range(0,len(mealpySteps)),mealpySteps)),len(RANDOM_STEPS))
    data_series = [
    # (range(0, len(SA_STEPS)), SA_STEPS, 'SA'),
    # (range(0, len(RANDOM_STEPS)), RANDOM_STEPS, 'RANDOM'),
    (range(0, len(RANDOM_STEPS_MIN)), RANDOM_STEPS_MIN, 'Random NMHH'),
    (range(0, len(SA_STEPS_MIN)), SA_STEPS_MIN, 'NMHH'),
    # (range(0, len(CUSTOMHYS_5D)), CUSTOMHYS_5D, 'CUSTOMHyS'),
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
              (problemName,"hhanalysis/logs/randomHH/qing.json"),
              (problemName,"hhanalysis/logs/styblinskitang.json"),
              (problemName,"hhanalysis/logs/randomHH/styblinskitang.json")],"problems")
    
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
    performances=[comparisonSeriesFor(100,dim,"PROBLEM_STYBLINSKITANG",optimizers) for dim in [5,50,100,500]]
    
    titles=[f"Styblinksi Tang {dim} dimensions" for dim in [5,50,100,500]]
    plot_series([performances],[titles], x_label='steps', y_label=' best fitness (log)',scales=['log'],file_name=f"../../plots/HH_SA_RAND_CUSTOMHYS_PROBLEM_STYBLINSKITANG.svg")
{
    # performances=[comparisonSeriesFor(100,dim,"PROBLEM_SCHWEFEL223") for dim in [5,50,100,500]]
    # titles=[f"Schwefel 2.23 {dim} dimensions" for dim in [5,50,100,500]]
    # plot_series(performances,titles, x_label='steps', y_label='log cost',scale='log',file_name=f"plots/HH_SA_RAND_CUSTOMHYS_Schwefel.svg")

    # performances=[comparisonSeriesFor(100,dim,"PROBLEM_QING") for dim in [5,50,100,500]]
    # titles=[f"Qing {dim} dimensions" for dim in [5,50,100,500]]
    # plot_series(performances,titles, x_label='steps', y_label='log cost',scale='log',file_name=f"plots/HH_SA_RAND_CUSTOMHYS_QING.svg")
}
    
optimizerMethodsComparisonPlot()