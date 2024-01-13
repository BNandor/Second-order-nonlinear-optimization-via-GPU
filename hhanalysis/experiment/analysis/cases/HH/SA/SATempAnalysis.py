import sys
import os
sys.path.insert(0, '../../../')

from commonAnalysis import *
from importData import *
from dict_hash import sha256
from commonPlots import *
from common import *
import tabloo
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

def SATempAnalysis():
    # (experimentColumns,minF,HH-SA-Temp,HH-SA-alpha)
    # {
    tempView=createTestGroupView(SA_EXPERIMENT_RECORDS_PATH+"/records.json",
                                    (filterMetricPropertiesMinMedIQR,"hashSHA256"),
                                    recordToExperiment,
                                    set(['HH-SA-temp','HH-SA-alpha','trialStepCount']),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    enrichAndFilterSATemp)
    # }
    # tempView=createTestGroupView(SA_EXPERIMENT_RECORDS_PATH,
    #                                 (filterMetricPropertiesMinMedIQR,"hashSHA256"),
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
    # tempView=tempView[tempView["modelSize"]==50] 
    # tempView=tempView[tempView["baselevelIterations"]==100]
    tempView=tempView.sort_values(by=['modelSize','baselevelIterations'],ascending=False)
    tempView=tempView.reset_index(drop=True)
    # print(tempView)
    print(tempView.to_latex(index=False))
    # tempView = tempView.sort_values('modelSize', ascending=False)
    print(tempView)
    grouped_data = tempView.groupby(['modelSize', 'baselevelIterations', 'HH-SA-temp'])
    
    # Aggregate the 'minMedIQR' values for each group
    result = grouped_data['minMedIQR'].agg(list).reset_index()
    
    # Explode the lists in 'minMedIQR' to separate rows
    result = result.explode('minMedIQR')

    # Convert 'minMedIQR' to numeric (assuming it's stored as strings)
    result['minMedIQR'] = pd.to_numeric(result['minMedIQR'])
    
# Create violin plots for each series
    # Create violin plots with separate axes for each series
# Create boxplots with separate rows for each 'baselevelIterations'
    prop = font_manager.FontProperties(fname=f'{ROOT}/plots/fonts/times-ro.ttf')    
    # plt.rcParams['font.family'] = prop.get_name()
    

    sns.set(font="serif",style="whitegrid")
    g = sns.FacetGrid(result, col='modelSize', row='baselevelIterations', height=4, aspect=1.5, margin_titles=True, sharey=False)
    g.map(sns.boxplot, 'HH-SA-temp', 'minMedIQR', palette="pastel")
    # g.map(sns.violinplot, 'HH-SA-temp', 'minMedIQR', split=True, inner="quart", palette="muted")
    g.set_axis_labels('SA temperature', 'median + IQR','iterations')
    g.set_titles(col_template="{col_name} dimensions", row_template="{row_name} iterations")
    g.despine(left=True, right=True, top=True, bottom=True)
    # plt.suptitle('Boxplot of minMedIQR for Each Model Size and Baselevel Iterations', y=0.9)
    plt.rcParams['font.family'] = 'serif'
    plt.savefig(f"{ROOT}/plots/parameterTuning.svg")
    
    plt.show()



    # tempRankView=tempView.groupby(['HH-SA-temp']).count().reset_index()[['HH-SA-temp','HH-SA-alpha','trialStepCount','modelSize']]
    # tempRankView=tempRankView.rename({'modelSize':'bestCount'})
    # print(tempRankView)
    # print(tempRankView.to_html())
    # }

SATempAnalysis()