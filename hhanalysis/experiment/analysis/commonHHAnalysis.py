from commonAnalysis import *
from importData import *
from dict_hash import sha256
from commonPlots import *
from common import *
import tabloo
import numpy as np

def methodsComparison(all,metadata,block=True, barplotMapping=lambda x: 'blue',labelfunction= lambda x:x, optimizerOrderlist=[]):
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
    
    # printMinResultEachRow(transpose,['problemName','modelSize'],optimizersSet)
    
    addWilcoxRankSumResultToEachRow(transpose,['problemName','modelSize'],[f'{column}-samples' for column in metadata['optimizers']])
    statisticsforDimension=calculateWilcoxRanksumStatisticsForEachDimension(transpose,metadata['optimizers'])
    optimizerUsed= [x for _, x in sorted(zip(optimizerOrderlist, [opt for opt in metadata['optimizers']]))] if optimizerOrderlist!=[] else metadata['optimizers']
    printMinMedIQRStdHighlighWilcoxRanksums(transpose,optimizerUsed)

    # printStatisticsOfWilcoxRanksums(transpose,metadata['optimizers'])
    # printScoreOfWilcoxRanksums(transpose,metadata['optimizers'])
    scoresPerDim=printScoreOfWilcoxRanksumsPerDim(transpose,metadata['optimizers'])
    plot_optimizer_scores(scoresPerDim,barplotMapping,labelfunction)
    printStatisticsOfWilcoxRanksumsForEachDimension(statisticsforDimension)
    # plotWilcoxRanksums(transpose,6,len(metadata["modelSize"]),
    #                    list(map(lambda name:name.replace('-BIG',''),metadata['optimizers'])),
    #                 #    filename=f"plots/WILCOX_{[metadata[savecol] for savecol in metadata['saveMetadataColumns']]}.svg",
    #                    filename=None,
    #                    figsize=(13,8),blockPlot=True)
    
    # (dimensions,methods,values)=plotDataForWilcoxRanksumsComparisonPlot(statisticsforDimension,metadata['optimizers'])
    # plotMethodsComparison(dimensions,methods,values,'Cases','Winrates','Optimizer performances',block)

    # tabloo.show(comparisonTableData)
    # tabloo.show(all)
    # missingExperiments=transpose[list(transpose[metadata['optimizers']].columns[transpose[metadata['optimizers']].isnull().any()])+['modelSize','problemName']]
    # filteredExperiments=transpose[['problemName','modelSize']+metadata['optimizers']]
    # filteredExperiments=filteredExperiments.rename(columns=lambda c:c.replace('/benchmarks/dim/2_100/pop/30',''))
    transpose=transpose.sort_values(by=['problemName','modelSize'])
    # tabloo.show(transpose)
    # print(transpose.to_latex(index=False))
    # printMinMedIQRStdHighlighWilcoxRanksums(transpose,metadata['optimizers'])
    # export the styled dataframe to LaTeX
    # print(comparisonTableData.to_latex(index=False))
    
    # printMinAvgStdHighlighWilcoxRanksums(transpose,metadata['optimizers'])
    # printMinMedIQRStdHighlighWilcoxRanksums(transpose,metadata['optimizers'])
    # printLatexMinAvgStd(transpose,metadata['optimizers'])
    # return (mealpyMHs,nmhh)

def all5000IterationResults():
    testGroupDF=createTestGroupView(SA_EXPERIMENT_RECORDS_PATH,
                                    (filterMetricPropertiesMinMedIQR,"hashSHA256"),
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
def createOperatorTransitionHeatMapsAt(path,baselevelIterations=100):
    testGroupDF=createTestGroupView(path,
                                    (operatorTransitionMetric,"hashSHA256"),
                                    recordToExperiment,
                                    set(),
                                    set(["minMedIQR"]),
                                    {'minMedIQR':'min'},
                                    enrichAndFilterSA)
    testGroupDF=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('baselevelIterations',[baselevelIterations])])]                                    
    testGroupDF['GD_ITER_%']=testGroupDF['GD_ITER']/testGroupDF['baselevelIterations']
    testGroupDF['LBFGS_ITER_%']=testGroupDF['LBFGS_ITER']/testGroupDF['baselevelIterations']
    testGroupDF=testGroupDF.drop(['baselevelIterations','GD_ITER','LBFGS_ITER','HH-SA-alpha','populationSize','trialStepCount','problemPath','HH-SA-temp','trialSampleSizes'],axis=1)                                    
    testGroupDF['P_GA-DE']=testGroupDF.apply(lambda row: json.dumps([
       [row['GA->GA'],row['GA->DE']],
       [row['DE->GA'],row['DE->DE']],
    ]),axis=1)
    testGroupDF['P_GD-LBFGS']=testGroupDF.apply(lambda row: json.dumps([
       [row['GD->GD'],row['GD->LBFGS']],
       [row['LBFGS->GD'],row['LBFGS->LBFGS']],
    ]),axis=1)
    # tabloo.show(testGroupDF)
    return testGroupDF 
def getProbabilityTransitionsMatchOne(df,dim):
    assert(len(df[df['modelSize']==dim]['P'].to_list())==1)
    return np.array(json.loads(df[df['modelSize']==dim]['P'].to_list()[0]))
def getPerturbProbabilityTransitionsMatchOne(df,dim):
    assert(len(df[df['modelSize']==dim]['P_GA-DE'].to_list())==1)
    return np.array(json.loads(df[df['modelSize']==dim]['P_GA-DE'].to_list()[0]))
def getRefineProbabilityTransitionsMatchOne(df,dim):
    assert(len(df[df['modelSize']==dim]['P_GD-LBFGS'].to_list())==1)
    return np.array(json.loads(df[df['modelSize']==dim]['P_GD-LBFGS'].to_list()[0]))
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
    testGroupDF=createCategoryTransitionHeatMapsAt(SA_GA_DE_GD_LBFGS_RECORDS_PATH+"/records.json",baselevelIterations=baselevelIterations)
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

def createOperatorTransitionProbabilityHeatMap():
    baselevelIterations=100
    testGroupDF=createOperatorTransitionHeatMapsAt(SA_GA_DE_GD_LBFGS_RECORDS_PATH+"/records.json",baselevelIterations=baselevelIterations)
    # rastrigin=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('problemName',['PROBLEM_RASTRIGIN'])])]                                    
    rosenbrock=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('problemName',['PROBLEM_ROSENBROCK'])])]                                  
    # qing=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('problemName',['PROBLEM_QING'])])]                                    
    # trid=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('problemName',['PROBLEM_TRID'])])]                                    
    # schwefel=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('problemName',['PROBLEM_SCHWEFEL223'])])]                                    
    styblinskitang=testGroupDF[selectAllMatchAtLeastOne(testGroupDF,[('problemName',['PROBLEM_STYBLINSKITANG'])])]                                  
    Ps=[]
    xticks=[]
    yticks=[]
    perturbStates=['GA','DE']
    refineStates=['GD','LBFGS']
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
            plotRow.append(getPerturbProbabilityTransitionsMatchOne(df,dim))
            rowxTicks.append(perturbStates)
            rowyTicks.append(perturbStates)
            rowTitles.append('Perturb transitions')
            xlabelTitleRow.append('Next operator')
            ylabelTitleRow.append('Current operator')
            plotRow.append(getRefineProbabilityTransitionsMatchOne(df,dim))
            rowxTicks.append(refineStates)
            rowyTicks.append(refineStates)
            rowTitles.append('Refine transitions')
            figureTitleRow.append(f"{problem} {dim}")   
            xlabelTitleRow.append('Next operator')
            ylabelTitleRow.append('Current operator') 
        Ps.append(plotRow)
        xticks.append(rowxTicks)
        yticks.append(rowyTicks)
        titles.append(rowTitles)
        figuretitles.append(figureTitleRow)
        xlabelTitles.append(xlabelTitleRow)
        ylabelTitles.append(ylabelTitleRow)

    width_ratios=[1,1]
    height_ratios=[1]
    rowsize=3
    plotHeatmap(Ps,len(problems),len(dims)*2,xticks,yticks,titles,xlabelTitles,ylabelTitles,figuretitles,
                        width_ratios=width_ratios,height_ratios=height_ratios,subfigdim=(len(problems),len(dims)) ,figsize=(17,len(problems)*rowsize),
                        filename=f"plots/P_{problems}_{dims}_{baselevelIterations}-operatorP.svg",color='Blues')

# methodsComparison(['PROBLEM_ROSENBROCK'],[10,15,30,50,100,500,750], False)
# methodsComparison(['PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING'],[10,15,30,50,100,500,750],False)
# methodsComparison(['PROBLEM_TRID'],[10,15,30,50,100,500,750],False)
# methodsComparison(['PROBLEM_SCHWEFEL223'],[10,15,30,50,100,500,750],False)
# methodsComparison(['PROBLEM_ROSENBROCK'],[1,2,3,4,5,6,7,8,9], False)
# methodsComparison(['PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING'],[1,2,3,4,5,6,7,8,9],False)
# methodsComparison(['PROBLEM_TRID'],[1,2,3,4,5,6,7,8,9],False)
# methodsComparison(['PROBLEM_SCHWEFEL223'],[1,2,3,4,5,6,7,8,9],True)

# # Non-separable
# methodsComparison(['PROBLEM_ROSENBROCK','PROBLEM_TRID'],[10,15,30,50,100,500,750], False)
# methodsComparison(['PROBLEM_ROSENBROCK','PROBLEM_TRID'],[1,2,3,4,5,6,7,8,9], False)
# # Separable
# methodsComparison(['PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_SCHWEFEL223'],[10,15,30,50,100,500,750],False)
# methodsComparison(['PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_SCHWEFEL223'],[1,2,3,4,5,6,7,8,9],True)

# # Convex-unimodal
# methodsComparison(['PROBLEM_SCHWEFEL223','PROBLEM_TRID'],[10,15,30,50,100,500,750], False)
# methodsComparison(['PROBLEM_SCHWEFEL223','PROBLEM_TRID'],[1,2,3,4,5,6,7,8,9], False)
# # Nonconvex-multimodal
# methodsComparison(['PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_ROSENBROCK'],[10,15,30,50,100,500,750],False)
# methodsComparison(['PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_ROSENBROCK'],[1,2,3,4,5,6,7,8,9],True)

# # Convex-unimodal
# methodsComparison(['PROBLEM_SCHWEFEL223','PROBLEM_TRID'],[30,50,100,500,750], False)
# methodsComparison(['PROBLEM_SCHWEFEL223','PROBLEM_TRID'],[1,2,3,4,5,6], False)
# # Nonconvex-multimodal
# methodsComparison(['PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_ROSENBROCK'],[30,50,100,500,750],False)
# methodsComparison(['PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_ROSENBROCK'],[1,2,3,4,5,6],True)

# methodsComparison(['PROBLEM_SCHWEFEL223','PROBLEM_TRID','PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_ROSENBROCK'],[30,50,100,500,750], False)
# methodsComparison(['PROBLEM_SCHWEFEL223','PROBLEM_TRID','PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_ROSENBROCK'],[1,2,3,4,5,6], False)

# methodsComparison(['PROBLEM_SCHWEFEL223','PROBLEM_TRID','PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_ROSENBROCK'],[1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750], False)

# all5000IterationResults()
# createTransitionProbabilityHeatMap()
# createOperatorTransitionProbabilityHeatMap()
# createMethodsCostEvolutionPlots()
