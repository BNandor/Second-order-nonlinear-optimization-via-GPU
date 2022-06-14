from inspect import indentsize
from turtle import left, right
from cv2 import GFTTDetector
import testlogs
import pandas as pd 
import concepts
import matplotlib.pyplot as mp
import seaborn as sns
import scipy.stats as stats

# {'objects': ['King Arthur', 'Sir Robin', 'holy grail'], 'properties': ['human', 'knight', 'king', 'mysterious'], 'context': [[0, 1, 2], [0, 1], [3]]}

def saveContext(joinedlogs,contextName):
    resultContext=joinedlogs.copy()
    resultContext.reset_index(inplace=True)
    resultContext=resultContext.to_dict('index')
    ctx=mapToFormalConcept(resultContext)
    # ctx.lattice.graphviz(view=True)
    ctx.tofile(contextName)

def mapToFormalConcept(resultsDic):
    populationSizes={1:"singlePop",5:"mediumPop",20:"maxPop"}
    DEiterations={0:"noDE",1:"smallDE",4:"mediumDE",19:"bigDE",100:"DE_APPROACH"}
    minimizerIterations={100:"smallestIteration",1000:"smallIteration",5000:"mediumIteration",50000:"maxIteration"}
    nodecounts={10:"smallModel",100:"mediumModel",1000:"largeModel"}
    solvermethods=["OPTIMIZER_MIN_DE","OPTIMIZER_SIMPLE_DE","OPTIMIZER_MIN_INIT_DE"]
    maxDistanceAsBoxFractions={0.1:'smallEdges',0.5:'bigEdges'}
    testCount=5
    # if row["final f: _gd"] > row["final f: _lbfgs"]:
    #     return ['lbfgsbetter']
    # else:
    #     return ['gd']
    properties=[solvermethods[0],
                solvermethods[1],
                solvermethods[2],
                '3DRandom1',
                nodecounts[10],
                nodecounts[100],
                nodecounts[1000],
                minimizerIterations[100],
                minimizerIterations[1000],
                minimizerIterations[5000],
                minimizerIterations[50000],
                populationSizes[1],
                populationSizes[5],
                populationSizes[20],
                DEiterations[0],
                DEiterations[1],
                DEiterations[4],
                DEiterations[19],
                DEiterations[100],
                maxDistanceAsBoxFractions[0.1],
                maxDistanceAsBoxFractions[0.5],
                "LBFGSWin",
                "GDWin"]
    
    propertiesInverse={
        "solver":lambda solver:properties.index(solver),
        "problem":lambda problem:properties.index(problem),
        "nodecount":lambda modelsize:properties.index(nodecounts[modelsize]),
        "totaliterations":lambda it:properties.index(minimizerIterations[it]),
        "population":lambda pop:properties.index(populationSizes[pop]),
        "deIteration":lambda it:  properties.index(DEiterations[it]) if it < 20  else properties.index(DEiterations[100]),
        "distFraction":lambda frac:properties.index(maxDistanceAsBoxFractions[frac]),
        "optimizer":lambda row:properties.index("LBFGSWin") if row["final f: _gd"] > row["final f: _lbfgs"] else properties.index("GDWin")
    }

    ctx={'objects':list(map(lambda key:str(key),resultsDic.keys())),'properties':properties,'context':[]}
    for row in resultsDic.keys():
        attributes=[propertiesInverse["solver"](resultsDic[row]['solver']),
                    propertiesInverse["problem"](resultsDic[row]['problem']),
                    propertiesInverse["nodecount"](resultsDic[row]['nodecount']),
                    propertiesInverse["totaliterations"](resultsDic[row]['totaliterations']),
                    propertiesInverse["population"](resultsDic[row]['population']),
                    propertiesInverse["deIteration"](resultsDic[row]['deIteration']),
                    propertiesInverse["distFraction"](resultsDic[row]['distFraction']),
                    propertiesInverse["optimizer"](resultsDic[row])]
        ctx['context'].append(attributes)
    print(ctx)
    return concepts.Context.fromdict(ctx)
testCaseColumns=['solver','problem','nodecount','edges','anchors','totaliterations','population','deIteration','distFraction','testcase']
combinationCaseColumns=['solver','problem','nodecount','edges','anchors','totaliterations','population','deIteration','distFraction','optimizer']

GDCases = testlogs.readLogs("/home/spaceman/dissertation/finmat/ParallelLBFGS/SNLP3D/problems/gridtest/csv/3D/GD/metrics/metrics-3D-random-problem-1-sample2.csv")
LBFGSCases = testlogs.readLogs("/home/spaceman/dissertation/finmat/ParallelLBFGS/SNLP3D/problems/gridtest/csv/3D/LBFGS/metrics/metrics-3D-random-problem-1-sample2.csv")
columnTypes={'final f: ': 'double','nodecount': 'double','edges': 'double','anchors': 'double','totaliterations': 'double','population': 'double','deIteration': 'double','distFraction': 'double'}
lbfgsDF = pd.DataFrame(LBFGSCases).astype(columnTypes).drop_duplicates(subset=testCaseColumns, keep='first')#.head(100)
gdDF = pd.DataFrame(GDCases).astype(columnTypes).drop_duplicates(subset=testCaseColumns, keep='first')#.head(100)
lbfgsDF["optimizer"]="LBFGS"
gdDF["optimizer"]="GD"

# print(gdDF) 
# print(lbfgsDF) 
# joinedlogs=lbfgsDF.set_index(testCaseColumns).join(gdDF.set_index(testCaseColumns),lsuffix='_lbfgs', rsuffix='_gd')
# print(joinedlogs) 

# # successCount=joinedlogs.query(" `final f: _gd` >= `final f: _lbfgs`").count()['final f: _lbfgs']
# # total=joinedlogs.count()['final f: _lbfgs']
# # print('Percentage where LBFGS is better than GD')
# # print(successCount / total)

# # simpleSuccessPerDistFrac=joinedlogs.query('deIteration == 0 and `final f: _gd` >= `final f: _lbfgs`').groupby(['distFraction']).count()['final f: _gd']
# # simpleTotalsPerDistFrac=joinedlogs.query('deIteration == 0 ').groupby(['distFraction']).count()['final f: _gd']
# # deSuccessPerDistFrac=joinedlogs.query('deIteration > 0 and `final f: _gd` >= `final f: _lbfgs`').groupby(['distFraction']).count()['final f: _gd']
# # deTotalsPerDistFrac=joinedlogs.query('deIteration > 0').groupby(['distFraction']).count()['final f: _gd']
# # print('No DE: Percentage where LBFGS is better than GD per distFraction ')
# # print(simpleSuccessPerDistFrac/simpleTotalsPerDistFrac)
# # print('DE: Percentage where LBFGS is better than GD per distFraction ')
# # print(deSuccessPerDistFrac/deTotalsPerDistFrac)

# minimumsPerProblem=joinedlogs.loc[joinedlogs.groupby(["nodecount","distFraction"])["final f: _lbfgs"].idxmin()]
# # minimumsPerProblemJoined=minimumsPerProblem.join(joinedlogs)
# print(minimumsPerProblem.to_csv('minLBFGS10-MIN_INIT_DE.csv'))
# saveContext(joinedlogs,"biggestContext.ctx")
all=pd.concat([lbfgsDF,gdDF])
# print(all)
print(all.groupby(combinationCaseColumns, as_index=False))
# allAveragedByTestcase=all.groupby(combinationCaseColumns).mean()
allSTDByTestcase=all.groupby(combinationCaseColumns, as_index=False).agg(
                      {'final f: ':['mean','std']})
allSTDByTestcase[('final f: ', 'std')]=allSTDByTestcase[('final f: ', 'std')]/allSTDByTestcase[('final f: ', 'mean')]
allSTDByTestcase.columns=allSTDByTestcase.columns.to_flat_index()

# allSTDByTestcase=all.groupby(combinationCaseColumns).std()
# print(allAveragedByTestcase)
# print(allSTDByTestcase)
# orderedAllByFCost=allAveragedByTestcase.sort_values(['final f: ','nodecount','distFraction'],ascending=True).groupby(['nodecount','distFraction']).head(10)
# orderedAllByFCost=allSTDByTestcase.sort_values([('final f: ', 'mean'),('nodecount',''),('distFraction','')],ascending=True).groupby([('nodecount',''),('distFraction','')]).head(50)
# # print(orderedAllByFCost)
# orderedAllByFCost.to_csv('costOrderTestCaseSTD.csv')
# orderedAllByFCost[(orderedAllByFCost[('distFraction','')]==0.5) & (orderedAllByFCost[('nodecount','')]==1000)  ].plot( y=[('final f: ', 'std')], kind="bar", figsize=(9, 8),title='cost std',legend=False)
# orderedAllByFCost[(orderedAllByFCost[('distFraction','')]==0.1) & (orderedAllByFCost[('nodecount','')]==1000) ].plot( y=[('final f: ', 'std')], kind="bar", figsize=(9, 8),title='cost std',legend=False)

# orderedAllByFCost[(orderedAllByFCost[('distFraction','')]==0.5) & (orderedAllByFCost[('nodecount','')]==10) ].plot( y=[('final f: ', 'mean')], kind="bar", figsize=(9, 8),title='cost',legend=False)
# orderedAllByFCost[(orderedAllByFCost[('distFraction','')]==0.1) & (orderedAllByFCost[('nodecount','')]==10)  ].plot( y=[('final f: ', 'mean')], kind="bar", figsize=(9, 8),title='cost',legend=False)

mp.show()

## ANOVA
problem=1000
dist=0.1

bestA=allSTDByTestcase.loc[allSTDByTestcase[(allSTDByTestcase[('nodecount','')]==problem) & (allSTDByTestcase[('distFraction','')]==dist)][('final f: ', 'mean')].idxmin()]
print(bestA)
AtotalIterations=bestA[('totaliterations','')]
APop=bestA[('population','')]
ADeGen=bestA[('deIteration','')]
AOptimizer=bestA[('optimizer','')]
ASolver=bestA[('solver','')]

A=all[( all['nodecount']==problem)  \
            & (all['distFraction']==dist) \
            & (all['totaliterations']==AtotalIterations) \
            & (all['population']==APop) \
            & (all['deIteration']==ADeGen) \
            & (all['optimizer']==AOptimizer) \
            & (all['solver']==ASolver) ]
print(A['final f: '])
A['final f: '].to_csv(f'{problem}-{dist}-A.csv')
BtotalIterations=AtotalIterations
BPop=1
BDeGen=0
BOptimizer=AOptimizer
BSolver=ASolver

B=all[( all['nodecount']==problem)  \
            & (all['distFraction']==dist) \
            & (all['totaliterations']==BtotalIterations) \
            & (all['population']==BPop) \
            & (all['deIteration']==BDeGen) \
            & (all['optimizer']==BOptimizer) \
            & (all['solver']==BSolver) ]

print(B['final f: '])
B['final f: '].to_csv(f'{problem}-{dist}-B.csv')
AB=pd.DataFrame()
AB=A['final f: '].reset_index().join(B['final f: '].reset_index(),lsuffix='A', rsuffix='B')
print(AB)
stat = pd.melt(AB.reset_index(), id_vars=['index'], value_vars=['final f: A','final f: B'])
stat.columns=['index','configurations','f-cost']
ax = sns.boxplot(x='configurations', y='f-cost', data=stat, color='#99c2a2')
ax = sns.swarmplot(x="configurations", y="f-cost", data=stat, color='#7d0013')
# mp.show()

# stats f_oneway functions takes the groups as input and returns ANOVA F and p value

print(stats.f_oneway(AB['final f: A'], AB['final f: B']))