from cv2 import GFTTDetector
import testlogs
import pandas as pd 
# {'objects': ('King Arthur', 'Sir Robin', 'holy grail'), 'properties': ('human', 'knight', 'king', 'mysterious'), 'context': [(0, 1, 2), (0, 1), (3,)]}
def mapResultRowToAttributes(row):
    if row["final f: _gd"] > row["final f: _lbfgs"]:
        return ['lbfgsbetter']
    else:
        return ['gd']

def mapToFormalConcept(resultingDF):
    return resultingDF.apply(mapResultRowToAttributes, axis=1, result_type='expand')
testCaseColumns=['solver','problem','nodecount','edges','anchors','totaliterations','population','deIteration','distFraction','testcase']

GDCases = testlogs.readLogs("./SNLP3D/problems/gridtest/csv/3D/GD/metrics/metrics-3D-random-problem-1-sample2.csv")
LBFGSCases = testlogs.readLogs("./SNLP3D/problems/gridtest/csv/3D/LBFGS/metrics/metrics-3D-random-problem-1-sample2.csv")
columnTypes={'final f: ': 'double','nodecount': 'double','edges': 'double','anchors': 'double','totaliterations': 'double','population': 'double','deIteration': 'double','distFraction': 'double'}
lbfgsDF = pd.DataFrame(LBFGSCases).astype(columnTypes).drop_duplicates(subset=testCaseColumns, keep='first')
gdDF = pd.DataFrame(GDCases).astype(columnTypes).drop_duplicates(subset=testCaseColumns, keep='first')
lbfgsDF["optimizer"]="LBFGS"
gdDF["optimizer"]="GD"

print(gdDF) 
print(lbfgsDF) 
joinedlogs=lbfgsDF.set_index(testCaseColumns).join(gdDF.set_index(testCaseColumns),lsuffix='_lbfgs', rsuffix='_gd')
resultContext=mapToFormalConcept(joinedlogs)
resultContext.reset_index(inplace=True)
# resultContext=resultContext.to_dict('index')
print(joinedlogs) 
print(resultContext)

successCount=joinedlogs.query(" `final f: _gd` >= `final f: _lbfgs`").count()['final f: _lbfgs']
total=joinedlogs.count()['final f: _lbfgs']
print('Percentage where LBFGS is better than GD')
print(successCount / total)

simpleSuccessPerDistFrac=joinedlogs.query('deIteration == 0 and `final f: _gd` >= `final f: _lbfgs`').groupby(['distFraction']).count()['final f: _gd']
simpleTotalsPerDistFrac=joinedlogs.query('deIteration == 0 ').groupby(['distFraction']).count()['final f: _gd']
deSuccessPerDistFrac=joinedlogs.query('deIteration > 0 and `final f: _gd` >= `final f: _lbfgs`').groupby(['distFraction']).count()['final f: _gd']
deTotalsPerDistFrac=joinedlogs.query('deIteration > 0').groupby(['distFraction']).count()['final f: _gd']
print('No DE: Percentage where LBFGS is better than GD per distFraction ')
print(simpleSuccessPerDistFrac/simpleTotalsPerDistFrac)
print('DE: Percentage where LBFGS is better than GD per distFraction ')
print(deSuccessPerDistFrac/deTotalsPerDistFrac)


