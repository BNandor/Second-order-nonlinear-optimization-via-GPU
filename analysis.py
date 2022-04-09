from cv2 import GFTTDetector
import testlogs
import pandas as pd 

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
print(joinedlogs.head(10))
print(joinedlogs[joinedlogs["final f: _gd"] > joinedlogs["final f: _lbfgs"]].count())