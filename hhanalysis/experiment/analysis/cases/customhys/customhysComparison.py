import sys
import os

sys.path.insert(0, '../..')
from commonAnalysis import *
from importData import *
from dict_hash import sha256
from commonPlots import *
from common import *
from loadCustomHYS import *
import tabloo
import numpy as np

def getGroups(recordsPath):
    controlGroup=getCustomHySControlGroupDF()
    testGroupDF=createTestGroupView(recordsPath,
                                    (filterMetricPropertiesMinMedIQR,"hashSHA256"),
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

def SAAnalysis():
    controlGroup,testGroupDF=getGroups(SA_EXPERIMENT_RECORDS_PATH)
    independentColumns=["baselevelIterations","modelSize","problemName"]
    comparisonDF=getComparison(controlGroup,testGroupDF,independentColumns)
    # print(comparisonDF)
    
    comparisonDF=comparisonDF.sort_values(by=["problemName",'modelSize','baselevelIterations'])
    createAndSaveContext(comparisonDF,independentColumns)
    comparisonDF=comparisonDF.drop(['HHBetter'],axis=1)
    print(comparisonDF.to_latex(index=False))

SAAnalysis()