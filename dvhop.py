from oct2py import octave
from dvhopConverter import convertCSVtoSNLP
import os
import numpy as np
import runConfig

PWD=os.getcwd()
DVHOPPATH=f"{PWD}/SNLP2D/problems/dvhopbench"
DVHOPWSNLOCALPATH=f"{DVHOPPATH}/WSN-localization"
DVHOPLOCALERRPATH=f"{DVHOPWSNLOCALPATH}/Localization Error"
octave.eval('pkg load statistics')

def setupProblem():
    octave.eval(f"cd {DVHOPWSNLOCALPATH}")
    octave.eval("runWSN();")

def currentError():
    octave.eval(f"cd {DVHOPWSNLOCALPATH}")
    octave.eval("cd 'Localization Error'")
    return octave.eval("calculate_localization_error;")

def copyProblemToCSVs():
    octave.eval(f"cd {DVHOPWSNLOCALPATH}")
    octave.eval(f"load './Deploy Nodes/coordinates.mat'")
    octave.eval(f"load './Topology Of WSN/neighbor.mat'")
    return (octave.pull('neighbor_matrix'),octave.pull('neighbor_dist'),octave.pull('all_nodes'))

def saveToSNLPCSV(neighbors,distances,all_nodes):
    np.savetxt(f"{DVHOPPATH}/exportedanchorflags.csv", all_nodes['anc_flag'].astype(int), fmt='%i', delimiter=",")
    np.savetxt(f"{DVHOPPATH}/exportedanchors.csv", all_nodes['estimated'], fmt='%.20f',delimiter=",")
    np.savetxt(f"{DVHOPPATH}/exporteddistances.csv", distances,  fmt='%.20f',delimiter=",")
    np.savetxt(f"{DVHOPPATH}/exportedneighbors.csv", neighbors.astype(int), fmt='%i', delimiter=",")

def exportToSNLP():
    (neighbors,distances,all_nodes)=copyProblemToCSVs()
    saveToSNLPCSV(neighbors,distances,all_nodes)
    anchorCsv=f'{DVHOPPATH}/exportedanchors.csv'
    anchorFlagsCsv=f'{DVHOPPATH}/exportedanchorflags.csv'
    neighborsCsv=f'{DVHOPPATH}/exportedneighbors.csv'
    distancesCsv=f'{DVHOPPATH}/exporteddistances.csv'
    convertCSVtoSNLP(anchorCsv,anchorFlagsCsv,neighborsCsv,distancesCsv, DVHOPPATH)

def evaluateLocalizationError(bestModelCsvPath, verbose=False):
    octave.eval(f"cd '{DVHOPLOCALERRPATH}'")
    octave.eval(f"load 'result.mat'")
    all_nodes=octave.pull('all_nodes')
    if verbose:
        print(all_nodes['estimated'])

    bestModel=np.loadtxt(bestModelCsvPath)
    bestModel=bestModel.reshape((bestModel.shape[0]//all_nodes['estimated'].shape[1], all_nodes['estimated'].shape[1]))
    if verbose:
        print(f"hybrid model: {bestModel}")
    anchorsEncountered=0
    for i in range(all_nodes['anc_flag'].shape[0]):
        if all_nodes['anc_flag'][i] != 1:
            all_nodes['estimated'][i]=bestModel[i-anchorsEncountered]
        else:
            anchorsEncountered+=1
    if verbose:
        print(all_nodes['estimated'])
    octave.push('all_nodes',all_nodes)
    octave.eval(f"save 'result.mat' all_nodes comm_r")


def runHybirdDESolver():
    popsize=400
    DEGen=10
    totalIterations=20000
    iterations=totalIterations//(DEGen+1)
    nodecount=27
    maxDist=400
    box=100
    solver="OPTIMIZER_MIN_DE"
    problemPath=DVHOPPATH
    problemName="exportedDVHop.snlp"                  
    anchorName="exportedDVHop.snlpa"
    runConfig.hybridDE(popsize,DEGen,totalIterations,iterations,nodecount,maxDist,box,problemPath,problemName,anchorName,solver)

setupProblem() # -> coordinates.mat, neighbor.mat, result.mat
print(f'DVHop error: {currentError()}')# result.mat -> error
exportToSNLP() # coordinates.mat, neighbor.mat, result.mat -> exportedDVHop.snlp,exportedDVHop.snlpa
runHybirdDESolver() # exportedDVHop.snlp,exportedDVHop.snlpa -> finalModelLBFGS.csv, finalModelGD.csv
evaluateLocalizationError("finalModelGD.csv")# finalModelGD.csv -> result.mat
print(f'HybridDE GD error: {currentError()}')# result.mat -> error
evaluateLocalizationError("finalModelLBFGS.csv")# finalModelLBFGS.csv -> result.mat
print(f'HybridDE LBFGS error: {currentError()}')# result.mat -> error