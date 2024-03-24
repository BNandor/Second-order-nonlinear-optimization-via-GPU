import sys
sys.path.insert(0, '../../..')
sys.path.insert(1, '..')
from commonSPRT import *
from commonAnalysis import *
from importData import *
from loadHHData import *
import pandas as pd
import json
import math
import matplotlib.pyplot as plt

# Define constants
# ALPHA = 0.10  # Type I error rate (probability of  accepting worse mean )
# BETA = 0.10   # Type II error rate (probability of rejecting  better mean )
ALPHA = 0.01  # Type I error rate (probability of  accepting worse mean )
BETA = 0.01  # Type II error rate (probability of rejecting  better mean )

def histPlotOf(data):
    plt.hist(data, bins=30, edgecolor='black')  # Adjust bins as needed
    plt.title('Performnce histogram')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
# Define function for sequential probability ratio test
def sequential_probability_ratio_test(samples1,samples2,cohensD):
    n = len(samples2)
    for i in range(2,n):
        hypthesisTest=checkIfNewMeanIsLessThanComparisonSequentialT(samples1,samples2[0:i],ALPHA,BETA,cohensD)
        # hypthesisTest=checkIfNewMeanIsDifferentThanComparisonSequentialT(samples1,samples2[0:i],ALPHA,BETA,cohensD)
        if hypthesisTest == 0:
            print(f"New sample mean >= previous mean")
            return (i,0) 
        if hypthesisTest == 1:
            print(f"New sample mean < previous mean")
            return (i,1)         
    return (n,-1)  # If stopping condition not met, return total sample size

def testTwoPopulations():
    benchmark=[
                        0.5429738176056949,
                        0.3904087936644329,
                        0.21738261697429684,
                        0.7569152672365734,
                        0.6636842344448614,
                        0.7550485788764545,
                        0.5572621572189366,
                        0.656315050019201,
                        0.7421065266245553,
                        0.6548644034711302,
                        0.960015361080622,
                        0.8938982702092609,
                        0.8043638477981409,
                        0.9215228620611345,
                        0.9788746814823708,
                        1.110288971504063,
                        1.1361687223804333,
                        1.176900258412499,
                        3.439007726178548,
                        2.094269584231863,
                        1.6660836115058837,
                        1.5318739065460598,
                        1.5763820571198623,
                        1.3654827727044447,
                        7.012946432498922,
                        1.3954021098573262,
                        3.452347147673346,
                        1.368717718764665,
                        1.358432604932464,
                        2.030839606466009
                        ]
    test=[    0.008729960036173812,
                        0.00020766319696338676,
                        0.004014640007557633,
                        0.001323749573656016,
                        0.0017283612201699405,
                        0.005094135291096225,
                        0.0010922455482282095,
                        0.00199741939360402,
                        0.01086300864706505,
                        0.009573463882908433,
                        0.017688524698039092,
                        0.012988943793564077,
                        0.013229151657803433,
                        0.012288545657040762,
                        0.011949827815615828,
                        0.020018842608720392,
                        0.12023130036430216,
                        0.08266486518023937,
                        0.0453900351125959,
                        0.04856147239154726,
                        0.09021634972700882,
                        0.028179348288607985,
                        0.022688801834523452,
                        0.7139110361939422,
                        0.349356310133345,
                        0.1363435588219593,
                        0.7013776144273298,
                        0.2573545138809953,
                        0.5347734288454773,
                        0.17576049728174403
                        ]
    histPlotOf(benchmark)
    histPlotOf(test)
    # sample_size = sequential_probability_ratio_test(benchmark, test_mean, test_sigma, b_sigma)
    (sample_size,hyp) = sequential_probability_ratio_test( benchmark,test,0.5)
    print("Sample size required:", sample_size)

def sprtCheckTrials(trials):
    trialCount=len(trials)
    if trialCount<1:
        print("trial size not  large enough for SPRT")
        return  False
    
    best_perf=trials[0]['performanceSamples']
    best_mean=np.average(best_perf)
    # histPlotOf(best_perf)
    totalSamplesNeeded=len(best_perf)
    for i in range(1,trialCount):
        (sample_size,hyp) = sequential_probability_ratio_test(best_perf, trials[i]['performanceSamples'], 1)
        totalSamplesNeeded+=sample_size
        
        avg=np.average(trials[i]['performanceSamples'][0:sample_size])
        print(f"best: {best_perf} -> {best_mean}")
        print(f"new: {trials[i]['performanceSamples'][0:sample_size]} -> {avg}")
        print("Sample size required:", sample_size)
        if hyp == 0:
           continue
        avg=np.average(trials[i]['performanceSamples'])#avg
        totalSamplesNeeded+=len(best_perf)-sample_size
        if avg<best_mean:
            best_mean=avg
            # best_perf=trials[i]['performanceSamples'][0:sample_size]
            best_perf=trials[i]['performanceSamples']
            
            # histPlotOf(best_perf)
    print(f"total samples needed: {totalSamplesNeeded}")
    print(f"total samples saved: {1-totalSamplesNeeded/(len(trials[0]['performanceSamples'])*trialCount)}")
    return (best_perf,best_mean)
def testHH(methodExperiments,problems,dimensions):
    metadata={
            # "minMetricColumn":'minAvg',
            # "metricsAggregation":{'minAvg':'min'},
            # "mergeOn":mergeOnAvg,
            "minMetricColumn":'minMedIQR',
            "metricsAggregation":{'minMedIQR':'min'},
            "mergeOn":mergeOnMinMedIQR,
            'optimizers':list(),
            "saveMetadataColumns":["minMetricColumn",'optimizers','baselevelIterations'],
            "baselevelIterations": [100],
            "problems":problems,
            "modelSize":dimensions,
            "datasets":{}
        }
    hhdata=loadDataMap()
    all=pd.concat([hhdata[methodAndExperiment[0]](metadata,methodAndExperiment[1]) for methodAndExperiment in methodExperiments])
    all=all[selectAllMatchAtLeastOne(all,[('baselevelIterations',metadata["baselevelIterations"]),('modelSize',metadata["modelSize"]),('problemName',metadata["problems"])])]
    all=all.sort_values(by=['modelSize',"problemName",metadata["minMetricColumn"]])
    for index,row in all.iterrows():
        print(f'{row["hyperLevel-id"]}-{row["modelSize"]}-{row["problemName"]}')
        print(sprtCheckTrials(json.loads(row['trials'])))
        
# testTwoPopulations()  
    
allmethods=[('nmhh2Perf','/')]
initialproblems=['PROBLEM_SCHWEFEL223','PROBLEM_TRID','PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_ROSENBROCK']
extraProblems=['PROBLEM_MICHALEWICZ']
            #    'PROBLEM_DIXONPRICE','PROBLEM_LEVY','PROBLEM_SCHWEFEL','PROBLEM_SUMSQUARES','PROBLEM_SPHERE']
# extraProblems=['PROBLEM_MICHALEWICZ',
#                'PROBLEM_DIXONPRICE','PROBLEM_LEVY','PROBLEM_SCHWEFEL','PROBLEM_SUMSQUARES','PROBLEM_SPHERE']
allproblems=initialproblems+extraProblems
# alldimensions=[5,6,7,8,9,10,15,30,50,100,500,750]
alldimensions=[5]
testHH(allmethods,['PROBLEM_RASTRIGIN'],alldimensions)