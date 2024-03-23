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
ALPHA = 0.05  # Type I error rate (probability of  accepting worse mean )
BETA = 0.01   # Type II error rate (probability of rejecting  better mean )

def histPlotOf(data):
    plt.hist(data, bins=30, edgecolor='black')  # Adjust bins as needed
    plt.title('Performnce histogram')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
# Define function for sequential probability ratio test
def sequential_probability_ratio_test(samples, mu_benchmark, sigma_benchmark, sigma_algorithm):
    n = len(samples)
    S = 0
    for i in range(n):
        (hypthesisTest,newS)=checkHypothesisThatNewMeanIsLessThanComparison(S,samples[i], mu_benchmark, sigma_benchmark, sigma_algorithm,ALPHA,BETA)
        S=newS
        if hypthesisTest == 0:
            print(f"New sample mean >= previous mean")
            return i + 1  
        if hypthesisTest == 1:
            print(f"New sample mean < previous mean")
            return i + 1         
    return n  # If stopping condition not met, return total sample size

def testTwoPopulations():
    benchmark=[
                        0.02049679057950057,
                        0.004445760631146774,
                        0.015378615726532333,
                        0.00452613517780351,
                        0.0019975773880870564,
                        0.012346161594132864,
                        0.018611775794284725,
                        2.6642438113280717e-05,
                        0.000710765687912358,
                        0.004585284223475624,
                        0.0069053002126033295,
                        0.010096730542697144,
                        0.003656441971195858,
                        0.003545159077037895,
                        0.022287949791108548,
                        0.024081238645433597,
                        0.03940339933143971,
                        0.04888119102887783,
                        0.13568593557152786,
                        0.05227062633783166,
                        0.0334306735867592,
                        0.09718171756840081,
                        0.10025155209611493,
                        0.2106642812769366,
                        0.2397424495127332,
                        0.507279574418081,
                        0.4959552855093482,
                        0.5491868225740084,
                        0.6502495699901902,
                        0.6847592020864028
                        ]
    test=[    0.018219730024854452,
                        0.0002102307253720227,
                        0.013993001414741456,
                        0.006150925765575561,
                        9.489123401082855e-06,
                        0.004393089819820088,
                        0.006922995672583384,
                        0.0024206701079522194,
                        0.0012213809499506331,
                        0.002935869803999426,
                        0.001559965344500364,
                        0.005794526087775937,
                        0.01298023761969837,
                        0.014696163870423393,
                        0.024999410693453187,
                        0.026967928902512664,
                        0.03343290753119628,
                        0.03475700134870132,
                        0.034430675474898774,
                        0.02739433237373591,
                        0.04677695934837057,
                        0.047757620328153595,
                        0.04871301453374981,
                        0.05226013445595956,
                        0.0697272468387018,
                        0.1531188193648238,
                        0.04913809414032108,
                        0.14491433312288043,
                        0.04956957300408982,
                        0.08468691244999399
                        ]

    b_mean=np.average(benchmark)
    print(f"benchmark mean: {b_mean}")
    b_sigma=np.std(benchmark)
    print(f"benchmark std: {b_mean}")

    test_mean=np.average(test)
    print(f"test mean: {test_mean}")
    test_sigma=np.std(test)
    print(f"test std: {test_sigma}")

    # sample_size = sequential_probability_ratio_test(benchmark, test_mean, test_sigma, b_sigma)
    sample_size = sequential_probability_ratio_test(test, b_mean, b_sigma, b_sigma)
    print("Sample size required:", sample_size)

def sprtCheckTrials(trials):
    trialCount=len(trials)
    if trialCount<10:
        print("trial size not  large enough for SPRT")
        return  False
    best_mean=math.inf
    best_sigma=None
    average_sigma=np.average([ np.std(perf) for perf in  list(map(lambda t:t['performanceSamples'],trials[0:10]))])

    for i in range(10):
        avg=np.average(trials[i]['performanceSamples'])
        if avg<best_mean:
            best_mean=avg
            best_sigma=np.std(trials[i]['performanceSamples'])
    
    for i in range(10,trialCount-10):
        sample_size = sequential_probability_ratio_test(trials[i]['performanceSamples'], best_mean, best_sigma, np.std(trials[i]['performanceSamples'][0:20]))
        # sample_size = sequential_probability_ratio_test(test, b_mean, b_sigma, test_sigma)
        print("Sample size required:", max(20,sample_size))
        avg=np.average(trials[i]['performanceSamples'][0:sample_size])
        if avg<best_mean:
            best_mean=avg
            best_sigma=np.std(trials[i]['performanceSamples'][0:max(20,sample_size)])
        histPlotOf(trials[i]['performanceSamples'])

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
        sprtCheckTrials(json.loads(row['trials']))
        
# testTwoPopulations()  
    
allmethods=[('nmhh2Perf','/')]
initialproblems=['PROBLEM_SCHWEFEL223','PROBLEM_TRID','PROBLEM_RASTRIGIN','PROBLEM_STYBLINSKITANG','PROBLEM_QING','PROBLEM_ROSENBROCK']
extraProblems=['PROBLEM_MICHALEWICZ']
            #    'PROBLEM_DIXONPRICE','PROBLEM_LEVY','PROBLEM_SCHWEFEL','PROBLEM_SUMSQUARES','PROBLEM_SPHERE']
# extraProblems=['PROBLEM_MICHALEWICZ',
#                'PROBLEM_DIXONPRICE','PROBLEM_LEVY','PROBLEM_SCHWEFEL','PROBLEM_SUMSQUARES','PROBLEM_SPHERE']
allproblems=initialproblems+extraProblems
# alldimensions=[5,6,7,8,9,10,15,30,50,100,500,750]
alldimensions=[50]
testHH(allmethods,['PROBLEM_RASTRIGIN'],alldimensions)