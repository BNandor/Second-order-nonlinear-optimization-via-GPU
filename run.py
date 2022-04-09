import subprocess

from numpy import append
import SNLP
import visualize
import re
import os
import generate
import csv
import json
import testlogs


def runOptimizerWith(flags):
    process = subprocess.Popen(['make', 'buildAndRun', f'NVCCFLAGS={flags}'],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
    return process.stdout

def consumeOutput(stdout,consumeline):
    logs=[]
    for linebuffer in stdout:
        line=linebuffer.decode("utf-8")
        logs.append(line)
        consumeline(line)
    return logs

def filterTransformWrite(lines,linecondition,transform,outPath,outName):
    if not os.path.exists(outPath):
        os.makedirs(f"{outPath}")
    outfile = open(f"{outPath}/{outName}", "w")
    [outfile.write(transform(line)) for line in lines if linecondition(line)]
    outfile.close()  

def appendColumnsTo(outPath,outName,columns):
    if not os.path.exists(outPath):
        os.makedirs(f"{outPath}")
    outfile = open(f"{outPath}/{outName}", 'a+', newline ='')
    outfile.write(columns+"\n")
    outfile.close()  

def findReplaceFirst(logs,find,replace,replacedWith):
    finds=[log for log in logs if re.search(find,log)]
    if len(finds) == 0:
        print(f"Cant find {find} from {len(logs)} lines")
    return finds[0].rstrip().replace(replace,replacedWith)

def appendMetrics(logs,metrics:SNLP.Metrics,metricsName):
    metricKeys=["time ms :","threads:","iterations:","final f: ","fevaluations: "]
    for metricKey in metricKeys:
        metrics.options[metricKey]=findReplaceFirst(logs,metricKey,metricKey,"")
    appendColumnsTo(outPath=metrics.path,outName=metricsName,columns=json.dumps(metrics.options))

def runSNLP(problem:SNLP.OptProblem,optionalFlags):
    print(f"Running: {problem.name} with {problem.optimizer}")
    currentProblem = SNLP.SNLP(problem)
    logs=consumeOutput(runOptimizerWith(currentProblem.flags( optionalFlags)),lambda line:print(line))
    filterTransformWrite(logs,
                     lambda line:re.search("xCurrent",line), 
                     lambda line: line.replace("xCurrent",""),
                     currentProblem.optproblem.outputPath,
                     SNLP.XHIST)
    filterTransformWrite(logs,
                     lambda line:re.search("^f:",line), 
                     lambda line: line.replace("f:",""),
                     currentProblem.optproblem.outputPath,
                     SNLP.FHIST)                  
    appendMetrics(logs,currentProblem.optproblem.metrics ,SNLP.METRICS)

def runSNLP3D(problemPath,problemName,anchorName,populationName,residualSizes,modelsize,framesize,optionalFlags,metricsConfig):
    problemLBFGS = SNLP.OptProblem(name="PROBLEM_SNLP3D", 
                        optimizer="LBFGS", 
                        inputPaths=[f"{problemPath}/{problemName}", 
                                    f"{problemPath}/{anchorName}",
                                    f"{problemPath}/{populationName}"],
                        outputPath=f"{problemPath}/csv/3D/LBFGS",
                        constantsSizes=residualSizes,
                        modelsize=modelsize,
                        framesize=framesize,
                        metrics=SNLP.Metrics( path=f"{problemPath}/csv/3D/LBFGS/metrics", options=metricsConfig))
    problemGD = SNLP.OptProblem(name="PROBLEM_SNLP3D", 
                        optimizer="GD", 
                        inputPaths=[f"{problemPath}/{problemName}", 
                                    f"{problemPath}/{anchorName}",
                                    f"{problemPath}/{populationName}"],
                        outputPath=f"{problemPath}/csv/3D/GD",
                        constantsSizes=residualSizes,
                        modelsize=modelsize,
                        framesize=framesize,
                        metrics=SNLP.Metrics( path=f"{problemPath}/csv/3D/GD/metrics", options=metricsConfig))
    problems = [problemLBFGS,problemGD]
    [runSNLP(problem=problem, optionalFlags=optionalFlags) for problem in problems]
    # fvisualizer = visualize.FVisualizer(problems)
    # fvisualizer.visualize()
    # xvisualizer = visualize.SNLP3DVisualizer(problems)
    # xvisualizer.visualize()

def runSNLP2D(problemPath,problemName,anchorName,populationName,residualSizes,modelsize,framesize,optionalFlags,metricsConfig):
    problemLBFGS = SNLP.OptProblem(name="PROBLEM_SNLP", 
                        optimizer="LBFGS", 
                        inputPaths=[f"{problemPath}/{problemName}", 
                                    f"{problemPath}/{anchorName}",
                                    f"{problemPath}/{populationName}"],
                        outputPath=f"{problemPath}/csv/2D/LBFGS",
                        constantsSizes=residualSizes,
                        modelsize=modelsize,
                        framesize=framesize,
                        metrics=SNLP.Metrics( path=f"{problemPath}/csv/2D/LBFGS/metrics", options=metricsConfig))
    problemGD = SNLP.OptProblem(name="PROBLEM_SNLP", 
                        optimizer="GD", 
                        inputPaths=[f"{problemPath}/{problemName}", 
                                    f"{problemPath}/{anchorName}",
                                    f"{problemPath}/{populationName}"],
                        outputPath=f"{problemPath}/csv/2D/GD",
                        constantsSizes=residualSizes,
                        modelsize=modelsize,
                        framesize=framesize,
                        metrics=SNLP.Metrics( path=f"{problemPath}/csv/2D/GD/metrics", options=metricsConfig))
    problems = [problemLBFGS,problemGD]
    [runSNLP(problem=problem,optionalFlags=optionalFlags) for problem in problems]
    # fvisualizer = visualize.FVisualizer(problems)
    # fvisualizer.visualize()
    # xvisualizer = visualize.SNLP2DVisualizer(problems)
    # xvisualizer.visualize()
 
populationSizes=[1,5,20]
DEiterations=[0,4,19]
minimizerIterations=[100,1000,5000,50000]
nodecounts=[10,100,1000]
solvermethods=["OPTIMIZER_MIN_DE"]
maxDistanceAsBoxFractions=[0.1,0.5]
testCount=5
ANCHOR_BOUNDING_BOX=1000
INITIAL_POP_BOUNDING_BOX=20*ANCHOR_BOUNDING_BOX

GDCases=testlogs.readCases("/home/spaceman/dissertation/finmat/ParallelLBFGS/SNLP3D/problems/gridtest/csv/3D/GD/metrics/metrics-3D-random-problem-1-sample2.csv")

for nodecount in nodecounts:
    for populationSize in populationSizes:
        for totalIterations in minimizerIterations:
            for deIteration in DEiterations:
                for solver in solvermethods:
                    for maxDistFraction in maxDistanceAsBoxFractions:
                        DEiteration=deIteration
                        if populationSize <4:
                            DEiteration=0
                        iterations=totalIterations/(deIteration+1)
                        diffEvolutionOptions=f"-DPOPULATION_SIZE={populationSize} -DDE_ITERATION_COUNT={DEiteration} -D{solver}"
                        iterationOptions=f"-DITERATION_COUNT={iterations}"
                        metricOptions=f"-DGLOBAL_SHARED_MEM {iterationOptions} {diffEvolutionOptions}"              
                        currentflags=f" {metricOptions} "

                        generator=generate.Generate3DRandomProblem1(nodecount=nodecount, 
                                                                        outPath="./SNLP3D/problems/gridtest",
                                                                        problemName=f"random2{nodecount}-{maxDistFraction}.snlp",
                                                                        anchorName=f"random2{nodecount}-{maxDistFraction}.snlpa")


                        problemSize=generator.generateSNLPProblem(int(ANCHOR_BOUNDING_BOX*maxDistFraction))
                        anchorSize=generator.generateSNLPProblemAnchors(ANCHOR_BOUNDING_BOX)

                        for testCase in range(testCount):
                            print(f"testcase {testCase}\n")
                            populationGenerator = generate.PopulationGenerator(populationSize,generator.modelsize(),generator.outPath,f"random2{nodecount}-{maxDistFraction}-{populationSize}-{testCase}.pop",boundingBox=INITIAL_POP_BOUNDING_BOX)
                            populationGenerator.generate()
                            testconfig={"solver":solver,
                                        "problem":generator.name(),
                                        "nodecount":nodecount,
                                        "edges":problemSize,
                                        "anchors":anchorSize,
                                        "totaliterations":totalIterations,
                                        "population":populationSize,
                                        "deIteration":DEiteration,
                                        "distFraction":maxDistFraction,
                                        "testcase":testCase}
                            if(testlogs.caseIdentifier(testconfig) in GDCases):
                                print(f"skipping {testconfig}")
                                continue
                            print(testconfig)
                            runSNLP3D(generator.outPath,
                                generator.problemName,
                                generator.anchorName,
                                populationGenerator.outName,
                                [problemSize,
                                anchorSize],
                                generator.modelsize(),
                                100,currentflags,testconfig)

                        # generator=generate.Generate2DStructuredProblem1(nodecount=20, 
                        #                                                 outPath="./SNLP2D/problems/gridtest",
                        #                                                 problemName="spiral.snlp",                     
                        #                                                 anchorName="spiral.snlpa")

                        # problemSize=generator.generateSNLPProblem(400)
                        # anchorSize=generator.generateSNLPProblemAnchors(1000)
                        # options=[generator.name(),generator.modelsize(),problemSize,anchorSize,metricOptions]
                        # runSNLP2D(generator.outPath,
                        #           generator.problemName,
                        #           generator.anchorName,
                        #           populationGenerator.outName,
                        #           [problemSize,
                        #           anchorSize],
                        #           generator.modelsize(),
                        #           5,
                        #           currentflags,options)