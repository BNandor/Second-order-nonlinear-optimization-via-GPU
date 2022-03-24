import subprocess

from numpy import append
import SNLP
import visualize
import re
import os
import generate
import csv


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
    write = csv.writer(outfile)
    write.writerows([columns])
    outfile.close()  

def findReplaceFirst(logs,find,replace,replacedWith):
    finds=[log for log in logs if re.search(find,log)]
    if len(finds) == 0:
        print(f"Cant find {find} from {len(logs)} lines")
    return finds[0].rstrip().replace(replace,replacedWith)

def appendMetrics(logs,metrics:SNLP.Metrics,metricsName):
    metricKeys=["time ms :","threads:","iterations:","final f: "]
    measurements=[findReplaceFirst(logs,metricKey,metricKey,"") for metricKey in metricKeys]
    appendColumnsTo(outPath=metrics.path,outName=metricsName,columns=metrics.options + measurements)

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

def runSNLP3D(problemPath,problemName,anchorName,residualSizes,modelsize,framesize,optionalFlags,metricsConfig):
    problemLBFGS = SNLP.OptProblem(name="PROBLEM_SNLP3D", 
                        optimizer="LBFGS", 
                        inputPaths=[f"{problemPath}/{problemName}", 
                                    f"{problemPath}/{anchorName}"],
                        outputPath=f"{problemPath}/csv/3D/LBFGS",
                        constantsSizes=residualSizes,
                        modelsize=modelsize,
                        framesize=framesize,
                        metrics=SNLP.Metrics( path=f"{problemPath}/csv/3D/LBFGS/metrics", options=metricsConfig))
    problemGD = SNLP.OptProblem(name="PROBLEM_SNLP3D", 
                        optimizer="GD", 
                        inputPaths=[f"{problemPath}/{problemName}", 
                                    f"{problemPath}/{anchorName}"],
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

def runSNLP2D(problemPath,problemName,anchorName,residualSizes,modelsize,framesize,optionalFlags,metricsConfig):
    problemLBFGS = SNLP.OptProblem(name="PROBLEM_SNLP", 
                        optimizer="LBFGS", 
                        inputPaths=[f"{problemPath}/{problemName}", 
                                    f"{problemPath}/{anchorName}"],
                        outputPath=f"{problemPath}/csv/2D/LBFGS",
                        constantsSizes=residualSizes,
                        modelsize=modelsize,
                        framesize=framesize,
                        metrics=SNLP.Metrics( path=f"{problemPath}/csv/2D/LBFGS/metrics", options=metricsConfig))
    problemGD = SNLP.OptProblem(name="PROBLEM_SNLP", 
                        optimizer="GD", 
                        inputPaths=[f"{problemPath}/{problemName}", 
                                    f"{problemPath}/{anchorName}"],
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

metricOptions="-DGLOBAL_SHARED_MEM"              
# metricOptions=""
# currentflags=f"-DPRINT {metricOptions}"
currentflags=f" {metricOptions}"

generator=generate.Generate3DStructuredProblem1(nodecount=1000, 
                                                outPath="./SNLP3D/problems/poly100",
                                                problemName="spiral.snlp",
                                                anchorName="spiral.snlpa")


problemSize=generator.generateSNLPProblem(100)
anchorSize=generator.generateSNLPProblemAnchors(1000)

options=[generator.name(),generator.modelsize(),problemSize,anchorSize,metricOptions]
testCount=10
for testCase in range(testCount):
    print(f"testcase {testCase}\n")
    runSNLP3D(generator.outPath,
          generator.problemName,
          generator.anchorName,
          [problemSize,
          anchorSize],
          generator.modelsize(),
          10,currentflags,options)

# generator=generate.Generate2DStructuredProblem1(nodecount=20, 
#                                                 outPath="./SNLP2D/problems/poly100",
#                                                 problemName="spiral.snlp",                     
#                                                 anchorName="spiral.snlpa")

# problemSize=generator.generateSNLPProblem(400)
# anchorSize=generator.generateSNLPProblemAnchors(1000)
# options=[generator.name(),generator.modelsize(),problemSize,anchorSize,metricOptions]
# runSNLP2D(generator.outPath,
#           generator.problemName,
#           generator.anchorName,
#           [problemSize,
#           anchorSize],
#           generator.modelsize(),
#           5,
#           currentflags,options)