import subprocess
import SNLP
import visualize
import re
import os
import generate

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

def runSNLP(problem:SNLP.OptProblem):
    print(f"Running: {problem.name} with {problem.optimizer}")
    currentProblem = SNLP.SNLP(problem)
    logs=consumeOutput(runOptimizerWith(currentProblem.flags("-DPRINT")),lambda line:print(line))
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

def runSNLP3D(problemPath,problemName,anchorName,residualSizes,modelsize):
    problemLBFGS = SNLP.OptProblem(name="PROBLEM_SNLP3D", 
                        optimizer="LBFGS", 
                        inputPaths=[f"{problemPath}/{problemName}", 
                                    f"{problemPath}/{anchorName}"],
                        outputPath=f"{problemPath}/csv/3D/LBFGS",
                        constantsSizes=residualSizes,
                        modelsize=modelsize)
    problemGD = SNLP.OptProblem(name="PROBLEM_SNLP3D", 
                        optimizer="GD", 
                        inputPaths=[f"{problemPath}/{problemName}", 
                                    f"{problemPath}/{anchorName}"],
                        outputPath=f"{problemPath}/csv/3D/GD",
                        constantsSizes=residualSizes,
                        modelsize=modelsize)
    problems = [problemLBFGS,problemGD]
    [runSNLP(problem=problem) for problem in problems]
    fvisualizer = visualize.FVisualizer(problems)
    fvisualizer.visualize()
    xvisualizer = visualize.SNLP3DVisualizer(problems)
    xvisualizer.visualize()

def runSNLP2D(problemPath,problemName,anchorName,residualSizes,modelsize):
    problemLBFGS = SNLP.OptProblem(name="PROBLEM_SNLP", 
                        optimizer="LBFGS", 
                        inputPaths=[f"{problemPath}/{problemName}", 
                                    f"{problemPath}/{anchorName}"],
                        outputPath=f"{problemPath}/csv/2D/LBFGS",
                        constantsSizes=residualSizes,
                        modelsize=modelsize)
    problemGD = SNLP.OptProblem(name="PROBLEM_SNLP", 
                        optimizer="GD", 
                        inputPaths=[f"{problemPath}/{problemName}", 
                                    f"{problemPath}/{anchorName}"],
                        outputPath=f"{problemPath}/csv/2D/GD",
                        constantsSizes=residualSizes,
                        modelsize=modelsize)
    problems = [problemLBFGS,problemGD]
    [runSNLP(problem=problem) for problem in problems]
    fvisualizer = visualize.FVisualizer(problems)
    fvisualizer.visualize()
    xvisualizer = visualize.SNLP2DVisualizer(problems)
    xvisualizer.visualize()


# generator=generate.Generate3DStructuredProblem1(nodecount=120, 
#                                                 outPath="./SNLP3D/problems/poly100",
#                                                 problemName="spiral.snlp",
#                                                 anchorName="spiral.snlpa")

# problemSize=generator.generateSNLPProblem(100)
# anchorSize=generator.generateSNLPProblemAnchors(1000)

# runSNLP3D(generator.outPath,
#           generator.problemName,
#           generator.anchorName,
#           [problemSize,
#           anchorSize],
#           generator.modelsize())

generator=generate.Generate2DStructuredProblem1(nodecount=10, 
                                                outPath="./SNLP2D/problems/poly100",
                                                problemName="spiral.snlp",                     
                                                anchorName="spiral.snlpa")

problemSize=generator.generateSNLPProblem(400)
anchorSize=generator.generateSNLPProblemAnchors(1000)

runSNLP2D(generator.outPath,
          generator.problemName,
          generator.anchorName,
          [problemSize,
          anchorSize],
          generator.modelsize())