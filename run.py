import subprocess
import SNLP
import visualize
import re
import os

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

def runSNLP3D():
    problemLBFGS= SNLP.OptProblem(name="PROBLEM_SNLP3D", 
                        optimizer="LBFGS", 
                        inputPaths=["./SNLP3D/problems/poly100/poly100.snlp", 
                                    "./SNLP3D/problems/poly100/poly100.snlpa"],
                        outputPath="./SNLP3D/problems/poly100/csv/3D/LBFGS")

    problemGD= SNLP.OptProblem(name="PROBLEM_SNLP3D", 
                        optimizer="GD", 
                        inputPaths=["./SNLP3D/problems/poly100/poly100.snlp", 
                                    "./SNLP3D/problems/poly100/poly100.snlpa"],
                        outputPath= "./SNLP3D/problems/poly100/csv/3D/GD")
    problems = [problemLBFGS,problemGD]
    [runSNLP(problem=problem) for problem in problems]
    fvisualizer = visualize.FVisualizer(problems)
    fvisualizer.visualize()
    xvisualizer = visualize.SNLP3DVisualizer(problems)
    xvisualizer.visualize()

def runSNLP2D():
    problemLBFGS = SNLP.OptProblem(name="PROBLEM_SNLP", 
                        optimizer="LBFGS", 
                        inputPaths=["./SNLP/problems/poly100/poly100.snlp", 
                                    "./SNLP/problems/poly100/poly100.snlpa"],
                        outputPath="./SNLP/problems/poly100/csv/2D/LBFGS")
    problemGD = SNLP.OptProblem(name="PROBLEM_SNLP", 
                        optimizer="GD", 
                        inputPaths=["./SNLP/problems/poly100/poly100.snlp", 
                                    "./SNLP/problems/poly100/poly100.snlpa"],
                        outputPath="./SNLP/problems/poly100/csv/2D/GD")
    problems = [problemLBFGS,problemGD]
    [runSNLP(problem=problem) for problem in problems]
    fvisualizer = visualize.FVisualizer(problems)
    fvisualizer.visualize()
    xvisualizer = visualize.SNLP2DVisualizer(problems)
    xvisualizer.visualize()

runSNLP3D()