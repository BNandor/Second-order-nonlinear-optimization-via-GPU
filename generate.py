import os
import random

class Generate3DStructuredProblem1:
    def __init__(self,nodecount,outPath,problemName,anchorName) -> None:
        self.nodecount=nodecount
        self.outPath=outPath
        self.problemName=problemName
        self.anchorName=anchorName

    def modelsize(self):
        return self.nodecount*3

    def name(self):
        return "3DStructure1"

    def generateSNLPProblem(self,maxDist):
        if not os.path.exists(self.outPath):
            os.makedirs(f"{self.outPath}")
        n=self.nodecount
        print(f"Writing SNLP problem to {self.outPath}/{self.problemName} with {n} nodes and {maxDist} maxDist")
        outfile = open(f"{self.outPath}/{self.problemName}", "w")
        d=maxDist        
        problem=[]
        for i in range(0,n-1):# n-1 edges
            a=i
            b=i+1
            problem.append(f"{a} {b} {d}\n")
        d=d*1.5
        for i in range(0,n-2):
            a=i
            b=i+2
            problem.append(f"{a} {b} {d}\n")
        for i in range(0,n-3):
            a=i
            b=i+3
            problem.append(f"{a} {b} {d}\n")
        problem[-1]=problem[-1].rstrip()
        [outfile.write(line) for line in problem]
        outfile.close()  
        return len(problem)

    def generateSNLPProblemAnchors(self,boundingBoxSize):
        if not os.path.exists(self.outPath):
            os.makedirs(f"{self.outPath}")
        n=self.nodecount
        print(f"Writing SNLP3D anchors to {self.outPath}/{self.anchorName} with {n} nodes and {boundingBoxSize} bounding box")
        outfile = open(f"{self.outPath}/{self.anchorName}", "w")
        minX = -boundingBoxSize/2
        maxX = boundingBoxSize/2
        minY = -boundingBoxSize/2
        maxY = boundingBoxSize/2
        minZ = -boundingBoxSize/2
        maxZ = boundingBoxSize/2
        problem=[]
        problem.append(f"{minX} {minY} {minZ} {0} {0}\n")
        problem.append(f"{maxX} {maxY} {maxZ} {n-1} {0}\n")
        problem[-1]=problem[-1].rstrip()
        [outfile.write(line) for line in problem]
        outfile.close()  
        return len(problem)

class Generate2DStructuredProblem1:
    def __init__(self,nodecount,outPath,problemName,anchorName) -> None:
        self.nodecount=nodecount
        self.outPath=outPath
        self.problemName=problemName
        self.anchorName=anchorName

    def modelsize(self):
        return self.nodecount*2

    def name(self):
        return "2DStructure1"


    def generateSNLPProblem(self,maxDist):
        if not os.path.exists(self.outPath):
            os.makedirs(f"{self.outPath}")
        n=self.nodecount
        print(f"Writing SNLP problem to {self.outPath}/{self.problemName} with {n} nodes and {maxDist} maxDist")
        outfile = open(f"{self.outPath}/{self.problemName}", "w")
        d=maxDist        
        problem=[]
        for i in range(0,n-1):# n-1 edges
            a=i
            b=i+1
            problem.append(f"{a} {b} {d}\n")
        d=d*1.5
        for i in range(0,n-2):
            a=i
            b=i+2
            problem.append(f"{a} {b} {d}\n")
        for i in range(0,n-3):
            a=i
            b=i+3
            problem.append(f"{a} {b} {d}\n")
        problem[-1]=problem[-1].rstrip()
        [outfile.write(line) for line in problem]
        outfile.close()  
        return len(problem)

    def generateSNLPProblemAnchors(self,boundingBoxSize):
        if not os.path.exists(self.outPath):
            os.makedirs(f"{self.outPath}")
        n=self.nodecount
        print(f"Writing SNLP3D anchors to {self.outPath}/{self.anchorName} with {n} nodes and {boundingBoxSize} bounding box")
        outfile = open(f"{self.outPath}/{self.anchorName}", "w")
        minX = -boundingBoxSize/2
        maxX = boundingBoxSize/2
        minY = -boundingBoxSize/2
        maxY = boundingBoxSize/2
        problem=[]
        problem.append(f"{minX} {minY} {0} {0}\n")
        problem.append(f"{maxX} {maxY} {n-1} {0}\n")
        problem[-1]=problem[-1].rstrip()
        [outfile.write(line) for line in problem]
        outfile.close()  
        return len(problem)