import os
import random
from functools import reduce

class PopulationGenerator():
    def __init__(self, populationsize, modelsize, outPath,outName, boundingBox) -> None:
        self.populationsize = populationsize
        self.modelsize = modelsize
        self.outPath = outPath
        self.outName = outName
        self.boundingBox = boundingBox

    def generate(self):
        if not os.path.exists(self.outPath):
            os.makedirs(f"{self.outPath}")
        population=[]
        for i in range(self.populationsize):# n-1 edges
            randomModel=[str(-self.boundingBox/2 + random.random()*self.boundingBox) for i in range(self.modelsize)]
            line=reduce(lambda a, b: a +" "+ b+" ",randomModel,"")
            population.append(f"{line}\n")
        if not os.path.exists(f"{self.outPath}/{self.outName}"):
            outfile = open(f"{self.outPath}/{self.outName}", "w")
            [outfile.write(line) for line in population]
            outfile.close()  
        
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

class Generate3DRandomProblem1:
    def __init__(self,nodecount,outPath,problemName,anchorName) -> None:
        self.nodecount=nodecount
        self.outPath=outPath
        self.problemName=problemName
        self.anchorName=anchorName

    def modelsize(self):
        return self.nodecount*3

    def name(self):
        return "3DRandom1"

    def generateSNLPProblem(self,maxDist):
        if not os.path.exists(self.outPath):
            os.makedirs(f"{self.outPath}")
        n=self.nodecount
        edgC=5
        print(f"Writing SNLP problem to {self.outPath}/{self.problemName} with {n} nodes and {maxDist} maxDist")
        d=maxDist        
        problem=[]
        for i in range(n-edgC-1):
            a=i
            used=[]
            for j in range(1+random.randint(0,edgC)):
                b=random.randint(i,n-1)
                while a == b or b in used:
                    b=random.randint(i,n-1)
                used.append(b)
                d=random.random()*maxDist
                problem.append(f"{a} {b} {d}\n")
        problem[-1]=problem[-1].rstrip()
        if not os.path.exists(f"{self.outPath}/{self.problemName}"):
            outfile = open(f"{self.outPath}/{self.problemName}", "w")
            [outfile.write(line) for line in problem]
            outfile.close()  
            return len(problem)
        else: 
            return sum(1 for line in open(f"{self.outPath}/{self.problemName}", "r"))


    def generateSNLPProblemAnchors(self,boundingBoxSize):
        if not os.path.exists(self.outPath):
            os.makedirs(f"{self.outPath}")
        n=self.nodecount
        print(f"Writing SNLP3D anchors to {self.outPath}/{self.anchorName} with {n} nodes and {boundingBoxSize} bounding box")
        problem=[]
        minX=-boundingBoxSize/2
        minY=-boundingBoxSize/2
        minZ=-boundingBoxSize/2
        maxX=boundingBoxSize/2
        maxY=boundingBoxSize/2
        maxZ=boundingBoxSize/2
        for i in range(n):
            x=minX + random.random()*(maxX-minX)
            y=minY + random.random()*(maxY-minY)
            z=minZ + random.random()*(maxZ-minZ)
            i=random.randint(0,n-1)#
            d=random.random()*boundingBoxSize/10
            problem.append(f"{x} {y} {z} {i} {d}\n")
        problem[-1]=problem[-1].rstrip()
        if not os.path.exists(f"{self.outPath}/{self.anchorName}"):
            outfile = open(f"{self.outPath}/{self.anchorName}", "w")
            [outfile.write(line) for line in problem]
            outfile.close()  
            return len(problem)
        else: 
            return sum(1 for line in open(f"{self.outPath}/{self.anchorName}", "r"))