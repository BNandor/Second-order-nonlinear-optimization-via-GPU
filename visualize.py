import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
import SNLP
from typing import List
from collections import namedtuple
import time

class SNLP3DVisualizer:
    def __init__(self, optData:SNLP.OptProblem) -> None:
        self.optData=optData

    def visualize(self):
        history=f"{self.optData.outputPath}/{SNLP.XHIST}"
        problem=f"{self.optData.inputPaths[0]}"
        anchors=f"{self.optData.inputPaths[1]}"

        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        with open(history, 'r') as f:
            with open(problem,'r') as edgesf:
                with open(anchors,'r') as anchorsf:

                    edges = np.array([[float(num) for num in edge.rstrip().split(' ')] for edge in edgesf])
                    allines = [[float(num) for num in line.rstrip().split(',')] for line in f]
                    allanchors = np.array([[float(num) for num in anchor.rstrip().split(' ')] for anchor in anchorsf])
                    lines= []
                    for line in allines:
                        if len(line) == len(allines[0]):
                            lines.append(line)
                            
                    arr = np.array(lines)
                    
                    for line in arr:
                        edgeLines=np.array([ [[line[0::3][int(edge[0])],line[1::3][int(edge[0])],line[2::3][int(edge[0])]],[line[0::3][int(edge[1])],line[1::3][int(edge[1])],line[2::3][int(edge[1])]]] for edge in edges])
                        colors = np.array([[0,1.0/(1.0001**(abs(np.sum((line[0]-line[1])**2)-edge[2]))),0] for line,edge in zip(edgeLines,edges)])
                        colors[:,0]=1-colors[:,1]
                        line_segments = Line3DCollection(edgeLines, colors=colors,linestyle='solid',linewidth=1)
                        ax.add_collection(line_segments)
                        img=ax.scatter(line[0::3], line[1::3],line[2::3], s=20,c="g")
                        # anchorLines=np.array([ [[anchor[0],anchor[1]],[line[0::2][int(anchor[2])],line[1::2][int(anchor[2])]]] for anchor in allanchors])
                        # anchorSegments = LineCollection(anchorLines,linestyle='solid',linewidth=1)
                        # ax.add_collection(anchorSegments)
                        # img2=ax.scatter(allanchors[:,0], allanchors[:,1], s=20,c="r")
                        plt.draw()
                        plt.waitforbuttonpress()
                        img.remove()
                        plt.cla()

SNLP2DVisualizerData = namedtuple("SNLP2DVisualizerData", "edges xs anchors")
class SNLP2DVisualizer:
    def __init__(self,problems:List[SNLP.OptProblem]) -> None:
        self.problems=problems

    def getProblemData(self,problem:SNLP.OptProblem)->SNLP2DVisualizerData:
        historypath=f"{problem.outputPath}/{SNLP.XHIST}"
        problempath=f"{problem.inputPaths[0]}"
        anchorspath=f"{problem.inputPaths[1]}"
        with open(historypath, 'r') as xhistf:
            with open(problempath,'r') as edgesf:
                with open(anchorspath,'r') as anchorsf:
                    edges = np.array([[float(num) for num in edge.rstrip().split(' ')] for edge in edgesf])
                    allx = [[float(num) for num in xline.rstrip().split(',')] for xline in xhistf]
                    xs= []
                    for x in allx:
                        if len(x) == len(allx[0]):
                            xs.append(x)
                    allanchors = np.array([[float(num) for num in anchor.rstrip().split(' ')] for anchor in anchorsf])
                    return SNLP2DVisualizerData(edges=edges,xs=np.array(xs),anchors=allanchors)
    def setAx(self,ax,problem:SNLP.OptProblem):
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
        
    def setAxTitles(self,ax,problem):
        ax.set_title(problem.optimizer)

    def visualize(self):
        fig, axs = plt.subplots(1, len(self.problems),figsize=(15,15))
        [self.setAx(ax,problem) for ax,problem in zip(axs,self.problems)]
        data:List[SNLP2DVisualizerData]=[self.getProblemData(problem=problem) for problem in self.problems]
        for i in range(max(d.xs.size for d in data)):
            [self.setAxTitles(ax,problem) for ax,problem in zip(axs,self.problems)]
            for j in range(len(self.problems)):
                currentX=data[j].xs[min(i,data[j].xs.size)]
                currentEdges=data[j].edges
                currentAnchors=data[j].anchors
                currentAx=axs[j]
                edgeLines=np.array([ [[currentX[0::2][int(edge[0])],currentX[1::2][int(edge[0])]],[currentX[0::2][int(edge[1])],currentX[1::2][int(edge[1])]]] for edge in currentEdges])
                colors = np.array([[0,1.0/(1.0001**(abs(np.sum((currentX[0]-currentX[1])**2)-edge[2]))),0] for line,edge in zip(edgeLines,currentEdges)])
                colors[:,0]=1-colors[:,1]
                line_segments = LineCollection(edgeLines, colors=colors,linestyle='solid',linewidth=1)
                currentAx.add_collection(line_segments)
                currentAx.scatter(currentX[0::2], currentX[1::2], s=20,c="g")
                anchorLines=np.array([ [[anchor[0],anchor[1]],[currentX[0::2][int(anchor[2])],currentX[1::2][int(anchor[2])]]] for anchor in currentAnchors])
                anchorSegments = LineCollection(anchorLines,linestyle='solid',linewidth=1)
                currentAx.add_collection(anchorSegments)
                currentAx.scatter(currentAnchors[:,0], currentAnchors[:,1], s=20,c="r")
            plt.draw()
            plt.waitforbuttonpress()
            [ax.clear() for ax in axs]
            plt.cla()

class FVisualizer:
    def __init__(self,problems:List[SNLP.OptProblem]) -> None:
        self.problems=problems
        self.skipfirst=100

    def visualize(self):
        fhistories=[f"{problem.outputPath}/{SNLP.FHIST}" for problem in self.problems]
        fvalues=[]
        for fhistory in fhistories:
            with open(fhistory, 'r') as flines:            
                fvalues.append(np.array([float(fline.rstrip())  for fline in flines]))

        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot()
        ax.set_xlabel('iterations')
        ax.set_ylabel('cost')
        plt.title(f"{self.problems[0].name}")
        for fvalue,problem in zip(fvalues,self.problems):
            ax.plot(fvalue[self.skipfirst:],label=f"{problem.optimizer}") 
        ax.legend()
        plt.draw()
        plt.waitforbuttonpress()