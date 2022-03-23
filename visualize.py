import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
import SNLP
from typing import List
from collections import namedtuple
import time

SNLPVisualizerData = namedtuple("SNLPVisualizerData", "edges xs anchors")

class SNLP3DVisualizer:
    def __init__(self, problems:List[SNLP.OptProblem]) -> None:
        self.problems=problems

    def getProblemData(self,problem:SNLP.OptProblem)->SNLPVisualizerData:
        historypath=f"{problem.outputPath}/{SNLP.XHIST}"
        problempath=f"{problem.inputPaths[0]}"
        anchorspath=f"{problem.inputPaths[1]}"
        with open(historypath, 'r') as xhistf:
            with open(problempath,'r') as edgesf:
                with open(anchorspath,'r') as anchorsf:
                    edges = np.array([[float(num) for num in edge.rstrip().split(' ')] for edge in edgesf])
                    allx = [[num for num in xline.rstrip().split(',')] for xline in xhistf]
                    allx = [xline for xline in allx if len(xline) == len(allx[0])]
                    xs = [[float(num) for num in xline] for xline in allx]
                    allanchors = np.array([[float(num) for num in anchor.rstrip().split(' ')] for anchor in anchorsf])
                    return SNLPVisualizerData(edges=edges,xs=np.array(xs),anchors=allanchors)
    def setAx(self,ax):
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    def setAxTitles(self,ax,problem):
        ax.set_title(problem.optimizer)

    def visualize(self):
        fig, axs = plt.subplots(1, len(self.problems),figsize=(15,15),subplot_kw=dict(projection='3d'))
        [self.setAx(ax) for ax in axs]
        data:List[SNLPVisualizerData]=[self.getProblemData(problem=problem) for problem in self.problems]
        for i in range(max(d.xs.size for d in data)):
            [self.setAxTitles(ax,problem) for ax,problem in zip(axs,self.problems)]
            for j in range(len(self.problems)):
                currentX=data[j].xs[min(i,data[j].xs.size)]
                currentEdges=data[j].edges
                currentAnchors=data[j].anchors
                currentAx=axs[j]
                edgeLines=np.array([ [[currentX[0::3][int(edge[0])],currentX[1::3][int(edge[0])],currentX[2::3][int(edge[0])]],[currentX[0::3][int(edge[1])],currentX[1::3][int(edge[1])],currentX[2::3][int(edge[1])]]] for edge in currentEdges])
                colors = np.array([[0,1.0/(1.0001**(abs(np.sum((currentX[0]-currentX[1])**2)-edge[2]))),0] for line,edge in zip(edgeLines,currentEdges)])
                colors[:,0]=1-colors[:,1]
                line_segments = Line3DCollection(edgeLines, colors=colors,linestyle='solid',linewidth=1)
                currentAx.add_collection(line_segments)
                currentAx.scatter(currentX[0::3], currentX[1::3],currentX[2::3], s=1,c="g")
                anchorLines=np.array([ [[anchor[0],anchor[1],anchor[2]],[currentX[0::3][int(anchor[3])],currentX[1::3][int(anchor[3])],currentX[2::3][int(anchor[3])]]] for anchor in currentAnchors])
                anchorSegments = Line3DCollection(anchorLines,linestyle='solid',linewidth=1)
                currentAx.add_collection(anchorSegments)
                currentAx.scatter(currentAnchors[:,0], currentAnchors[:,1],currentAnchors[:,2], s=20,c="r")
            plt.draw()
            plt.waitforbuttonpress()
            [ax.clear() for ax in axs]
            plt.cla()

class SNLP2DVisualizer:
    def __init__(self,problems:List[SNLP.OptProblem]) -> None:
        self.problems=problems

    def getProblemData(self,problem:SNLP.OptProblem)->SNLPVisualizerData:
        historypath=f"{problem.outputPath}/{SNLP.XHIST}"
        problempath=f"{problem.inputPaths[0]}"
        anchorspath=f"{problem.inputPaths[1]}"
        with open(historypath, 'r') as xhistf:
            with open(problempath,'r') as edgesf:
                with open(anchorspath,'r') as anchorsf:
                    edges = np.array([[float(num) for num in edge.rstrip().split(' ')] for edge in edgesf])
                    allx = [[num for num in xline.rstrip().split(',')] for xline in xhistf]
                    allx = [xline for xline in allx if len(xline) == len(allx[0])]
                    xs = [[float(num) for num in xline] for xline in allx]
                    allanchors = np.array([[float(num) for num in anchor.rstrip().split(' ')] for anchor in anchorsf])
                    return SNLPVisualizerData(edges=edges,xs=np.array(xs),anchors=allanchors)
                    
    def setAx(self,ax):
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
        
    def setAxTitles(self,ax,problem):
        ax.set_title(problem.optimizer)

    def visualize(self):
        fig, axs = plt.subplots(1, len(self.problems),figsize=(15,15))
        [self.setAx(ax) for ax in axs]
        data:List[SNLPVisualizerData]=[self.getProblemData(problem=problem) for problem in self.problems]
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
        self.skipfirst=100//problems[0].framesize

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