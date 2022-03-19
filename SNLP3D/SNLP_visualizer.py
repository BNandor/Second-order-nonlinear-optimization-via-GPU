#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection


problemName="poly100"
problempath="./problems/"+problemName
history=problempath+"/csv/"+problemName+".csv"
problem=problempath+"/"+problemName+".snlp"
anhors=problempath+"/"+problemName+".snlpa"

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# ax.axis('equal')
with open(history, 'r') as f:
    with open(problem,'r') as edgesf:
        with open(anhors,'r') as anchorsf:

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