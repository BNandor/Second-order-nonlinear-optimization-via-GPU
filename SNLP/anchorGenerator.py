import random

n=1000
anchorCount=10
maxDist = 1000
minX = -10000
maxX = 10000
minY = -10000
maxY = 10000
for i in range(anchorCount):
    x=minX + random.random()*(maxX-minX)
    y=minY + random.random()*(maxY-minY)
    i=random.randint(0,n-1)#
    d=random.random()*maxDist
    print(f"{x} {y} {i} {d}")
