import random

n=100
anchorCount=10
maxDist = 100
minX = -10000
maxX = 10000
minY = -10000
maxY = 10000
minZ = -10000
maxZ = 10000

for i in range(anchorCount):
    x=minX + random.random()*(maxX-minX)
    y=minY + random.random()*(maxY-minY)
    z=minZ + random.random()*(maxZ-minZ)
    i=random.randint(0,n-1)#
    d=random.random()*maxDist
    print(f"{x} {y} {z} {i} {d}")
