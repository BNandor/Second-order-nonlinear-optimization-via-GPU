import runConfig
popsize=10
DEGen=1
totalIterations=100
iterations=totalIterations//(DEGen+1)
nodecount=27
maxDist=400
box=100
solver="OPTIMIZER_MIN_DE"
outPath="./SNLP2D/problems/dvhopbench"
problemName="exportedDVHop.snlp"                
anchorName="exportedDVHop.snlpa"
runConfig.hybridDE(popsize,DEGen,totalIterations,iterations,nodecount,maxDist,box,outPath,problemName,anchorName,solver)