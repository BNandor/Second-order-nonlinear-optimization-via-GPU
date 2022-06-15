import pandas as pd
def dropRowColumnByFlag(flags,df):
    varyingNeighborsData=df.copy(deep=True)
    varyingNeighborsData=varyingNeighborsData.drop(varyingNeighborsData.columns[[i for i in range(len(flags[0])) if flags[0][i] ==1]], axis=1)
    varyingNeighborsData=varyingNeighborsData.drop([i for i in range(len(flags[0])) if flags[0][i] ==1], axis=0)
    varyingNeighborsData=varyingNeighborsData.reset_index(drop=True)
    varyingNeighborsData.columns = range(varyingNeighborsData.columns.size)
    return varyingNeighborsData

def getSNLP(anchorFlagsData,neighborData,distancesData,outPath):
    varyingNeighborsData=dropRowColumnByFlag(anchorFlagsData,neighborData)
    varyingNeighborsDistanceData=dropRowColumnByFlag(anchorFlagsData,distancesData)
    n=varyingNeighborsDistanceData.shape[0]
    print(varyingNeighborsData)
    print(varyingNeighborsDistanceData)
    outfile = open(outPath, "w")
    for i in range(n):
        for j in range(n):
            if i < j and varyingNeighborsData[i][j]!=0:
                outfile.write(f"{i} {j} {varyingNeighborsDistanceData[i][j]}\n")
    outfile.close()            

def getSNLPA(anchorData,anchorFlagsData,neighborData,distancesData,outPath):
    outfile = open(outPath, "w")
    i=0
    for anchorFlag in anchorFlagsData[0]:
        if anchorFlag ==1: # is anchor
            for j in range(anchorFlagsData[0].shape[0]):
                if neighborData[i][j]==1:
                    outfile.write(f"{anchorData[0][i]} {anchorData[1][i]} {j-sum(anchorFlagsData[0][0:j])} {distancesData[i][j]}\n")

        i+=1
    outfile.close()            
    
anchorCsv='anchors.csv'
anchorFlagsCsv='anchorflags.csv'
neighborsCsv='neighbors.csv'
distancesCsv='distances.csv'
anchorData = pd.read_csv(anchorCsv,header=None,delimiter=" ")
anchorFlagsData = pd.read_csv(anchorFlagsCsv,header=None,delimiter=" ")
neighborData = pd.read_csv(neighborsCsv,header=None,delimiter=" ")
distancesData = pd.read_csv(distancesCsv,header=None,delimiter=" ")

print(anchorData)
print(neighborData)
print(distancesData)
print(anchorFlagsData)

getSNLP(anchorFlagsData,neighborData,distancesData,"exported.snlp")
getSNLPA(anchorData,anchorFlagsData,neighborData,distancesData,"exported.snlpa")
