import json
import json5
from dict_hash import sha256
import itertools

def loadJsonFrom(path,ignoreTrailingCommas=False):
    f = open(path)
    if ignoreTrailingCommas:
        return json5.load(f)
    else:
        return json.load(f)

def stringifyList(list):
    return [str(item) for item in list]


def zipWithProperty(list,property):
    print([property]*len(list))
    return zip([property]*len(list),list)

def mapExperimentListToDict(experiment):
    paramsDict={}
    for param in experiment:
        if param[1] != None:
            paramsDict[param[0]]=param[1]
    return json.loads(json.dumps(paramsDict))

def hashOfExperiment(experiment):
    return sha256(experiment)

def possibleExperimentIds(experimentParams):
    ids=[]
    variations=list(itertools.product(*list(experimentParams.values())))
    for experiment in variations:
        experimentDict=mapExperimentListToDict(experiment=experiment)
        ids.append(hashOfExperiment(experimentDict))
    return ids

def matchOneIdInIndex(index,ids):
    mathcingRandomIs=list(filter(lambda id: id in index,ids))
    assert len(mathcingRandomIs) == 1
    return mathcingRandomIs[0]

def printMinResultEachRow(df,experimentCols,columns):
    for  index,row in df.iterrows():
        minResult=min([row[column] for column in columns])
        methodsHavingMinResult=[column for column in columns if row[column] == minResult]
        print(f"{[row[column] for column in experimentCols]}{methodsHavingMinResult}->{minResult}")
