import pandas as pd 
import concepts
from common import *

def getConcepts(objects,properties,objectValues):
    ctx={'objects':objects,'properties':properties,'context':[]}
    for row in objectValues:
        attributes=[properties.index(attribute) for attribute in row]
        ctx['context'].append(attributes)
    return concepts.Context.fromdict(ctx)

def saveContext(ctx, to):
    ctx.tofile(to)

def createContext(comparisonDF,attributeColumns):
    properties=[list(zip([column] * comparisonDF[column].unique().size,comparisonDF[column].unique()))    for column in attributeColumns]
    properties=[p for ps in properties for p in ps ]
    # print([row for row in comparisonDF.to_dict().keys()])
    # print(properties)
    comparisonDict=comparisonDF.to_dict(orient='index')
    objectAttributes=[]
    for index in comparisonDict.keys():
        attributes=[]
        for column in attributeColumns:
            attributes.append((column,comparisonDict[index][column]))
        objectAttributes.append(attributes)
    
    # print(objectAttributes)
    # print([stringifyList(oAttrs) for oAttrs in objectAttributes])
    return getConcepts(
            stringifyList(comparisonDF.index.to_list()),
            stringifyList(properties),
            [stringifyList(oAttrs) for oAttrs in objectAttributes])