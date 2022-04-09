import json

def caseIdentifier(log):
    return f"{log['nodecount'] } { log['edges']} {log['anchors']} {log['totaliterations']} {log['population']} {log['deIteration']} {log['distFraction']} {log['testcase']}"
    

def readCases(path):
    metrics=open(path,'r')
    logs=[]
    for log in metrics:
        logs.append(json.loads(log))
    return list(map(lambda log: caseIdentifier(log),logs))

def readLogs(path):
    metrics=open(path,'r')
    return [json.loads(log) for log in metrics] 
