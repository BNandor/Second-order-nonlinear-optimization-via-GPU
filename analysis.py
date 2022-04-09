import json

def caseIdentifier(log):
    return f"{log['nodecount'] } { log['edges']} {log['anchors']} {log['totaliterations']} {log['population']} {log['deIteration']} {log['distFraction']} {log['testcase']}"

def cases(path):
    metrics=open(path,'r')
    lines = []
    for line in metrics:
        lines.append(line)

    logs=[]
    for (i,line) in enumerate(lines):
        print(line.replace("}\n",f", \"testcase\": {i%2}"+"}"))
        logs.append(json.loads(line))
    return list(map(lambda log: caseIdentifier(log),logs))
