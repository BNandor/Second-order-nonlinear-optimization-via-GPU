import json
import json5

def loadJsonFrom(path,ignoreTrailingCommas=False):
    f = open(path)
    if ignoreTrailingCommas:
        return json5.load(f)
    else:
        return json.load(f)

def stringifyList(list):
    return [str(item) for item in list]