from collections import namedtuple
from platform import node
backslash="\\"
dquote='"'

OptProblem = namedtuple("OptProblem", "name optimizer inputPaths outputPath constantsSizes modelsize framesize metrics")
Metrics= namedtuple("Metrics","path options")

XHIST="xhistory.csv"
FHIST="fhistory.csv"
METRICS="metrics.csv"

class SNLP:
    def __init__(self,optproblem:OptProblem) -> None:
        self.optproblem=optproblem

    def flags(self,optionalFlags) -> str:
        if len(self.optproblem.inputPaths) !=3:
            raise Exception(f'SNLP error: input file paths are: {self.optproblem.inputPaths}')
        return f"-DOPTIMIZER={self.optproblem.optimizer}\
                 -D{self.optproblem.name}\
                 -DPROBLEM_PATH={backslash}{dquote}{self.optproblem.inputPaths[0]}{backslash}{dquote} \
                 -DPROBLEM_ANCHOR_PATH={backslash}{dquote}{self.optproblem.inputPaths[1]}{backslash}{dquote} \
                 -DPROBLEM_INPUT_POPULATION_PATH={backslash}{dquote}{self.optproblem.inputPaths[2]}{backslash}{dquote} \
                 -DRESIDUAL_CONSTANTS_COUNT_1={self.optproblem.constantsSizes[0]} \
                 -DRESIDUAL_CONSTANTS_COUNT_2={self.optproblem.constantsSizes[1]} \
                 -DX_DIM={self.optproblem.modelsize} \
                 -DFRAMESIZE={self.optproblem.framesize} \
                 {optionalFlags}"