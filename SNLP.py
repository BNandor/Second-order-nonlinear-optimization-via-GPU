from collections import namedtuple
backslash="\\"
dquote='"'

OptProblem = namedtuple("OptProblem", "name optimizer inputPaths outputPath ")

XHIST="xhistory.csv"
FHIST="fhistory.csv"

class SNLP:
    def __init__(self,optproblem:OptProblem) -> None:
        self.optproblem=optproblem
    def flags(self,optionalFlags) -> str:
        if len(self.optproblem.inputPaths) !=2:
            raise Exception(f'SNLP3D error: input file paths are: {self.optproblem.inputPaths}')
        return f"-DOPTIMIZER={self.optproblem.optimizer} -D{self.optproblem.name} -DPROBLEM_PATH={backslash}{dquote}{self.optproblem.inputPaths[0]}{backslash}{dquote} -DPROBLEM_ANCHOR_PATH={backslash}{dquote}{self.optproblem.inputPaths[1]}{backslash}{dquote} {optionalFlags}"