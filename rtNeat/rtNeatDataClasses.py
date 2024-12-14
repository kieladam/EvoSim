from dataclasses import dataclass

# These are both genes, and can be mutated and passed to offspring.
# They can be compared between individuals to calculate speciation.
@dataclass
class rtNode:
    idx:int = 0
    isInput:bool = False
    isOutput:bool = False
    layer:int = 0
    bias:float = 0
    actFunc = None

    def __str__(self) -> str:
        s = f'idx: {self.idx}\n'
        if self.isInput: s += 'Input\n'
        elif self.isOutput: s += 'Output\n'
        else: s += f'Layer: {self.layer}\n'
        s += f'Bias: {self.bias}\n'
        s += f'Activation: {self.actFunc}\n'
        return s

@dataclass
class rtEdge:
    inNode:int = 0
    outNode:int = 0
    idx:tuple = (0, 0)
    weight:float = 0
    enabled:bool = True
    innovation:int = 0

    def __str__(self) -> str:
        s = f'idx: {self.idx}\n'
        s += f'Weight: {self.weight}\n'
        s += f'Innovation: {self.innovation}\n'
        if self.enabled:
            s += 'Enabled\n'
        else:
            s += 'Disabled\n'
        return s

#create a default genome? all inputs connected to all outputs or something.
@dataclass
class Genome:
    nodes = []
    edges = []
    nodeChance:float = 0.03 #These two values will control how quickly nets can grow.
    edgeChance:float = 0.1
    hiddenLayers:float = 0