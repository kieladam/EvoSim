import ActivationFunctions as af
import numpy as np

# These are both genes, and can be mutated and passed to offspring.
# They can be compared between individuals to calculate speciation.
class rtNode():
    def __init__(self, idx, isInput=False, isOutput=False):
        self.idx:int = idx
        self.isInput:bool = isInput
        self.isOutput:bool = isOutput
        self.layer:int = 0
        self.bias:float = 0
        self.actFunc = None
        self.highPerturbationChance:float = 0.1

    def __str__(self) -> str:
        s = f'idx: {self.idx}\n'
        if self.isInput: s += 'Input\n'
        elif self.isOutput: s += 'Output\n'
        else: s += f'Layer: {self.layer}\n'
        s += f'Bias: {self.bias}\n'
        s += f'Activation: {self.actFunc}\n'
        return s
    
    def mutate(self, rng : np.random.Generator):
        if rng.random() < self.highPerturbationChance:
            self.bias = rng.uniform(-1, 1)
        else:
            self.bias += rng.normal(0, 0.3)


class rtEdge:
    def __init__(self, inNode, outNode):
        self.inNode:int = inNode
        self.outNode:int = outNode
        self.idx:tuple = (inNode, outNode)
        self.weight:float = 0
        self.enabled:bool = True
        self.innovation:int = 0
        self.highPerturbationChance:float = 0.1

    def __str__(self) -> str:
        s = f'idx: {self.idx}\n'
        s += f'Weight: {self.weight}\n'
        s += f'Innovation: {self.innovation}\n'
        if self.enabled:
            s += 'Enabled\n'
        else:
            s += 'Disabled\n'
        return s
    
    def mutate(self, rng: np.random.Generator):
        if rng.random() < self.highPerturbationChance:
            self.weight = rng.uniform(-1, 1)
        else:
            self.weight += rng.normal(0, 0.3)

class Genome:
    def __init__(self):
        self.nodes = [] #indices of the global array of nodes held in RTNEAT
        self.edges = [] #tuples that are the keys to the dict of edges
        self.nodeChance:float = 0.03 #These two values will control how quickly nets can grow.
        self.edgeChance:float = 0.07
        self.toggleChance:float = 0.1
        self.hiddenLayers:int = 0

class RTNEAT:
    #default init creates a fully connected network with no hidden layers. maybe a sparcer initial network is better?
    #papers show areound 0.25 chance to initialise each edge.
    #each genome contains its own nodes. 
    #each genome contains its own edges. but their innovation values are kept as a global dict.
    def __init__(self, inputs, outputs, population, initialEdgeChance, rng: np.random.Generator):
        self.genomes = []
        self.edgeIVs = dict()
        self.innovation = 0
        for j in range(population):
            g = Genome()
            for i in range(inputs):
                n = rtNode(i, isInput=True)
                n.layer = 0
                g.nodes.append(n)
            for i in range(outputs):
                n = rtNode(i+inputs, isOutput=True)
                n.actFunc = af.randomFunction(rng)
                n.layer = 1
                g.nodes.append(n)
            pairs = []
            for n in range(inputs):
                for m in range(outputs):
                    pairs.append((n, m+inputs))
            for pair in pairs:
                if rng.random() < initialEdgeChance:
                    e = rtEdge(inNode=pair[0], outNode=pair[1])
                    if e.idx not in self.edgeIVs:
                        e.innovation = self.innovation
                        self.edgeIVs[e.idx] = e.innovation
                        self.innovation += 1
                    else:
                        e.innovation = self.edgeIVs[e.idx]
                    e.weight = rng.uniform(-1, 1)
                    g.edges.append(e)
            self.genomes.append(g)

    def packageGenome(self, genomeIDX):
        n = dict()
        g = self.genomes[genomeIDX]
        for node in g.nodes:
            print(node)
            n[node.idx] = node.layer
        e = []
        for edge in g.edges:
            e.append((edge.idx[0], edge.idx[1], edge.weight))
        return (n, e)
    
    def addEdge(self, start, end):
        e = rtEdge(start, end)
        if e.idx in self.edgeIVs:
            e.innovation = self.edgeIVs[e.idx]
        else:
            e.innovation = self.innovation
            self.edgeIVs[e.idx] = e
            self.innovation += 1
        return e

    def mutate(self, genome:Genome, rng: np.random.Generator):
        #options are new edge, new node, toggle edge
        r = rng.random()
        if r < genome.nodeChance:
            pass
        elif r < genome.nodeChance + genome.edgeChance:
            start: rtNode = rng.choice(genome.nodes)
            end: rtNode = rng.choice(genome.nodes)
            #we can technically have recursion, even to itself
            #but not if it's an input or output (to itself, at least)
            if start.idx == end.idx: #recursion on itself
                if start.isInput or start.isOutput: # no self recursion on inputs or outputs
                    print('No self recursion on shell nodes.')
                    return
            if (start.idx, end.idx) in self.edgeIVs: #this edge already exists
                print('Edge already exists.')
                return
            #we gucci! make a new edge.
            e = self.addEdge(start, end)
            e 
            



            pass
        elif r < genome.nodeChance + genome.edgeChance + genome.toggleChance:
            pass