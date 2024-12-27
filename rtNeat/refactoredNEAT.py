import numpy as np
import copy
from dataclasses import dataclass
import ActivationFunctions as af
from operator import attrgetter

@dataclass
class Gene:
    expression:float = 0.0
    enabled:bool = True

@dataclass
class Neuron(Gene):
    name:str = ''
    idx:int = 0
    actFunc = None
    layer:int = 0
    output:bool = False

@dataclass
class Synapse(Gene):
    start:int = 0
    end:int = 0
    innovation:int = 0

@dataclass
class Genome:
    genes:list[Gene] = None
    neurons:list[Neuron] = None
    synapses:list[Synapse] = None
    outputLayer:int = 1

class Population:
    def __init__(self, popCount:int, inputs:int, outputs:int, rng:np.random.Generator, synDensity:float, debug:bool=False):
        self.genomes = []
        self.innovations = {}
        self.rng = rng
        self.smallPerturbationChance = 0.9
        self.structuralMutationChances = (0.05, 0.3, 0.2) #new node, new connection, change activation
        self.debug = debug
        if synDensity < 0 or synDensity > 1:
            print('Synapse Density should be between 0 and 1. Clamping now.')
            synDensity = np.max(0, np.min(1, synDensity))
        for i in range(popCount):
            g = Genome()
            g.genes = []
            for j in range(inputs):
                n = Neuron(name='input{j}', idx=len(g.genes))
                n.actFunc = af.InputFunc
                g.genes.append(n)
            for j in range(outputs):
                n = Neuron(idx=len(g.genes))
                n.actFunc = af.randomFunction(self.rng)
                n.layer = 1
                n.name = 'output{j}'
                n.output = True
                g.genes.append(n)
            for j in range(inputs):
                for k in range(outputs):
                    if self.rng.uniform(0, 1.0) < synDensity:
                        s = Synapse(start=j, end=inputs + k)
                        if (s.start, s.end) in self.innovations:
                            s.innovation = self.innovations[(s.start, s.end)]
                        else:
                            self.innovations[(s.start, s.end)] = len(self.innovations)
                            s.innovation = self.innovations[(s.start, s.end)]
                        s.expression = self.rng.uniform(-1.0, 1.0)
                        g.genes.append(s)
            g.synapses = [synapse for synapse in g.genes if isinstance(synapse, Synapse)]
            g.neurons = [neuron for neuron in g.genes if isinstance(neuron, Neuron)]
            self.genomes.append(g)

    def backTrace(self, nIDX:int):
        backSynapses = [k for k, v in self.innovations.items() if k[1] == nIDX]
        if not backSynapses:
            return 0
        return 1 + np.max([self.backTrace(s) for s, e in backSynapses])
    
    def updateGenes(self, gIDX:int):
        g:Genome = self.genomes[gIDX]
        g.synapses = [synapse for synapse in g.genes if isinstance(synapse, Synapse)]
        g.neurons = [neuron for neuron in g.genes if isinstance(neuron, Neuron)]
        g.neurons = sorted(g.neurons, key=attrgetter('layer'))
        if self.debug:
            for i in g.neurons:
                print(f'{i.idx}:{i.layer}')
            print('------')
        
    def addNeuron(self, gIDX:int):
        '''Cannot add backwards links; only splits existing synapses.'''
        g:Genome = self.genomes[gIDX]
        n = Neuron(idx=len(g.genes))
        while True:
            s:Synapse = self.rng.choice(g.synapses)
            if s.enabled == False:
                continue
            break
        s.enabled = False
        s1 = Synapse(expression=s.expression, start=s.start, end=n.idx)
        self.innovations[(s1.start, s1.end)] = len(self.innovations)
        s1.innovation = self.innovations[(s1.start, s1.end)]
        s2 = Synapse(expression=1.0, start=n.idx, end=s.end)
        self.innovations[(s2.start, s2.end)] = len(self.innovations)
        s2.innovation = self.innovations[(s2.start, s2.end)]
        n.layer = g.genes[s.end].layer
        n.actFunc = af.randomFunction(self.rng)
        g.genes.append(n)
        g.genes.append(s1)
        g.genes.append(s2)
        self.updateGenes(gIDX)
        for n in g.neurons:
            newlayer = self.backTrace(n.idx)
            if newlayer >= g.outputLayer:
                g.outputLayer += 1
            n.layer = newlayer
        for n in g.neurons:
            if n.output:
                n.layer = g.outputLayer
        self.updateGenes(gIDX)
        
    def addSynapse(self, gIDX:int):
        '''Can add backwards synapses. These can lead to recursive connections.
        That's the plan, anyway. Disabling them for now.'''
        g:Genome = self.genomes[gIDX]
        loopCut = 0
        while True:
            loopCut += 1
            n1:Neuron = self.rng.choice(g.neurons)
            n2:Neuron = self.rng.choice(g.neurons)
            if n1.layer >= n2.layer: #start is same or greater than end
                continue
            '''if (n1.layer == 0 or n1.layer == g.outputLayer) and n1.layer == n2.layer:
                continue
            if n2.layer == 0:
                continue'''
            if loopCut > 50:
                break
            if (n1.idx, n2.idx) in self.innovations: #synapse already exists
                continue
            break
        if loopCut <= 50:
            s = Synapse(expression=self.rng.uniform(-1.0, 1.0), start=n1.idx, end=n2.idx)
            self.innovations[(s.start, s.end)] = len(self.innovations)
            s.innovation = self.innovations[(s.start, s.end)]
            g.genes.append(s)
            self.updateGenes(gIDX)
            print('Added Synapse')
    
    def changeActivation(self, gIDX:int):
        g:Genome = self.genomes[gIDX]
        while True:
            n:Neuron = self.rng.choice(g.neurons)
            if n.layer == 0:
                continue
            break
        g.genes[n.idx].actFunc = af.randomFunction(self.rng)

    def mutate(self, gIDX:int):
        #All expressions are perturbed, by a small amount (P(0.9)) or a large amount(P(0.1))
        #Then there is the chance of a structural mutation happening.
        for i in range(len(self.genomes[gIDX].genes)):
            g:Gene = self.genomes[gIDX].genes[i]
            if self.rng.uniform(0.0, 1.0) < self.smallPerturbationChance:
                g.expression += self.rng.normal(0, 0.3)
            else:
                g.expression = self.rng.uniform(-1.0, 1.0)
        if self.rng.uniform(0.0, 1.0) < self.structuralMutationChances[0]:
            print('Added Neuron')
            self.addNeuron(gIDX)
        if self.rng.uniform(0.0, 1.0) < self.structuralMutationChances[1]:
            print('Trying to add Synapse')
            self.addSynapse(gIDX)
        if self.rng.uniform(0.0, 1.0) < self.structuralMutationChances[2]:
            print('Changed Activation')
            self.changeActivation(gIDX)

    def reproduce(self, gIDX:int):
        #TODO: crossover within species 
        newGenome = copy.deepcopy(self.genomes[gIDX])
        newGenome.idx = len(self.genomes)
        self.genomes.append(newGenome)
        self.mutate(newGenome.idx)

    def die(self, gIDX:int):
        self.genomes[gIDX] = None
    
    def packageGenome(self, gIDX:int):
        n = dict()
        neurons = [neuron for neuron in self.genomes[gIDX].genes if isinstance(neuron, Neuron)]
        for neuron in neurons:
            n[neuron.idx] = neuron.layer
        e = []
        synapses = [synapse for synapse in self.genomes[gIDX].genes if isinstance(synapse, Synapse)]
        for synapse in synapses:
            if synapse.enabled:
                e.append((synapse.start, synapse.end, synapse.expression, synapse.innovation))
        return (n, e)
    
    def evaluate(self, gIDX:int) -> list[float]:
        g:Genome = self.genomes[gIDX]
        outputs = []
        for n in g.neurons:
            if n.layer != 0:
                n.expression = 0.0
            for s in g.synapses:
                if s.end == n.idx and s.enabled:
                    n.expression += g.genes[s.start].expression * s.expression
            n.expression = n.actFunc(n.expression)
            if n.layer == g.outputLayer:
                outputs.append(n.expression)
        return outputs
            






