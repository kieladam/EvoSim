import numpy as np
import copy
from dataclasses import dataclass
import ActivationFunctions as af
from operator import attrgetter

@dataclass
class Gene:
    expression:float = 0.0 #for neurons their expression is their value passed on. Synapses their expression is their weight.
    enabled:bool = True

@dataclass
class Neuron(Gene):
    name:str = ''
    idx:int = 0 #index in the gene list for a genome
    actFunc = None #function used on the neuron's value before passing on as it's expression
    layer:int = 0 #layer in the network. 0 is inputs, outputs are always the last layer.
    output:bool = False #whether a neuron is an output neuron. useful for adjust network shape as new neurons are added.

@dataclass
class Synapse(Gene):
    start:int = 0 #where the synapse starts. should match to a neuron's idx. Synapse takes value from here.
    end:int = 0 #where the synapse ends. should match the a neuron's idx. Synapse deposits it's expression here.
    innovation:int = 0 #the innvation index of the synapse. This is a population-wide list, allowing for speciation. Tracks the 'historical origins' of mutations.

@dataclass
class Genome:
    genes:list[Gene] = None #the genes in a genome. Contains neurons and synapses
    neurons:list[Neuron] = None #a subset of the genes, calculated when it changes. Useful for some functions.
    synapses:list[Synapse] = None #a subset of the genes, calculated when it changes. Useful for some funstions.
    outputLayer:int = 1

class Population:
    def __init__(self, 
                 popCount:int, #total starting biota
                 inputs:int, #sensory inputs for biota
                 outputs:int, #behavioural outputs for biota
                 rng:np.random.Generator, #rng for testing with same seeds. TODO: add a default
                 synDensity:float, #starting density of synapses in genomes. 0-1, 0=none, 1=all
                 debug:bool=False): #print terminal outputs
        self.genomes = [] #each genome is an individual in the population.
        self.innovations = {} #tracked across population, to allow gene crossover.
        self.rng = rng
        self.smallPerturbationChance = 0.9 #chance for synapse weights to update by a small amount on mutation. Otherwise they reset to a random -1 to 1
        self.structuralMutationChances = (0.05, 0.3, 0.2) #chance for new node, new connection, change activation on mutation. Each is checked.
        self.debug = debug
        if synDensity < 0 or synDensity > 1:
            print('Synapse Density should be between 0 and 1. Clamping now.')
            synDensity = np.max(0, np.min(1, synDensity))
        for i in range(popCount):
            g = Genome()
            g.genes = []
            for j in range(inputs):
                n = Neuron(name='input_{j}', idx=len(g.genes))
                n.actFunc = af.InputFunc
                g.genes.append(n)
            for j in range(outputs):
                n = Neuron(idx=len(g.genes))
                n.actFunc = af.randomFunction(self.rng)
                n.layer = 1
                n.name = 'output_{j}'
                n.output = True
                g.genes.append(n)
            #for each input/output pair, randomly create a synapse based on synDensity.
            for j in range(inputs):
                for k in range(outputs):
                    if self.rng.random() < synDensity:
                        s = Synapse(start=j, end=inputs + k)
                        if (s.start, s.end) in self.innovations:
                            #innovations are population-wide
                            s.innovation = self.innovations[(s.start, s.end)]
                        else:
                            self.innovations[(s.start, s.end)] = len(self.innovations)
                            s.innovation = self.innovations[(s.start, s.end)]
                        s.expression = self.rng.uniform(-1.0, 1.0)
                        g.genes.append(s)
            #update the helper subset lists.
            g.synapses = [synapse for synapse in g.genes if isinstance(synapse, Synapse)]
            g.neurons = [neuron for neuron in g.genes if isinstance(neuron, Neuron)]
            self.genomes.append(g)

    def backTrace(self, nIDX:int):
        '''Work backward from a neuron to find it's depth, recursively. Used for adjusting layers.
        Takes a neuron index.'''
        backSynapses = [k for k, v in self.innovations.items() if k[1] == nIDX] #get all the synapses that end with this neuron.
        if not backSynapses: #if there's none, we're done with this.
            return 0
        return 1 + np.max([self.backTrace(s) for s, e in backSynapses]) #otherwise, backtrace the synapses. Recursion!
    
    def updateGenes(self, gIDX:int):
        '''Updates the gene subset lists, sorting the neurons by layer.'''
        g:Genome = self.genomes[gIDX]
        g.synapses = [synapse for synapse in g.genes if isinstance(synapse, Synapse)]
        g.neurons = [neuron for neuron in g.genes if isinstance(neuron, Neuron)]
        g.neurons = sorted(g.neurons, key=attrgetter('layer'))
        if self.debug:
            for i in g.neurons:
                print(f'{i.idx}:{i.layer}')
            print('------')
        
    def addNeuron(self, gIDX:int, synapse:tuple[int, int] = None):
        '''Splits a synapse, adding a new neuron between it's start and end. Defaults to randomly chosen.
        The weights of the new synapses are adjusted so that there is the same expression resulting from the new subgraph.'''
        g:Genome = self.genomes[gIDX]
        n = Neuron(idx=len(g.genes))
        if synapse is None:
            while True:
                s:Synapse = self.rng.choice(g.synapses)
                if s.enabled == False:
                    continue
                break
        else:
            for i in g.synapses:
                if i.start == synapse[0] and i.end == synapse[1]:
                    s:Synapse = i
                    break
        assert s is not None, 'Chosen synapse cannot exist. Are you sure about the start and end neurons?'
        s.enabled = False
        s1 = Synapse(expression=s.expression, start=s.start, end=n.idx)
        self.innovations[(s1.start, s1.end)] = len(self.innovations) #the index of the innovation doubles as the history
        s1.innovation = self.innovations[(s1.start, s1.end)]
        s2 = Synapse(expression=1.0, start=n.idx, end=s.end)
        self.innovations[(s2.start, s2.end)] = len(self.innovations)
        s2.innovation = self.innovations[(s2.start, s2.end)]
        n.layer = g.genes[s.end].layer
        n.actFunc = af.randomFunction(self.rng)
        g.genes.append(n)
        g.genes.append(s1)
        g.genes.append(s2)
        self.updateGenes(gIDX) #updateGenes is called twice, since backtrace needs all the genes to trace...
        for n in g.neurons:
            newlayer = self.backTrace(n.idx)
            if newlayer >= g.outputLayer:
                g.outputLayer += 1
            n.layer = newlayer
        for n in g.neurons:
            if n.output:
                n.layer = g.outputLayer
        self.updateGenes(gIDX) #but then needs to be called again to put them in their newly found order.
        
    def addSynapse(self, gIDX:int):
        '''Can add backwards synapses. These can lead to recursive connections.
        That's the plan, anyway. Disabling them for now.'''
        g:Genome = self.genomes[gIDX]
        loopCut = 0 #loopCut is here as a failsafe, in case there are no valid synapses. 
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
            if loopCut > 50: #it will stop after 50 attempts with no valid new synapses to make.
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
        else:
            print('Could not add valid Synapse.')
    
    def changeActivation(self, gIDX:int):
        '''Changes the activation function of a neuron to a randomly chosen one from a list.'''
        g:Genome = self.genomes[gIDX]
        while True:
            n:Neuron = self.rng.choice(g.neurons)
            if n.layer == 0:
                continue
            break
        g.genes[n.idx].actFunc = af.randomFunction(self.rng)

    def mutate(self, gIDX:int):
        '''All expressions are perturbed, by a small amount (P(0.9)) or a large amount(P(0.1))
        Then there is the chance of a structural mutation happening.'''
        for i in range(len(self.genomes[gIDX].genes)):
            g:Gene = self.genomes[gIDX].genes[i]
            if self.rng.random() < self.smallPerturbationChance:
                g.expression += self.rng.normal(0, 0.3)
            else:
                g.expression = self.rng.uniform(-1.0, 1.0)
        if self.rng.random() < self.structuralMutationChances[0]:
            if self.debug: print('Adding Neuron')
            self.addNeuron(gIDX)
        if self.rng.random() < self.structuralMutationChances[1]:
            if self.debug: print('Trying to add Synapse')
            self.addSynapse(gIDX)
        if self.rng.random() < self.structuralMutationChances[2]:
            if self.debug: print('Changing Activation')
            self.changeActivation(gIDX)

    def reproduce(self, gIDX:int):
        '''Will make a copy of a chosen genome and then mutate it, adding it to the Population.'''
        #TODO: crossover within species 
        newGenome = copy.deepcopy(self.genomes[gIDX])
        newGenome.idx = len(self.genomes)
        self.genomes.append(newGenome)
        self.mutate(newGenome.idx)

    def die(self, gIDX:int):
        '''Sets the index of a genome as None. This could be bad. Need to check genomes are not none when using them.'''
        self.genomes[gIDX] = None
    
    def packageGenome(self, gIDX:int):
        '''Packages a genome's data in a useful way for the visualiser.'''
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
        '''The game-loop part of the brain. 
        Will forward propagate input values and return an array of output values to map to behaviours.'''
        g:Genome = self.genomes[gIDX]
        outputs = []
        for n in g.neurons: #set non-input neuron's expression to zero.
            if n.layer != 0:
                n.expression = 0.0
            for s in g.synapses: #for each node, for each enabled synapse ending at this node...
                if s.end == n.idx and s.enabled:
                    n.expression += g.genes[s.start].expression * s.expression #...multiply by synapse wieght and sum.
            n.expression = n.actFunc(n.expression) #after values are summed, activate!
            if n.layer == g.outputLayer:
                outputs.append(n.expression)
        return outputs
            






