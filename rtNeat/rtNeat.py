import numpy as np
import rtNeatDataClasses as dc
import ActivationFunctions as af
import Visualisations as vi


class RTNEAT:
    #default init creates a fully connected network with no hidden layers. maybe a sparcer initial network is better?
    #papers show areound 0.25 chance to initialise each edge.
    def __init__(self, inputs, outputs, population, initialEdgeChance):
        self.genomes = []
        self.nodes: dc.rtNode = []
        for i in range(inputs):
            n = dc.rtNode(len(self.nodes), isInput=True)
            n.layer = 0
            self.nodes.append(n)
        for i in range(outputs):
            n = dc.rtNode(len(self.nodes), isOutput=True)
            n.actFunc = af.randomFunction()
            n.layer = 1
            self.nodes.append(n)
        self.edges: dc.rtEdge = []
        self.edx = dict()
        self.innovation = 0
        for n in self.nodes:
            for m in self.nodes:
                if n.isInput and m.isOutput:
                    if np.random.random() < initialEdgeChance:
                        e = dc.rtEdge(inNode=n.idx, outNode=m.idx)
                        e.idx = (e.inNode, e.outNode)
                        e.innovation = self.innovation
                        e.weight = np.random.uniform(-1, 1)
                        self.edges.append(e)
                        self.edx[e.idx] = e.innovation
                        self.innovation += 1
        defaultGenome = dc.Genome()
        for n in self.nodes:
            defaultGenome.nodes.append(n.idx)
        for e in self.edges:
            defaultGenome.edges.append(e)
        for i in range(population):
            self.genomes.append(defaultGenome)

demo = RTNEAT(5, 10, 10, 0.25)
layers = dict()
for n in demo.nodes:
    layers[n.idx] = n.layer
weights = []
for e in demo.edges:
    weights.append(e.weight)
plt = vi.draw_directed_graph(demo.edx.keys(), weights, layers)
plt.show()