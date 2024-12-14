import numpy as np
import rtNeatDataClasses as dc
import ActivationFunctions as af
import Visualisations as vi

rng = np.random.default_rng(seed=42)

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
        self.edges = dict()
        self.edx = dict()
        self.innovation = 0
        for n in self.nodes:
            for m in self.nodes:
                if n.isInput and m.isOutput:
                    if rng.random() < initialEdgeChance:
                        e = dc.rtEdge(inNode=n.idx, outNode=m.idx)
                        e.idx = (e.inNode, e.outNode)
                        e.innovation = self.innovation
                        e.weight = rng.uniform(-1, 1)
                        self.edges[e.idx] = e
                        self.edx[e.idx] = e.innovation
                        self.innovation += 1
        defaultGenome = dc.Genome()
        for n in self.nodes:
            defaultGenome.nodes.append(n.idx)
        for e in self.edges:
            defaultGenome.edges.append(e)
        for i in range(population):
            self.genomes.append(defaultGenome)

    def packageGenome(self, genomeIDX):
        n = dict()
        for i in self.genomes[genomeIDX].nodes:
            node = self.nodes[i]
            n[node.idx] = node.layer
        e = []
        for i in self.genomes[genomeIDX].edges:
            edge = self.edges[i]
            e.append((edge.idx[0], edge.idx[1], edge.weight))
        return (n, e)
    
def main():
    demo = RTNEAT(5, 10, 10, 0.25)
    plt = vi.drawPhenotype(demo.packageGenome(0))
    plt.show()

if __name__ == "__main__":
    main()