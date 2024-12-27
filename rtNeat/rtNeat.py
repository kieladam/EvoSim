import numpy as np
import refactoredNEAT as rfN
import Visualisations as vi

rng = np.random.default_rng(seed=42)
    
def main():
    demo = rfN.Population(popCount=10, inputs=1, outputs=1, rng=rng, synDensity=1)
    viewGenome = 0
    plt = vi.drawPhenotype(demo.packageGenome(viewGenome))
    plt.show()
    demo.genomes[0].genes[0].expression = 2.3
    print(demo.genomes[0].synapses)
    print(demo.evaluate(0))
    demo.addNeuron(0)
    demo.mutate(0)
    print(demo.genomes[0].synapses)
    print(demo.genomes[0].genes[1].actFunc)
    print(demo.evaluate(0))
    plt = vi.drawPhenotype(demo.packageGenome(viewGenome))
    plt.show()

if __name__ == "__main__":
    main()