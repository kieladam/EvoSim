import numpy as np
import rtNeat.rtNeatImp as rfN
import Visualisations as vi

rng = np.random.default_rng(seed=42)
    
def main():
    demo = rfN.Population(popCount=10, inputs=5, outputs=5, rng=rng, synDensity=0.1)
    viewGenome = 0
    plt = vi.drawPhenotype(demo.packageGenome(viewGenome))
    plt.show()
    

if __name__ == "__main__":
    main()