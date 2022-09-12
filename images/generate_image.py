from cProfile import label
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    x = [25000, 50000, 100000, 250000, 500000, 1000000]
    
    nativeVis = [102.97, 29.3, 7.8, 1.3, 0, 0]
    nativeNonvis = [112.98, 30.15, 7.83, 1.3, 0, 0]

    scatteredVis = [1122.4, 665.6, 503.43, 69.44, 22.39, 7.82]
    scatteredNonvis = [1686.73, 1000.49, 654.64, 74.09, 23.95, 7.89]

    coherentVis = [1260.61, 897.67, 859.73, 479.44, 310.48, 190.69]
    coherentNonvis = [1897.59, 1236.93, 1247.28, 741.93, 440.62, 242.66]

    fig = plt.figure()
    plt.plot(x, coherentVis, "g", label="Coherent With Visualization ")
    plt.plot(x, scatteredVis, "b", label="Scattered With Visualization ")
    plt.plot(x, nativeVis, "r", label="Native With Visualization ")

    plt.plot(x, coherentNonvis, "g--", label="Coherent")
    plt.plot(x, scatteredNonvis, "b--", label="Scattered")
    plt.plot(x, nativeNonvis, "r--", label="Native")

    plt.legend()

    plt.title("FPS vs Number of Boids")
    plt.ylabel('FPS')
    plt.xlabel("Number of Boids")

    x = [5, 6, 7, 8, 9, 10]
    y1 = [166.83, 242.10, 259.3, 272.09, 258.91, 258.86]
    y2 = [1171.7, 1430.62, 1486.67, 1365.3, 1457.6, 1426.61]

    fig = plt.figure()
    plt.plot(x, y1, label="1M Boids Without Visualization")
    plt.plot(x, y2, label="100K Boids Without Visualization")
    
    plt.legend()

    plt.title("FPS vs Block Size")
    plt.ylabel('FPS')
    plt.xlabel("Block Size 2^(x)")

    x = [10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    y = [2069.2, 1983.6, 1898.64, 1615.14, 929.8, 601.04, 354.63]
    coherentNonvis = [2120, 2026.9, 1345.29, 1512.05, 795, 480.94, 268.7]


    fig = plt.figure()
    plt.plot(x, coherentNonvis, label="8 neighboring cells")
    plt.plot(x, y, label="27 neighboring cells")
    
    plt.legend()

    plt.title("FPS vs Block Size")
    plt.ylabel('FPS')
    plt.xlabel("Block Size 2^(x)")

    plt.show()