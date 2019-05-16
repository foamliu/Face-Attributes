import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from tqdm import tqdm

from config import pickle_file

if __name__ == "__main__":
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']

    beauties = []
    for sample in tqdm(samples):
        beauty = sample['attr']['beauty']
        beauties.append(beauty)

    bins = np.linspace(0, 100, 101)

    # the histogram of the data
    plt.hist(beauties, bins, density=True, alpha=0.5, label='1', facecolor='blue')

    mu = np.mean(beauties)
    sigma = np.std(beauties)
    y = scipy.stats.norm.pdf(bins, mu, sigma)
    plt.plot(bins, y, 'b--')
    plt.xlabel('beauty')
    plt.ylabel('beauty distribution')
    plt.title(r'Histogram: mu={:.4f},sigma={:.4f}'.format(mu, sigma))

    plt.savefig('images/beauty_dist.png')
    plt.grid(True)
    plt.show()
