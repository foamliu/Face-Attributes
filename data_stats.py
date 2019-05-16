import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from tqdm import tqdm

from config import pickle_file


def compute_distribution(name):
    print('computing {}...'.format(name))

    x = []
    for sample in tqdm(samples):
        value = sample['attr'][name]
        x.append(value)

    bins = np.linspace(0, 100, 101)

    # the histogram of the data
    plt.hist(x, bins, density=True, alpha=0.5, label='1', facecolor='blue')

    mu = np.mean(x)
    sigma = np.std(x)
    y = scipy.stats.norm.pdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel(name)
    plt.ylabel('{} distribution'.format(name))
    plt.title('Histogram: mu={:.4f}, sigma={:.4f}'.format(mu, sigma))

    plt.savefig('images/{}_dist.png'.format(name))
    plt.grid(True)
    plt.show()


def compute_angle_distribution(name):
    print('computing angle-{}...'.format(name))

    x = []
    for sample in tqdm(samples):
        value = sample['attr']['angle'][name]
        x.append(value)

    bins = np.linspace(-180, 180, 361)

    # the histogram of the data
    plt.hist(x, bins, density=True, alpha=0.5, label='1', facecolor='blue')

    mu = np.mean(x)
    sigma = np.std(x)
    y = scipy.stats.norm.pdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel('angle-{}'.format(name))
    plt.ylabel('angle-{} distribution'.format(name))
    plt.title('Histogram: mu={:.4f}, sigma={:.4f}'.format(mu, sigma))

    plt.savefig('images/angle_{}_dist.png'.format(name))
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']

    # compute_distribution('age')
    # compute_distribution('beauty')

    compute_distribution('pitch')
    compute_distribution('roll')
    compute_distribution('yaw')
