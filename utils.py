
# plotting utils

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_density_from_samples(samples, filepath='gmm-density-samples.png', show=True, save=True):
    fig = plt.figure()
    sns.kdeplot(x=samples[:, 0], y=samples[:, 1], cmap="Reds", fill=True, thresh=0, bw_adjust=0.5)
    plt.title('Gaussian Mixture Model Samples')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    if show:
        plt.show()
    else:
        plt.close(fig)
    if save:
        fig.savefig(filepath)
    
def plot_density(samples, rho, filepath='gmm-density.png', show=True, save=True):
    fig = plt.figure()
    sns.scatterplot(x=samples[:, 0], y=samples[:, 1], hue=rho, palette='Reds')
    plt.title('Gaussian Mixture Model Density')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    if show:
        plt.show()
    else:        
        plt.close(fig)
    if save:
        fig.savefig(filepath)
    
def plot_score_function(samples, eta, filepath='gmm-score-function.png', show=True, save=True):
    fig = plt.figure()
    plt.quiver(samples[:, 0], samples[:, 1], eta[:, 0], eta[:, 1])
    plt.title('Gaussian Mixture Model Score Function')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    if show:
        plt.show()
    else:
        plt.close(fig)
    if save:
        fig.savefig(filepath)
        
def plot_loss(losses, filepath='gmm-loss.png', show=True, save=True):
    fig = plt.figure()
    plt.plot(losses)
    plt.title('Score Function Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    if show:
        plt.show()
    else:
        plt.close(fig)
    if save:
        fig.savefig(filepath)