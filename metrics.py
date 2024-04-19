import torch
import numpy as np
import math
from sampler import GaussianMixture


def gaussian_kernel(x,h):
   return lambda z: (torch.exp(-torch.sum((x-z)**2,dim=-1)/(2*h**2))/(2*math.pi*h**2)**(x.shape[1]/2)).mean()

def density_estimation(x, kernel):
    return torch.stack([kernel(x[i]) for i in range(x.shape[0])])

def kl_divergence(p, q):
    return (p * (torch.log(p) - torch.log(q))).sum()

@torch.no_grad()
def Kl_measure(sampler,x_gen):  
    h = 0.1
    kernel = gaussian_kernel(x_gen,h)
    rho = density_estimation(x_gen, kernel)
    return kl_divergence(rho+1e-10,sampler.rho0(x_gen)+1e-10).item()

if __name__ == "__main__":
    sampler = GaussianMixture(k=8)
    x_gen = sampler.sample(5000)
    kl= Kl_measure(sampler,x_gen)
    print(f"KL divergence: {kl}")