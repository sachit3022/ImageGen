import os
import tqdm

import torch
from torch import nn as nn

import numpy as np
import skimage as ski

import matplotlib.pyplot as plt
from utils import *
from sampler import GaussianMixture
from models import MLP
from utils import set_seed
from metrics import Kl_measure

#device mps
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')



class ScoreBasedModelling:

    def __init__(self,N, tspan=(0., 1.)):
        self.tspan = tspan
        self.N = N
        self.t_list = torch.linspace(tspan[0], tspan[1], N)
    
    def forward_solve(self, u0, t, type='exact'):
        raise NotImplementedError
    def score_p_x_t_given_x0(self, xt, x0, t_list):
        raise NotImplementedError
    def reverse_solve(self, y, eta_t, tau):
        raise NotImplementedError
    
    def diffuse_samples(self, u0):
        t_index = torch.randint(0, self.N, (u0.shape[0],))

        xt = self.forward_solve(u0, t_index.reshape(-1,1), type='exact')
        y = self.score_p_x_t_given_x0(xt, u0, t_index.reshape(-1,1))

       

        t_list = self.t_list[t_index]
        #normalisation of time
        t_nn = (t_list - t_list.min()) / (t_list.max() - t_list.min())
        t_nn = 2 * t_nn - 1
        ut = torch.hstack([xt, t_nn[:,None]])
        l_t = (1 - torch.exp(-2*t_list))/2 #l_t = torch.ones_like(t_list) # self.one_minus_alpha_t_bar[t_index] #self.one_minus_alpha_t_bar[t_index] #self.one_minus_alpha_t_bar[t_index]
        
        return ut, y,l_t
    
    @torch.no_grad()
    def generate(self,P,score_model):
        y0 = torch.randn((P, 2), requires_grad=False)
        
        tau = self.tspan[-1] / self.N # step size
        t = torch.Tensor([self.tspan[-1]])
 
        y = torch.zeros((P, 2, self.N), requires_grad=False)
        y[:,:,0] = y0

        for i in tqdm.tqdm(range(self.N-1)):

            t_temp = ((t.repeat(P, 1) - self.tspan[0]) / (self.tspan[1] - self.tspan[0])) 
            t_temp = 2*t_temp - 1
            u = torch.hstack([y[:,:,i], t_temp])
            eta_t = score_model(u)
            y[:,:,i+1] = self.reverse_solve( y[:,:,i], eta_t,i)
            t -= tau
        return y[:,:,-1]


class FwdOrnsteinUhlenbeckProcess(ScoreBasedModelling):
    def __init__(self, N, tspan=(0., 1.)):
        super().__init__(N, tspan)

    def forward_solve(self, u0, t, type='exact'):
        t = self.t_list[t]
        w = torch.randn(u0.shape)
        if type == 'exact':
            return u0 * np.exp(-t) + np.sqrt(1 - np.exp(-2*t)) * w
        if type == 'sde':
            out = u0
            tau  = t / self.N
            for _ in range(self.N):
                out = out - tau * out + np.sqrt(2 * tau) * w
        return out 

    def score_p_x_t_given_x0(self, xt, x0, t_index):
        t_list = self.t_list[t_index]
        return (x0 * torch.exp(-1*t_list) - xt)/ (1 - torch.exp(-2*t_list))
    
    def reverse_solve(self, y, eta_t, i):
        beta = 0.1
        tau = self.tspan[1] / self.N
        w = torch.normal(0,1,(y.shape))
        const_temp1 = torch.Tensor([1+beta])
        const_temp2 = torch.Tensor([tau * beta])
        return y + tau*(y + const_temp1*eta_t) + (2*const_temp2)**(0.5)*w

class DDPMProcess(ScoreBasedModelling):
    def __init__(self, N, tspan=(0., 1.)):
        super().__init__(N, tspan)
        self.betas =  self.constant_beta_schedule()
        alpha_t = 1 -  self.betas
        alpha_t_bar = torch.cumprod(alpha_t, 0)
        self.sqrt_alpha_t_bar = torch.sqrt(alpha_t_bar)
        self.one_minus_alpha_t_bar = 1. - alpha_t_bar
    def constant_beta_schedule(self):
        return torch.ones((self.N,))* 0.02 
    
    def linear_beta_schedule(self):
        beta_start = 0.01
        beta_end = 0.03
        return torch.linspace(beta_start, beta_end, self.N)
    
    def sigmoid_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, self.N)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


    def cosine_beta_schedule(self, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        timesteps = self.N
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def score_p_x_t_given_x0(self, xt, x0, t_list):
        alpha_bar = self.sqrt_alpha_t_bar[t_list]
        one_minus_alpha_t_bar = self.one_minus_alpha_t_bar[t_list]
        return (alpha_bar*x0 - xt)/(one_minus_alpha_t_bar)

    def forward_solve(self, u0, t_list, type='exact'):
        w = torch.randn(u0.shape)
        alpha_bar = self.sqrt_alpha_t_bar[t_list]
        one_minus_alpha_t_bar = self.one_minus_alpha_t_bar[t_list]
        return alpha_bar*u0 + torch.sqrt(one_minus_alpha_t_bar)*w
    
    def reverse_solve(self, y, eta_t, i):
        w = torch.normal(0,1,(y.shape))
        beta_i = self.betas[i]
        const_beta_1 = 1. /torch.Tensor([(1-beta_i)**(0.5)])
        return const_beta_1*(y+beta_i*eta_t) + (beta_i)**(0.5)*w
    
class Trainer():
    def __init__(self, model, device):
        self.model = model.to(device)
        self.optimizer =  torch.optim.AdamW(self.model.parameters(), lr=1e-2)
        self.loss = nn.MSELoss(reduction='none').to(device)
        self.device = device

    def train(self, u, y, l_t, num_epochs):
        u, y, l_t = u.to(self.device), y.to(self.device), l_t.to(self.device)
        self.model.train()
        loss_list = torch.zeros(num_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-5)
        for i in tqdm.tqdm(range(num_epochs)):
            self.optimizer.zero_grad()
            y_pred = self.model(u)
            l = ((self.loss(y_pred, y).sum(dim=-1)*l_t)/l_t.sum()).sum()
            l.backward()
            self.optimizer.step()
            loss_list[i] = l.item()
            scheduler.step()
        self.model.eval()
        return loss_list


if __name__ == "__main__":
    N = 20000
    tspan = (1e-2,5.)
    N_steps = 500
    

    # set seed
    set_seed(42)

    sampler = GaussianMixture(k=8)
    diff_process  = FwdOrnsteinUhlenbeckProcess(N_steps,tspan)
    #diff_process = DDPMProcess(N_steps,tspan)
    
    
    ########## Generate samples ##########
    x0 = sampler.sample(N)
    plot_density_from_samples(x0, filepath='score_plots/1-samples.png', show=False, save=True)
    u, y, l_t = diff_process.diffuse_samples(x0)
    
    
    ########## Make model ##########

    d_in = 3
    d_hidden = 64
    n_blocks = 3
    d_out = 2
    num_epochs = 1000
    model = MLP(d_in, d_hidden, d_out,n_blocks)

    ####### Train model ########

    
    trainer = Trainer(model,device=DEVICE)
    loss_list = trainer.train(u, y, l_t, num_epochs)
    plot_loss(loss_list, 'score_plots/2-loss.png')

    ########## Generate samples ##########

    P = 5000 
    model = model.to('cpu')
    x_gen = diff_process.generate(P, model)

    ########## Measure the quality of generation ##########
    kl = Kl_measure(sampler,x_gen)
    print(f'KL divergence: {kl}')

    ########## Plot the results ##########
    plot_density_from_samples(x_gen.detach().numpy(), filepath=f'score_plots/3-reverse-density-final.png', show=True, save=True)



    """
    #for transformer
    x0 = gmm.sample(N)
    t_list = torch.randint(1,2501,(N,))
    t_temp = t_list/2500.0
    w = torch.randn((N, 2))
    temp = t_temp.repeat(2,1).T
    xt = torch.exp(-temp) * x0 + torch.sqrt(1 - torch.exp(-2*temp)) * w
    u = xt 
    y = (x0 * torch.exp(-1*t_temp.reshape(-1,1)) - xt)/ (1 - torch.exp(-2*t_temp.reshape(-1,1)))
    l_t =   torch.ones_like(t_list)#(1 - torch.exp(-2*t_temp))/2 
    model = Transformer(2, 32, 4)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
num_epochs = 100
loss_list = torch.zeros(num_epochs)

for i in tqdm.tqdm(range(num_epochs)):
    ls = 0
    for j in range(30):
        batch = torch.randint(0, N, (500,))
        u_b, y_b, l_tb,t_b = u[batch], y[batch], l_t[batch],t_list[batch]
        optimizer.zero_grad()
        y_pred = model(u_b,t_b)
        l = loss(y_pred, y_b)
        l.backward()
        optimizer.step()
        ls+=l.item()

    loss_list[i] = ls/30
    scheduler.step()

    # Let us implement the reverse Euler-Maruyama scheme.
model.eval()
with torch.no_grad():
    P = 500 # number of particles
    y0 = torch.randn((P, 2), requires_grad=False)

    N = 2500 # number of steps
    T =  1 # final time
    tau = T / N # step size

    t = torch.Tensor([T])
    beta = 0.1

    const_temp1 = torch.Tensor([1+beta])
    const_temp2 = torch.Tensor([tau * beta])

    y = torch.zeros((P, 2, N+1), requires_grad=False)
    for i in tqdm.tqdm(range(1,N)):
        t_temp = torch.ones(P,dtype=torch.long)*(N-i)
        eta_t = model(y[:,:,i],t_temp)
        # ------------
        # FILL THIS IN - START
        # ------------
        y[:,:,i+1] = y[:,:,i] + tau*(y[:,:,i] + (1+beta)*eta_t) + (2*tau*beta)**(0.5)*torch.normal(0,1,(P,2))
        t -=  tau
        # ------------
        # FILL THIS IN - END
    # ------------
    """











