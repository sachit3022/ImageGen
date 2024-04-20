[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/q_3tyBp7)

# Writeup


## Metric
The goal of this assesment is to understand various generative models. First question we need to address is how can we compare the samples generated by two methods. We defined a metric based on KL-divergence.  Why loss is not enough? If the training samples are less the loss will be smaller but even the validation loss is not right to compare. Lets give you simple example, Compare loss of Score and diffsion models. $L_{DDPM} = L_{Score} -C_1 + C_2$. where $C_1$
and $C_2$ are constants. 

We cannot used FD ( Frechet Distance ), which is a common metric for Images and it is not applicable in out case  as it assumes normality of the datapoints which is violated for GMMs.

### KL Distance 
$$p_{estimate}(x) = \frac{1}{hN}\Sigma_i^N  K(\frac{x - x_i}{h})n$$
$$q_{true}(x) = \Sigma \alpha_i \mathcal{N}(\mu_i, \sigma_i)$$
$$KLD = KL(q||p)$$
where K is any kernel, here we assume gaussian kernel for simplicity. Here the idea is that we are estimating the probabilty or desinty of the gnerated samples by gaussian kernel and comparing the actual desnity of the x given by q, which is mixture of gaussian. Smaller it is better the generation quality. 

Why are the generative model papers not using this metric. Here we are making huge assumption of estimating likelihood by kernel density. This holds only true for large sample size. For our example as it is a toy data this suits the purpose and we have a large sample size.


## Score based models & DDPM


In this problem we are solving an SDE of the form
$$dx_t = f(x,t)dt + g(t)dw$$
$$dx_t = [f(x,t) - g^2(t)\nabla_{x_t}\log p_t(x_t)]dt + g(t)d\bar{w}$$

We can have any function f and g and solve the SDE in the forward and then sample in the reverse direction.

Generally, we dont have the closed form for $\nabla_{x_t}\log p_t(x_t)$, In this example we have because we know the distribution of $x_0$, which is mixture of gaussian and the also $x_t = x_0  \ast \mathcal{N}(\mu,\sigma)$. However, we wil assume general case in which the distribution of $x_0$ is unknown, for such cases  $\nabla_{x_t}\log p_t(x_t)$ is not tractable.



One advantage we have is $p(x_t|x_0)$ is tractable.
$$\log p(x_t) = \log \mathop{\mathbb{E}}_{x_0} p(x_t|x_0)*p(x_0)$$
$$\log p(x_t) \ge \mathop{\mathbb{E}}_{x_0} [\log p(x_t|x_0)*p(x_0)]$$
$$\nabla_{x_t} \log p(x_t) \ge  \nabla_{x_t} \mathop{\mathbb{E}}_{x_0} [\log p(x_t|x_0)+ \log p(x_0)]$$
$$\nabla_{x_t} \log p(x_t) \ge\mathop{\mathbb{E}}_{x_0} [  \nabla_{x_t} \log p(x_t|x_0)]$$

Based on ELBO property we can replace the probabilty with the conditional probability and an expectation. 

We can derive the same thing as derived in the score based or energy based models, levaraging for small $\sigma$, $E_{q\sigma}(f(\tilde{x}) - \nabla\log({q_\sigma{\tilde{x}}}) ) = E_{p}(f(x) - \nabla\log({p(x)}) )$ and from here we can show that the E can be relplaced with conditionals by re arranging terms.


What is the performance without learning as we have true distributions p(x_t), we can have score in closed form and we can generate samples from these distributions. This will serve as a benchmark to aim at.
<center>

| Sampling   |   KL measure |
|----------|:-------------:|
| p(x) |  25.13 |

</center>



Now we know the closed form of the conditional in forward pass and in the reverse direction we replace $\nabla_{x_t}\log p_t(x_t)$ with $s_{\theta}(x,t)$.


During training we train with a score matching loss
$$L = \mathop{\mathbb{E}}_{x_0}\mathop{\mathbb{E}}_{t \sim U[0,T]} [\lambda(t)||s_{\theta}(x,t) -\nabla_{x_t}\log p_t(x_t)||_2^2 ]$$




Where can Improvements come from? 

There are three imprortant parameters in the score based models the improvements can come

1. Choice of SDE
2. Choice of $\lambda(t)$
3. Choice of model for $s_{\theta}(x,t)$


### Choice of SDE

In the paper DDPM, Ho et.al,  has shown that score models and denoising diffusion models are the same but with a different form for SDE. 

In this we will only write the SDE and the forward process, the backward for the following is avaliable in the code.

In FwdOrnstein-Uhlenbeck Process the forward SDE is given by 
$$f(x_t,t) = -x_t, g(t) = \sqrt{2}$$
$$x_t = e^{-t}x_0 + \sqrt{1- e^{-2t}} \epsilon$$

It can be shown the DDPM is shown as 

$$f(x_t,t) = \frac{-\beta_t}{2}, g(t) = \sqrt{\beta_t}$$
$$x_t =  \sqrt{1- \alpha_t}x_0 + \sqrt{\alpha_t} \epsilon$$

We can try multiple different $f(x_t,t),g(t)$ However, I donot have background in solving SDE. So  Here we show that using DDPM will improve the KLD metric.

### Choice of Weigting Loss function

The derivation of score will have a unfirom weight to the score matching accross all time steps. However, Ho et.al found that asining higher weight to larger variance score will improve the model performance. Indeed we also observe the similar performance. 

Ho et.al recommends
$$\lambda(t) \propto \frac{1}{\mathop{\mathbb{E}}_{x_0}(||-\nabla_{x_t}\log p_t(x_t|x_0)||_2^2)}$$

We will derive this for both DDPM and OU process.


DDPM 
$$\mathop{\mathbb{E}}_{x_0}(-\nabla_{x_t}\log p_t(x_t|x_0)) = \frac{\sqrt{1- \alpha_t} x_0 - x_t}{\sigma_i^2}$$
$$\mathop{\mathbb{E}}_{x_0}(-\nabla_{x_t}\log p_t(x_t|x_0)) = ||\frac{\sqrt{1- \alpha_t}x_0 - x_t}{(1- \alpha_t)}||_2^2$$
$$\mathop{\mathbb{E}}_{x_0}(-\nabla_{x_t}\log p_t(x_t|x_0)) = ||\frac{\epsilon}{\sqrt(1- \alpha_t)}||_2^2$$
$$\lambda(t) = \frac{1- \alpha_t}{2}$$

OU process, it s very similar the $\lambda(t)$ is nothing more than the variance schedule.

The variance schedule of OU process is 

$$\lambda(t) = \frac{1 - e^{-2t}}{2}$$


### choice of model

We can vary the size af the models and also bring changes to the architecture, We didnot find much grains by improving the depth of the model as well as adding batch norm or residual connections. One of the abbilation studies we can perform is to change the architectures from MLP to transformers and embedding time as a sinusoidal positional embeddings. However, we have observed this setup didnot yield much improvements.


## Results 

<p align="center">
  <table>
    <tr>
      <th>Diffusion process</th>
      <th>Score weight</th>
      <th>KLD metric</th>
    </tr>
    <tr>
      <td>FwdOrnstein-Uhlenbeck</td>
      <td>Uniform</td>
      <td>33.61</td>
    </tr>
    <tr>
      <td>FwdOrnstein-Uhlenbeck</td>
      <td>Inv Variance Schedule</td>
      <td>32.24</td>
    </tr>
    <tr>
      <td>DDPM</td>
      <td>Uniform</td>
      <td><b>28.04</\b></td>
    </tr>
    <tr>
      <td>DDPM</td>
      <td>Inv Variance Schedule</td>
      <td>28.62</td>
    </tr>
  </table>
</p>

The final distribution plot across different settings along with ground truth.
<table style="border-collapse: collapse;">
    <tr>
        <td colspan="2" style="border: 1px solid black; padding: 10px; text-align: center;"><img src="assets/1-samples.png" alt="Image 1"><br>Ground truth distribution</td>
    </tr>
    <tr>
        <td style="border: 1px solid black; padding: 10px; text-align: center;"><img src="assets/1-reverse-density-final.png" alt="Image 2"><br>FwdOrnstein-Uhlenbeck Uniform</td>
        <td style="border: 1px solid black; padding: 10px; text-align: center;"><img src="assets/2-reverse-density-final.png" alt="Image 3"><br>FwdOrnstein-Uhlenbeck Inv variance weight</td>
    </tr>
    <tr>
        <td style="border: 1px solid black; padding: 10px; text-align: center;"><img src="assets/3-reverse-density-final.png" alt="Image 4"><br>DDPM uniform</td>
        <td style="border: 1px solid black; padding: 10px; text-align: center;"><img src="assets/4-reverse-density-final.png" alt="Image 5"><br>DDPM Inv variance weight</td>
    </tr>
</table>

## VAE & Hierachial VAE & Learned variance DDPM



### Future work

1. Implement Hypenetworks condition on sinusodial time embeddings.
2. Work on conditional generation of mode.

### References
1. Chan, S. H. (2024). Tutorial on Diffusion Models for Imaging and Vision. arXiv preprint arXiv:2403.18103. https://arxiv.org/pdf/2403.18103.pdf.
2. Weng, Lilian. (Jul 2021). What are diffusion models? Lil’Log. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.
3. Vishnu Boddeti. (2024). Deep Learning. https://hal.cse.msu.edu/teaching/2024-spring-deep-learning/
4. Arash Vahdat. et al. (2022). CVPR. https://cvpr2022-tutorial-diffusion-models.github.io/