import numpy as np
import pymc3 as pm
from core.analytical_model import Fp2Fs

def MCMC(file_name):
    with pm.Model() as model:
        # load data
        data = np.loadtxt(f'{file_name}.txt', delimiter=',')
        data_X = data[:,0] * 2 * np.pi
        data_Y = data[:,1]

        # Define the prior distributions for the parameters
        AB = pm.Uniform("AB", lower=0, upper=0.7)
        alpha_ellip = pm.Uniform("alpha_ellip", lower=0, upper=0.7)
        beta_ellip = pm.Uniform("beta_ellip", lower=0, upper=0.7)
        sigma = pm.HalfNormal("sigma", sigma=1)


        # Define the likelihood function
        mu = Fp2Fs(data_X, AB, alpha_ellip, beta_ellip)
        likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=data_Y)

        # MCMC采样
        trace = pm.sample(10, tune=10, chains=2)  # 生成4条链，每条2000样本
        
    # 输出后验统计量
    pm.summary(trace).round(2)
    
if __name__ == '__main__':
    MCMC('K2-141b_K2')