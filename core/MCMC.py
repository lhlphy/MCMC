import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from core.analytical_model import Fp2Fs
import arviz as az
from multiprocessing import Pool
os.environ["OMP_NUM_THREADS"] = "1"

class MCMC:
    def __init__(self, target_name, file_name,  sigma=10, ndim=3, nwalkers=20, nsteps=1000, burnin=500):
        """
        初始化 MCMC 类。
        
        :param file_name: 数据文件名（不含扩展名)
        :param sigma: 噪声标准差（默认 10 ppm)
        :param ndim: 参数维度（默认 3)
        :param nwalkers: 游走者数量（默认 20)
        :param nsteps: 采样步数（默认 1000)
        :param burnin: 烧入期步数（默认 500)
        """
        self.file_name = file_name
        self.target_name = target_name
        self.sigma = sigma
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.burnin = burnin
        self.labels = ["AB", "alpha_ellip", "alpha_Doppler"]
        
        # 加载数据
        data = np.loadtxt(f'Target\{target_name}\{file_name}.txt', delimiter=',')
        self.data_X = data[:, 0] * 2 * np.pi
        self.data_Y = data[:, 1]

    def log_likelihood(self, params):
        """对数似然函数"""
        AB, alpha_ellip, alpha_Doppler = params
        model = Fp2Fs(self.data_X, AB, alpha_ellip, alpha_Doppler)
        return -0.5 * np.sum((self.data_Y - model) ** 2 / self.sigma**2 + np.log(2 * np.pi * self.sigma**2))
    
    def log_prior(self, params):
        """对数先验函数"""
        AB, alpha_ellip, alpha_Doppler = params
        if 0 <= AB <= 0.7 and 1 <= alpha_ellip <= 10 and 1 <= alpha_Doppler <= 10:
            return 0.0
        return -np.inf
    
    def log_posterior(self, params):
        """对数后验函数"""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)
    
    def sample(self):
        """运行 MCMC 采样并保存样本"""
        # 初始化游走者位置
        initial = np.zeros((self.nwalkers, self.ndim))
        initial[:, 0] = np.random.uniform(0, 0.7, self.nwalkers)  # AB
        initial[:, 1] = np.random.uniform(1, 10, self.nwalkers)   # alpha_ellip
        initial[:, 2] = np.random.uniform(1, 10, self.nwalkers)   # alpha_Doppler
        with Pool(processes=4) as pool:
            # 创建采样器
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, pool=pool)
            sampler.run_mcmc(initial, self.nsteps, progress=True)
            
        # 保存样本（展平后的样本，去除烧入期)
        samples = sampler.get_chain(discard=self.burnin, flat=True)
        np.save(f"Target\{self.target_name}\{self.file_name}_mcmc_samples.npy", samples)
        
        # 保存整个链以便绘制迹线图
        self.chain = sampler.get_chain()
        print("type of chain: ", type(self.chain))

        
        # 计算 Gelman-Rubin 统计量以评估收敛性
        trace = az.from_emcee(sampler)
        self.r_hat = az.rhat(trace)
        print("R_hat for each parameter:", self.r_hat)
        
        return samples
    
    def load_samples(self):
        """加载保存的 MCMC 样本"""
        return np.load(f"{self.file_name}_mcmc_samples.npy")
    
    def plot_fit(self, samples=None, num_samples=100):
        """绘制观测数据与模型预测的拟合图"""
        if samples is None:
            samples = self.load_samples()
        
        inds = np.random.randint(len(samples), size=num_samples)
        plt.figure(figsize=(10, 6))
        for ind in inds:
            sample = samples[ind]
            AB, alpha_ellip, alpha_Doppler = sample
            model_pred = self.Fp2Fs(self.data_X, AB, alpha_ellip, alpha_Doppler)
            plt.plot(self.data_X / (2 * np.pi), model_pred, "C1", alpha=0.1)
        plt.errorbar(self.data_X / (2 * np.pi), self.data_Y, yerr=self.sigma, fmt=".k", capsize=0, label="Data")
        plt.xlabel("Phase (normalized)")
        plt.ylabel("Flux")
        plt.legend()
        plt.savefig(f"Target\{self.target_name}\{self.file_name}_model_fit.png")
        plt.close()
    
    def plot_trace(self):
        """绘制迹线图以检查收敛性"""
        if not hasattr(self, 'chain'):
            print("请先运行 sample() 方法以获取链。")
            return
        
        plt.figure(figsize=(10, 8))
        for i in range(self.ndim):
            plt.subplot(self.ndim, 1, i+1)
            plt.plot(self.chain[:, :, i], "k", alpha=0.3)
            plt.ylabel(self.labels[i])
        plt.xlabel("Step number")
        plt.savefig(f"Target\{self.target_name}\{self.file_name}_trace.png")
        plt.close()
    
    def plot_corner(self, samples=None):
        """绘制角图展示后验分布"""
        if samples is None:
            samples = self.load_samples()
        
        fig = corner.corner(samples, labels=self.labels)
        plt.savefig(f"Target\{self.target_name}\{self.file_name}_corner.png")
        plt.close()
        
    def compute_rhat(self):
        """独立计算 Gelman-Rubin 统计量以评估收敛性"""
        if not hasattr(self, 'chain'):
            print("请先运行 sample() 方法以获取链。")
            return None
        
        # 从 emcee 的 sampler 中创建 ArviZ 兼容的 trace 对象
        trace = az.from_emcee(self.sampler)
        # 计算 Gelman-Rubin 统计量
        r_hat = az.rhat(trace)
        print("R_hat for each parameter:")
        for param, value in r_hat.items():
            print(f"{param}: {value:.3f}")
        return r_hat

# 示例用法
if __name__ == '__main__':
    mcmc = MCMC('K2-141b', 'K2')
    mcmc.sample()       # 采样并保存
    mcmc.plot_fit()     # 绘制拟合图
    mcmc.plot_trace()   # 绘制迹线图
    mcmc.plot_corner()  # 绘制角图
    
    # 加载样本并分析
    samples = mcmc.load_samples()
    print("样本形状:", samples.shape)
    