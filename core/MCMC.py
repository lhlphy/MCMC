import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from core.analytical_model import Fp2Fs
import arviz as az
from multiprocessing import Pool
os.environ["OMP_NUM_THREADS"] = "1"
from scipy.stats import truncnorm

class MCMC:
    def __init__(self, target_name, file_name,  sigma=10, ndim=4, nwalkers=120, nsteps=2000, burnin=1000):
        """
        初始化 MCMC 类。
        
        :param file_name: 数据文件名（不含扩展名)
        :param sigma: 噪声标准差（默认 10 ppm)
        :param ndim: 参数维度（默认 3)
        :param nwalkers: 游走者数量
        :param nsteps: 采样步数
        :param burnin: 烧入期步数
        """
        self.file_name = file_name
        self.target_name = target_name
        self.sigma = sigma
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.burnin = burnin
        self.labels = ["A", "alpha_ellip", "alpha_Doppler", "F"]
        
        # 加载数据, 使用 os.path.join 构建跨平台的文件路径
        path = os.path.join('Target', target_name, f'{file_name}.txt')
        data = np.loadtxt(path, delimiter=',')
        self.data_X = data[:, 0] * 2 * np.pi
        self.data_Y = data[:, 1]

    def log_likelihood(self, params):
        """对数似然函数"""
        AB, alpha_ellip, alpha_Doppler, F = params
        model = Fp2Fs(self.data_X, AB, alpha_ellip, alpha_Doppler, F)
        return -0.5 * np.sum((self.data_Y - model) ** 2 / self.sigma**2 + np.log(2 * np.pi * self.sigma**2))
    
    def log_prior(self, params):
        """对数先验函数"""
        AB, alpha_ellip, alpha_Doppler, F = params
        
        # AB: 均匀分布 [0, 0.7]
        if not (0 <= AB <= 0.7):
            return -np.inf
        log_prior_AB = 0.0  # 均匀分布的对数概率为常数
        
        # alpha_ellip: 非负正态分布，mu=5, sigma=5
        if alpha_ellip < 0:
            return -np.inf  # 确保非负
        mu, sigma = 5.0, 5.0
        log_prior_alpha_ellip = -0.5 * ((alpha_ellip - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
        # 注意：这里未完全归一化截断正态分布，但对 MCMC 影响不大（仅影响常数项）
        
        # alpha_Doppler: 均匀分布 [0, 10]
        if not (0 <= alpha_Doppler <= 10):
            return -np.inf
        log_prior_alpha_Doppler = 0.0
        
        # F: 非负正态分布，mu=0, sigma=0.1
        if F < 0 or F > 0.5:
            return -np.inf  # 确保非负
        mu, sigma = 0, 0.1
        log_prior_F = -0.5 * ((F - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

        
        return log_prior_AB + log_prior_alpha_ellip + log_prior_alpha_Doppler + log_prior_F
    
    def log_posterior(self, params):
        """对数后验函数"""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)
    
    def sample(self):
        """运行 MCMC 采样并保存样本"""
        # initialize the walkers positions
        initial = np.zeros((self.nwalkers, self.ndim))
        initial[:, 0] = np.random.uniform(0, 0.7, self.nwalkers)  # AB
        initial[:, 1] = np.abs(np.random.normal(loc=5.0, scale=5.0, size = self.nwalkers))  # alpha_elips
        initial[:, 2] = np.random.uniform(0, 10, self.nwalkers)   # alpha_Doppler
        # F
        mu, sigma = 0.0, 0.1
        a, b = (0 - mu) / sigma, (0.5 - mu) / sigma  # 标准化截断范围
        initial[:, 3] = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=self.nwalkers)
        
        # Create the EnsembleSampler object using a multiprocessing pool
        with Pool() as pool:  # multiprocessing 多进程池
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, pool=pool)
            sampler.run_mcmc(initial, self.nsteps, progress=True)
            
        # 保存样本（展平后的样本，去除烧入期)
        samples = sampler.get_chain(discard=self.burnin, flat=True)
        path = os.path.join('Target', self.target_name, f'{self.file_name}_mcmc_samples.npy')
        np.save(path, samples)
        
        # 保存整个链以便绘制迹线图
        self.chain = sampler.get_chain()
        # print("type of chain: ", type(self.chain))

        # 计算 Gelman-Rubin 统计量以评估收敛性
        self.sampler = sampler
        trace = az.from_emcee(sampler)
        self.r_hat = az.rhat(trace)
        r_hat_values = self.r_hat.to_array().values
        print("R_hat for each parameter:", r_hat_values)
        
        return samples
    
    def load_samples(self):
        """加载保存的 MCMC 样本"""
        path = os.path.join('Target', self.target_name, f'{self.file_name}_mcmc_samples.npy')
        return np.load(path)
    
    def plot_fit(self, samples=None, num_samples=100):
        """绘制观测数据与模型预测的拟合图"""
        if samples is None:
            samples = self.load_samples()
        
        inds = np.random.randint(len(samples), size=num_samples)
        plt.figure(figsize=(10, 6))
        for ind in inds:
            sample = samples[ind]
            AB, alpha_ellip, alpha_Doppler, F = sample
            model_pred = Fp2Fs(self.data_X, AB, alpha_ellip, alpha_Doppler, F)
            plt.plot(self.data_X / (2 * np.pi), model_pred, "C1", alpha=0.1)
        plt.errorbar(self.data_X / (2 * np.pi), self.data_Y, yerr=self.sigma, fmt=".k", capsize=0, label="Data")
        plt.xlabel("Phase (normalized)")
        plt.ylabel("Flux")
        plt.legend()
        path = os.path.join('Target', self.target_name, f'{self.file_name}_model_fit.png')
        plt.savefig(path)
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
        path = os.path.join('Target', self.target_name, f'{self.file_name}_trace.png')
        plt.savefig(path)
        plt.close()
    
    def plot_corner(self, samples=None):
        """绘制角图展示后验分布"""
        if samples is None:
            samples = self.load_samples()
        
        fig = corner.corner(samples, labels=self.labels)
        path = os.path.join('Target', self.target_name, f'{self.file_name}_corner.png')
        plt.savefig(path)
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
    mcmc = MCMC('K2-141b', 'K2', sigma=7.047)
    mcmc.sample()       # 采样并保存
    mcmc.plot_fit()     # 绘制拟合图
    mcmc.plot_trace()   # 绘制迹线图
    mcmc.plot_corner()  # 绘制角图
    mcmc.compute_rhat() # 计算 Gelman-Rubin 统计量
    
    # 加载样本并分析
    samples = mcmc.load_samples()
    print("样本形状:", samples.shape)
    