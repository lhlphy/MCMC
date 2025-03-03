import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from core.analytical_model import Fp2Fs

def MCMC(target_name, file_name):
    # 2. 读取数据
    data = np.loadtxt(f'Target\{target_name}\{file_name}.txt', delimiter=',')
    data_X = data[:,0] * 2 * np.pi
    data_Y = data[:,1]
    sigma = 10  # noise ppm

    # 3. 定义对数似然函数
    def log_likelihood(params, data_X, data_Y, sigma):
        AB, alpha_ellip, alpha_Doppler = params
        model = Fp2Fs(data_X, AB, alpha_ellip, alpha_Doppler)
        return -0.5 * np.sum((data_Y - model) ** 2 / sigma**2 + np.log(2 * np.pi * sigma**2))

    # 4. 定义对数先验函数（均匀分布 [0,1]）
    def log_prior(params):
        AB, alpha_ellip, alpha_Doppler = params
        if 0 <= AB <= 0.7 and 1 <= alpha_ellip <= 10 and 1 <= alpha_Doppler <= 10:
            return 0.0  # 在 [0,1] 范围内返回 0
        return -np.inf  # 超出范围返回 -∞

    # 5. 定义对数后验函数
    def log_posterior(params, data_X, data_Y, sigma):
        lp = log_prior(params)
        if not np.isfinite(lp):  # 如果先验为 -∞，直接返回
            return -np.inf
        return lp + log_likelihood(params, data_X, data_Y, sigma)

    # 6. 设置 MCMC 参数
    ndim = 3  # 参数数量 (AB, alpha_ellip, alpha_Doppler)
    nwalkers = 20  # 游走者数量
    nsteps = 1000  # 采样步数
    burnin = 500  # 烧入期步数

    # 初始化游走者的位置
    initial = np.zeros((nwalkers, ndim))
    initial[:, 0] = np.random.uniform(0, 0.7, nwalkers)  # AB
    initial[:, 1] = np.random.uniform(1, 10, nwalkers)  # alpha_ellip
    initial[:, 2] = np.random.uniform(1, 10, nwalkers)  # alpha_Doppler

    # 7. 运行 MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(data_X, data_Y, sigma))
    sampler.run_mcmc(initial, nsteps, progress=True)

    # 提取样本并丢弃烧入期
    samples = sampler.get_chain(discard=burnin, flat=True)

    # 8. 计算参数估计值
    AB_mcmc = np.mean(samples[:, 0])
    alpha_ellip_mcmc = np.mean(samples[:, 1])
    alpha_Doppler_mcmc = np.mean(samples[:, 2])
    AB_std = np.std(samples[:, 0])
    alpha_ellip_std = np.std(samples[:, 1])
    alpha_Doppler_std = np.std(samples[:, 2])

    # 打印结果
    print(f"AB: {AB_mcmc:.3f} ± {AB_std:.3f}")
    print(f"alpha_ellip: {alpha_ellip_mcmc:.3f} ± {alpha_ellip_std:.3f}")
    print(f"alpha_Doppler: {alpha_Doppler_mcmc:.3f} ± {alpha_Doppler_std:.3f}")

    # 9. 绘制角图
    labels = ["AB", "alpha_ellip", "alpha_Doppler"]
    fig = corner.corner(samples, labels=labels)
    plt.savefig("corner_plot.png")
    plt.close()

    print("角图已保存为 'corner_plot.png'")
    
if __name__ == '__main__':
    MCMC('K2-141b', 'K2-141b_K2')