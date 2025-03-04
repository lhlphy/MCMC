import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

from core.Class_MCMC import MCMC

# 示例用法
if __name__ == '__main__':
    mcmc = MCMC('GJ 367b', 'JWST', sigma=15, ndim=5, nwalkers=64, nsteps=2000, burnin=1000)
    mcmc.sample()       # 采样并保存
    mcmc.plot_fit()     # 绘制拟合图
    mcmc.plot_trace()   # 绘制迹线图
    mcmc.plot_corner()  # 绘制角图
    mcmc.compute_rhat() # 计算 Gelman-Rubin 统计量
    mcmc.estimate_parameters()  # 估计参数值
    
    # 加载样本并分析
    samples = mcmc.load_samples()
    print("样本形状:", samples.shape)