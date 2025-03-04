import numpy as np
from scipy.integrate import dblquad, quad, tplquad
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from core.parameters import PPs

# Constants List
Rs = PPs.Rs
Rp = PPs.Rp
e = PPs.eccentricity
a = PPs.semi_axis
Ts = PPs.Stellar_T
P = PPs.Period
Mp_J = PPs.Mp_J
Rs_S = PPs.Rs_S
Ms_S = PPs.Ms_S
alpha = np.arcsin(Rs / a)
lam1 = 0.43e-6
lam2 = 0.89e-6

def Toy_model(zenith, AB, F=0):
    # Surface temperature model: Toy Model
    condition = zenith > np.pi / 2
    S = (1 - AB)**(1/4)* Ts / np.sqrt(a / Rs)
    branch_true = (F / 2)**(1/4)  * S
    branch_false = ((F / 2 + (1 - 2 * F) * np.cos(zenith)))**(1/4) * S
    return np.where(condition, branch_true, branch_false)
    
import numpy as np

def B(lam, T):
    # 定义物理常数
    h = 6.626e-34  # Planck's constant
    c = 3.0e8      # Speed of light
    k = 1.38e-23   # Boltzmann constant
    
    # 条件：T < 10
    condition = T < 10
    
    # 计算黑体辐射公式
    A = np.exp(h * c / lam / k / T) - 1
    blackbody_result = 2 * h * c**2 / lam**5 / A
    
    # 根据条件选择返回值
    return np.where(condition, 0, blackbody_result)
    
def A_Fresnel(Theta = 0, A_normal = 0):
        
    I_angle = (np.pi - Theta) / 2
    SINI = np.sin(I_angle)
    COSI = np.cos(I_angle)  
    n = 2/(1- np.sqrt(A_normal)) -1
    Co1 = np.sqrt(n**2 - SINI**2)

    Rs = ((COSI - Co1) / (COSI + Co1)) **2
    Rp = ((Co1 - n**2 *COSI)/ (Co1 + n**2 *COSI))**2
    return (Rs+Rp)/2
    
def F_thermal(Theta_array, AB, F=0):
    # print('1')
    results = []
    Cor = Rp**2 *(1-AB)/ (np.pi * Rs**2 * quad(lambda lam: B(lam, Ts), lam1, lam2)[0] )
    for i, Theta in enumerate(Theta_array):
        # print(Theta)
        # if Theta > np.pi + 0.01:  # 关于np.pi对称 
        #     results.append(results[len(Theta_array) - i - 1])
        #     continue
            
        if Theta < alpha or Theta > 2*np.pi - alpha: # transit
            results.append(-(Rp/Rs)**2)
            # print((Rp/Rs)**2)
            continue
        elif np.abs(Theta - np.pi) < alpha: # eclipse

            results.append(0)
            continue

        def int_func(lam, theta, phi):
            zenith = np.arccos(np.cos(theta)*np.cos(phi))
            return B(lam, Toy_model(zenith, AB, F)) * np.cos(phi)**2 * np.cos(np.pi - Theta - theta)

        # 定义采样点
        phi_list = np.linspace(-np.pi / 2, np.pi / 2, 180)
        theta_list = np.linspace(np.pi/2 - Theta, 3*np.pi/2 - Theta, 180)
        lam_list = np.linspace(lam1, lam2, 20)

        # 构造广播数组
        theta_array = theta_list[:, np.newaxis, np.newaxis]  # 形状 (180, 1, 1)
        phi_array = phi_list[np.newaxis, :, np.newaxis]      # 形状 (1, 180, 1)
        lam_array = lam_list[np.newaxis, np.newaxis, :]      # 形状 (1, 1, 10)

        # # 矢量化计算 I_matrix
        I_matrix = int_func(lam_array, theta_array, phi_array)

        # 计算结果
        result = np.sum(I_matrix) * (phi_list[1] - phi_list[0]) * (theta_list[1] - theta_list[0]) * (lam_list[1] - lam_list[0])
        # result, _ = tplquad(
        #     int_func,
        #     -np.pi / 2, np.pi / 2,  # phi limits
        #     lambda phi: np.pi/2 - Theta ,
        #     lambda phi: 3* np.pi/2 - Theta ,  # theta limits -> 3* np.pi/2 - Theta ||if F=0 use np.pi/2
        #     lambda phi, theta: lam1,
        #     lambda phi, theta: lam2,  # lam limits
        #     epsabs=1e-3,       # Increase absolute tolerance
        #     epsrel=1e-3       # Increase relative tolerance
        # )
        results.append(result)

    results = np.array(results) * Cor *1e6
    return results

def F_specular(Theta_array, AB):
    SI = A_Fresnel(Theta_array, AB)*(Rp/Rs)**2 * alpha**2 /4 * (1-alpha**2 /24 *(2-np.cos(Theta_array))/ np.sin(Theta_array/2)**2)
    SI[(Theta_array < alpha) | (Theta_array > 2*np.pi - alpha) | (np.abs(Theta_array - np.pi) < alpha)] = 0
    return SI *1e6
    
def F_ellip(Theta_array, alpha_ellip):
    A_ellip = alpha_ellip /0.077 *Mp_J* Rs_S**3 *Ms_S**-2 *P**-2
    return A_ellip *(1 - np.cos(2* Theta_array - 2*np.pi)) 

def F_Doppler(Theta_array, alpha_Doppler):
    A_Doppler = alpha_Doppler/0.37 *Mp_J *Ms_S**(-2/3) *P**(-1/3)
    return A_Doppler *np.sin(Theta_array)

def Fp2Fs(Theta_array, AB, alpha_ellip, alpha_Doppler, F=0, delta =0):
    return F_thermal(Theta_array, AB, F) + F_specular(Theta_array, AB) + F_ellip(Theta_array, alpha_ellip) + F_Doppler(Theta_array, alpha_Doppler) + delta
