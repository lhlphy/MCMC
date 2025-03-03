import numpy as np
import matplotlib.pyplot as plt
from analytical_model import *
import time

# test
if __name__ == '__main__':
    t1 = time.time()
    print('test')
    Theta_array = np.linspace(0, 2*np.pi, 100)
    AB = 0.3
    alpha_ellipse = 1
    alpha_Doppler = 1
    F_thermal = F_thermal(Theta_array, AB)
    F_specular = F_specular(Theta_array, AB)
    F_Doppler = F_Doppler(Theta_array, alpha_Doppler)
    F_ellip = F_ellip(Theta_array, alpha_ellipse)
    print(f'time1: {time.time() - t1}')
    
    # # subplot, plot each function as a subplot, so 4 subplots in total
    # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    # axs[0, 0].plot(Theta_array, F_thermal)
    # axs[0, 0].set_title("F_thermal")
    # axs[0, 0].set_ylim([0, max(F_thermal)*1.2])
    # axs[0, 1].plot(Theta_array, F_specular)
    # axs[0, 1].set_title("F_specular")
    # axs[1, 0].plot(Theta_array, F_Doppler)
    # axs[1, 0].set_title("F_Doppler")
    # axs[1, 1].plot(Theta_array, F_ellip)
    # axs[1, 1].set_title("F_ellip")
    # plt.show()
    
    # plt.subplots()
    # plt.plot(Theta_array, Fp2Fs(Theta_array, AB, alpha_ellipse, alpha_Doppler))
    # plt.ylim([0, max(F_thermal + F_specular + F_Doppler + F_ellip)*1.1])
    # plt.show()
    