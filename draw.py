import json
import numpy as np
import random
import torch
import torch.nn as nn
from models import *
import os
import matplotlib.pyplot as plt
from opacus import PrivacyEngine




import warnings
warnings.simplefilter("ignore")


folder_log = "log"

folder_figure = "figure"

folder_models = "models"

def creat_dir(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

creat_dir(folder_figure)
creat_dir(folder_log)
creat_dir(folder_models)

plt.figure(1)

epsilon=['3','6','12','24','48']
acc_dpsgd_LSTM=[0.71384, 0.74072, 0.76432, 0.7734800000000001, 0.78788]
acc_dpadam_LSTM=[0.70808, 0.7388, 0.74912, 0.74796, 0.76104]
acc_sgd_LSTM=[0.82528, 0.82528, 0.82528, 0.82528, 0.82528]
acc_adam_LSTM=[0.8354, 0.8354, 0.8354, 0.8354, 0.8354]



plt.plot(epsilon, acc_dpsgd_LSTM, label='DP-SGD', marker='o')
plt.plot(epsilon, acc_dpadam_LSTM, label='DP-Adam', marker='s')
plt.plot(epsilon, acc_sgd_LSTM, label='SGD, $\epsilon=\infty$')
plt.plot(epsilon, acc_adam_LSTM, label='Adam, $\epsilon=\infty$')
plt.ylim(0.7, 0.85)
plt.legend()
plt.xlabel(r"Epsilon($\epsilon$), $\delta=8\times 10^{-5}$")
plt.ylabel("Accuracy")
plt.grid(linestyle='dashed', color="grey")
plt.title('LSTM')
plt.savefig(folder_figure + f'/LSTM' + '.png', dpi= 200)
plt.show()



plt.figure(2)

epsilon=['3','6','12','24','48']
acc_dpsgd_CNN=[0.73576, 0.74532, 0.75624, 0.76136, 0.76652]
acc_dpadam_CNN=[0.73284, 0.75844, 0.75948, 0.76836, 0.78052]     
acc_sgd_CNN=[0.82204, 0.82204, 0.82204, 0.82204, 0.82204]
acc_adam_CNN=[0.84372, 0.84372, 0.84372, 0.84372, 0.84372]



plt.plot(epsilon, acc_dpsgd_CNN, label='DP-SGD', marker='o')
plt.plot(epsilon, acc_dpadam_CNN, label='DP-Adam', marker='s')
plt.plot(epsilon, acc_sgd_CNN, label='SGD, $\epsilon=\infty$')
plt.plot(epsilon, acc_adam_CNN, label='Adam, $\epsilon=\infty$')
plt.ylim(0.7, 0.85)
plt.legend()
plt.xlabel(r"Epsilon($\epsilon$), $\delta=8\times 10^{-5}$")
plt.ylabel("Accuracy")
plt.grid(linestyle='dashed', color="grey")
plt.title('CNN')
plt.savefig(folder_figure + f'/CNN' + '.png', dpi= 200)
plt.show()


plt.figure(3)

epsilon=['3','6','12','24','48']
acc_dpsgd_FFN=[0.5378000000000001, 0.54488, 0.53484, 0.54888, 0.5615600000000001]    
acc_dpadam_FFN=[0.54028, 0.57252, 0.60408,0.54724, 0.61612]     
acc_sgd_FFN=[0.51896, 0.51896, 0.51896, 0.51896, 0.51896]
acc_adam_FFN=[0.83968, 0.83968, 0.83968, 0.83968, 0.83968]



plt.plot(epsilon, acc_dpsgd_FFN, label='DP-SGD', marker='o')
plt.plot(epsilon, acc_dpadam_FFN, label='DP-Adam', marker='s')
plt.plot(epsilon, acc_sgd_FFN, label='SGD, $\epsilon=\infty$')
plt.plot(epsilon, acc_adam_FFN, label='Adam, $\epsilon=\infty$')
plt.legend()
plt.xlabel(r"Epsilon($\epsilon$), $\delta=8\times 10^{-5}$")
plt.ylabel("Accuracy")
plt.grid(linestyle='dashed', color="grey")
plt.title('FFN')
plt.savefig(folder_figure + f'/FFN' + '.png', dpi= 200)
plt.show()