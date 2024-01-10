import json
import numpy as np
import random
import torch
import torch.nn as nn
from models import *
import os
import matplotlib.pyplot as plt


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

gradient_clipping_norm=['0.5','1','1.5','2','2.5', '3', '3.5', '4']
acc_dpsgd_CNN=[0.70236, 0.74472, 0.76176, 0.76456, 0.76684, 0.7670399999999999, 0.76448, 0.74912]
acc_dpadam_CNN=[0.75156, 0.7594, 0.75948, 0.76, 0.76064, 0.75244, 0.75948, 0.7574]



plt.plot(gradient_clipping_norm, acc_dpsgd_CNN, label='DP-SGD', marker='o')
plt.plot(gradient_clipping_norm, acc_dpadam_CNN, label='DP-Adam', marker='s')

plt.legend()
plt.xlabel(r"Gradient clipping norm")
plt.ylabel("Accuracy")
plt.grid(linestyle='dashed', color="grey")
plt.title('CNN')
plt.savefig(folder_figure + f'/gradient_clipping' + '.png', dpi= 200)
plt.show()



