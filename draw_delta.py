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

delta=['8e-6','8e-5','8e-4','8e-3','8e-2']
acc_dpsgd_CNN=[0.76108, 0.75624, 0.76724, 0.75944, 0.75844]
acc_dpadam_CNN=[0.74516, 0.75948, 0.7632, 0.75852, 0.77664]



plt.plot(delta, acc_dpsgd_CNN, label='DP-SGD', marker='o')
plt.plot(delta, acc_dpadam_CNN, label='DP-Adam', marker='s')

plt.legend()
plt.xlabel(r"Delta($\delta$)")
plt.ylabel("Accuracy")
plt.grid(linestyle='dashed', color="grey")
plt.title('CNN')
plt.savefig(folder_figure + f'/delta' + '.png', dpi= 200)
plt.show()



