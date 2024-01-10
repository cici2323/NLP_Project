import json
import numpy as np
import random
import os
import matplotlib.pyplot as plt



folder_log = "log"

to_run=[]

for l in [1e-3, 1e-2, 5e-3]:
    to_run.append({"privacy":False, "model": 'CNN', "Optim": 'Adam', "lr": l, "epochs": 20})
    for e in [3,6,12,24,48]:
        to_run.append({"privacy":True, "model": 'CNN', "Optim": 'Adam', "lr": l, "epochs": 20, "epsilon": e, "delta": 8e-5})

for l in [1, 1e-1, 5e-1]:
    to_run.append({"privacy":False, "model": 'CNN', "Optim": 'SGD', "lr": l, "epochs": 20})
    for e in [3,6,12,24,48]:
        to_run.append({"privacy":True, "model": 'CNN', "Optim": 'SGD', "lr": l, "epochs": 20, "epsilon": e, "delta": 8e-5})

table=[]

#here we use e=-1 to represent non dp, e=\infty
for e in [-1, 3, 6, 12, 24, 48]:
    settings=[]
    for l in [1e-3, 1e-2, 5e-3]:
        if e==-1:
            settings.append({"privacy":False, "model": 'CNN', "Optim": 'Adam', "lr": l, "epochs": 20})
        else:
            settings.append({"privacy":True, "model": 'CNN', "Optim": 'Adam', "lr": l, "epochs": 20, "epsilon": e, "delta": 8e-5})
    best_val_acc=0
    corresponding_test_acc=0
    
    for setting in settings:
        with open(folder_log+f'/test_acc_{setting}.txt', 'r') as file:
            test_and_val_accs=[]
            for line in file:
                acc = float(line)
                test_and_val_accs.append(acc)
            print(test_and_val_accs)
        if test_and_val_accs[1]>best_val_acc:
            best_val_acc = test_and_val_accs[1]
            corresponding_test_acc = test_and_val_accs[0]
            corresponding_setting = setting  
    
    with open(folder_log+f'/CNN_table.txt', 'a') as fp:
        fp.write(f'model: CNN, optim: Adam, epsilon: {e}:\n')            
        fp.write(f'Best_val_acc: {best_val_acc}\n')  
        fp.write(f'Corresponding_test_acc: {corresponding_test_acc}\n')  
        fp.write(f'Corresponding_setting: {corresponding_setting}\n')  


for e in [-1, 3, 6, 12, 24, 48]:
    settings=[]
    for l in [1, 1e-1, 5e-1]:
        if e==-1:
            settings.append({"privacy":False, "model": 'CNN', "Optim": 'SGD', "lr": l, "epochs": 20})
        else:
            settings.append({"privacy":True, "model": 'CNN', "Optim": 'SGD', "lr": l, "epochs": 20, "epsilon": e, "delta": 8e-5})
    best_val_acc=0
    corresponding_test_acc=0
    
    for setting in settings:
        with open(folder_log+f'/test_acc_{setting}.txt', 'r') as file:
            test_and_val_accs=[]
            for line in file:
                acc = float(line)
                test_and_val_accs.append(acc)
            print(test_and_val_accs)
        if test_and_val_accs[1]>best_val_acc:
            best_val_acc = test_and_val_accs[1]
            corresponding_test_acc = test_and_val_accs[0]
            corresponding_setting = setting  
    
    with open(folder_log+f'/CNN_table.txt', 'a') as fp:
        fp.write(f'model: CNN, optim: SGD, epsilon: {e}:\n')            
        fp.write(f'Best_val_acc: {best_val_acc}\n')  
        fp.write(f'Corresponding_test_acc: {corresponding_test_acc}\n')  
        fp.write(f'Corresponding_setting: {corresponding_setting}\n') 
