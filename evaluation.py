import json
import numpy as np
import random
import torch
import torch.nn as nn
from models import *
import os
import matplotlib.pyplot as plt
from opacus import PrivacyEngine
import sklearn

train_X_pad = np.loadtxt('train_X_pad.txt', dtype=int)
test_X_pad = np.loadtxt('test_X_pad.txt', dtype=int)
train_Y = np.loadtxt('train_Y.txt', dtype=int)
test_Y = np.loadtxt('test_Y.txt', dtype=int)

with open('vocab_1000.json', 'r') as file:
    vocab = json.load(file)

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



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

set_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

from sklearn.model_selection import train_test_split
train_X_pad, val_X_pad, train_Y, val_Y =train_test_split(train_X_pad, train_Y, test_size=0.5, random_state=42, stratify=train_Y)



from torch.utils.data import TensorDataset, DataLoader

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_Y), torch.from_numpy(train_X_pad))
val_data = TensorDataset(torch.from_numpy(val_Y), torch.from_numpy(val_X_pad))
test_data = TensorDataset(torch.from_numpy(test_Y), torch.from_numpy(test_X_pad))








from statistics import mean



def evaluation(model, criterion, test_loader, privacy_engine, delta=0, device=device):
    accs = []
    losses = []
    model.eval()
    labels_=[]
    preds_=[]
    labels_=np.array(labels_)
    preds_=np.array(preds_)
    with torch.no_grad():
        for label, text in test_loader:
            label = label.to(device)
            text = text.to(device)
            logits = model(text)
            loss = criterion(logits, label)

            preds = logits.argmax(-1)
            
            labels_=np.concatenate((labels_, label.cpu().numpy()))
            preds_=np.concatenate((preds_, preds.cpu().numpy()))


            n_correct = float(preds.eq(label).sum())
            batch_accuracy = n_correct / len(label)
            losses.append(float(loss))

            accs.append(batch_accuracy)
    recall = sklearn.metrics.recall_score(labels_, preds_)
    precision = sklearn.metrics.precision_score(labels_, preds_)
    f1 = sklearn.metrics.f1_score(labels_, preds_)
    auc = sklearn.metrics.roc_auc_score(labels_, preds_)
    printstr =  f"Test Accuracy: {mean(accs):.6f} | Test Loss: {mean(losses):.6f}"
    print(printstr)
    return mean(accs), mean(losses), recall, precision, f1, auc


def save_and_draw(train_accs, train_losses, val_accs, val_losses, setting):
    with open(folder_log+f'/train_loss_{setting}.txt', 'a') as fp:  
        for loss in train_losses:
            fp.write(f'{loss}\n')
    with open(folder_log+f'/val_loss_{setting}.txt', 'a') as fp:
        for loss in val_losses:
            fp.write(f'{loss}\n')
    with open(folder_log+f'/train_acc_{setting}.txt', 'a') as fp:  
        for acc in train_accs:
            fp.write(f'{acc}\n')
    with open(folder_log+f'/val_acc_{setting}.txt', 'a') as fp:
        for acc in val_accs:
            fp.write(f'{acc}\n')

    plt.figure(1)

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(linestyle='dashed', color="grey")
    plt.title(f'{setting}')
    plt.savefig(folder_figure + f'/loss_{setting}' + '.png', dpi= 200)
    plt.show()

    plt.figure(2)

    plt.plot(train_accs, label='Training Accuarcy')
    plt.plot(val_accs, label='Validation Accuarcy')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuarcy")
    plt.grid(linestyle='dashed', color="grey")
    plt.title(f'{setting}')
    plt.savefig(folder_figure + f'/acc_{setting}' + '.png', dpi= 200)
    plt.show()

if __name__ == "__main__":

    num_layers = 1
    vocab_size = len(vocab) + 1 #extra 1 for padding
    embedding_dim = 64
    output_dim = 2
    hidden_dim = 256
    filters=256 
    seq_len=500
    batch_size= 100
    device = 'cuda:1'


    to_run=[]
    to_run.append({'privacy': False, 'model': 'CNN', 'Optim': 'Adam', 'lr': 0.005, 'epochs': 20})
    to_run.append({'privacy': True, 'model': 'CNN', 'Optim': 'Adam', 'lr': 0.001, 'epochs': 20, 'epsilon': 3, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'CNN', 'Optim': 'Adam', 'lr': 0.001, 'epochs': 20, 'epsilon': 6, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'CNN', 'Optim': 'Adam', 'lr': 0.001, 'epochs': 20, 'epsilon': 12, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'CNN', 'Optim': 'Adam', 'lr': 0.001, 'epochs': 20, 'epsilon': 24, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'CNN', 'Optim': 'Adam', 'lr': 0.001, 'epochs': 20, 'epsilon': 48, 'delta': 8e-05})
    to_run.append({'privacy': False, 'model': 'CNN', 'Optim': 'SGD', 'lr': 0.1, 'epochs': 20})
    to_run.append({'privacy': True, 'model': 'CNN', 'Optim': 'SGD', 'lr': 0.1, 'epochs': 20, 'epsilon': 3, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'CNN', 'Optim': 'SGD', 'lr': 0.1, 'epochs': 20, 'epsilon': 6, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'CNN', 'Optim': 'SGD', 'lr': 0.1, 'epochs': 20, 'epsilon': 12, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'CNN', 'Optim': 'SGD', 'lr': 0.1, 'epochs': 20, 'epsilon': 24, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'CNN', 'Optim': 'SGD', 'lr': 0.1, 'epochs': 20, 'epsilon': 48, 'delta': 8e-05})
        
    to_run.append({'privacy': False, 'model': 'LSTM', 'Optim': 'Adam', 'lr': 0.005, 'epochs': 20})
    to_run.append({'privacy': True, 'model': 'LSTM', 'Optim': 'Adam', 'lr': 0.005, 'epochs': 20, 'epsilon': 3.0, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'LSTM', 'Optim': 'Adam', 'lr': 0.005, 'epochs': 20, 'epsilon': 6.0, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'LSTM', 'Optim': 'Adam', 'lr': 0.005, 'epochs': 20, 'epsilon': 12.0, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'LSTM', 'Optim': 'Adam', 'lr': 0.005, 'epochs': 20, 'epsilon': 24.0, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'LSTM', 'Optim': 'Adam', 'lr': 0.005, 'epochs': 20, 'epsilon': 48.0, 'delta': 8e-05})
    to_run.append({'privacy': False, 'model': 'LSTM', 'Optim': 'SGD', 'lr': 0.5, 'epochs': 20})
    to_run.append({'privacy': True, 'model': 'LSTM', 'Optim': 'SGD', 'lr': 0.5, 'epochs': 20, 'epsilon': 3.0, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'LSTM', 'Optim': 'SGD', 'lr': 0.5, 'epochs': 20, 'epsilon': 6.0, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'LSTM', 'Optim': 'SGD', 'lr': 0.5, 'epochs': 20, 'epsilon': 12.0, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'LSTM', 'Optim': 'SGD', 'lr': 0.5, 'epochs': 20, 'epsilon': 24.0, 'delta': 8e-05})
    to_run.append({'privacy': True, 'model': 'LSTM', 'Optim': 'SGD', 'lr': 0.5, 'epochs': 20, 'epsilon': 48.0, 'delta': 8e-05})
    
    for setting in to_run:
        print(setting)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)


        epochs = setting['epochs']
        learning_rate = setting['lr']
        if setting['model']=='DNN':
            model = DNN_model(vocab_size, embedding_dim, hidden_dim, output_dim, seq_len)
        elif setting['model']=='CNN':
            model = CNN_model(vocab_size, embedding_dim, filters, output_dim)
        elif setting['model']=='LSTM':
            if setting['privacy']:
                model = DPLSTM_model(vocab_size,embedding_dim, hidden_dim, output_dim, num_layers)
            else:
                model = LSTM_model(vocab_size,embedding_dim, hidden_dim, output_dim, num_layers)

        criterion = nn.CrossEntropyLoss()
        if setting['Optim']=='Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif setting['Optim']=='SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


        model.to(device)

        if setting['privacy']:
            max_per_sample_grad_norm = 1.5
            delta = setting['delta']
            epsilon = setting['epsilon']


            privacy_engine = PrivacyEngine(secure_mode=False)

            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                max_grad_norm=max_per_sample_grad_norm,
                target_delta=delta,
                target_epsilon=epsilon,
                epochs=epochs,
            )
        else:
            privacy_engine=False
            delta=0


        print(model)

        model.load_state_dict(torch.load(folder_models+f'/{setting}state_dict.pt'))

        print(model)
        test_acc, test_loss, recall, precision, f1, auc=evaluation(model, criterion, test_loader, privacy_engine, delta=delta, device=device)

        print('test acc:', test_acc)
        print('recall:', recall)
        print('precision:', precision)
        print('f1:', f1)
        print('auc:', auc)
        with open(folder_log+f'/evalucation_{setting}.txt', 'a') as fp:
            fp.write(f'{test_acc}\n')
            fp.write(f'{recall}\n')
            fp.write(f'{precision}\n')
            fp.write(f'{f1}\n')
            fp.write(f'{auc}\n')
        