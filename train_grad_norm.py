import json
import numpy as np
import random
import torch
import torch.nn as nn
from models import *
import os
import matplotlib.pyplot as plt
from opacus import PrivacyEngine

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

def train(model, criterion, optimizer, train_loader, epoch, privacy_engine, delta=0, device=device):
    accs = []
    losses = []
    model.train()
    for label, text in train_loader:
        label = label.to(device)
        text = text.to(device)

        logits = model(text)
        preds = logits.argmax(-1)
        loss = criterion(logits, label)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        preds = logits.argmax(-1)
        n_correct = float(preds.eq(label).sum())
        batch_accuracy = n_correct / len(label)

        accs.append(batch_accuracy)
        losses.append(float(loss))

    printstr = (
        f"Epoch {epoch+1}. Train Accuracy: {mean(accs):.6f} | Train Loss: {mean(losses):.6f}"
    )
    if privacy_engine:
        epsilon = privacy_engine.get_epsilon(delta)
        printstr += f" | (ε = {epsilon:.2f}, δ = {delta})"

    print(printstr)
    return mean(accs), mean(losses)



def val(model, criterion, val_loader, privacy_engine, delta=0, device=device):
    accs = []
    losses = []
    model.eval()
    with torch.no_grad():
        for label, text in val_loader:
            label = label.to(device)
            text = text.to(device)
            logits = model(text)
            loss = criterion(logits, label)

            preds = logits.argmax(-1)
            n_correct = float(preds.eq(label).sum())
            batch_accuracy = n_correct / len(label)
            losses.append(float(loss))

            accs.append(batch_accuracy)
    printstr =  f"Val Accuracy: {mean(accs):.6f} | Val Loss: {mean(losses):.6f}"
    if privacy_engine:
        epsilon = privacy_engine.get_epsilon(delta)
        printstr += f" (ε = {epsilon:.2f}, δ = {delta})"
    print(printstr)
    return mean(accs), mean(losses)


def test(model, criterion, test_loader, privacy_engine, delta=0, device=device):
    accs = []
    losses = []
    model.eval()
    with torch.no_grad():
        for label, text in test_loader:
            label = label.to(device)
            text = text.to(device)
            logits = model(text)
            loss = criterion(logits, label)

            preds = logits.argmax(-1)
            n_correct = float(preds.eq(label).sum())
            batch_accuracy = n_correct / len(label)
            losses.append(float(loss))

            accs.append(batch_accuracy)
    printstr =  f"Test Accuracy: {mean(accs):.6f} | Test Loss: {mean(losses):.6f}"
    if privacy_engine:
        epsilon = privacy_engine.get_epsilon(delta)
        printstr += f" (ε = {epsilon:.2f}, δ = {delta})"
    print(printstr)
    return mean(accs), mean(losses)


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


    to_run=[]
    



    for l in [1e-1]:
        #to_run.append({"privacy":False, "model": 'CNN', "Optim": 'SGD', "lr": l, "epochs": 20})
        for e in [12]:
            for c in [1.5, 3.5]:
                to_run.append({"privacy":True, "model": 'CNN', "Optim": 'SGD', "lr": l, "epochs": 20, "epsilon": e, "delta": 8e-5, "grad_norm": c})
    for l in [1e-3]:
        #to_run.append({"privacy":False, "model": 'CNN', "Optim": 'Adam', "lr": l, "epochs": 20})
        for e in [12]:
            for c in [1.5, 3.5]:
                to_run.append({"privacy":True, "model": 'CNN', "Optim": 'Adam', "lr": l, "epochs": 20, "epsilon": e, "delta": 8e-5, "grad_norm": c})


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
            max_per_sample_grad_norm = setting['grad_norm']
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
        train_accs=[]
        train_losses=[]
        val_accs=[]
        val_losses=[]
        best_val_acc=0

        for epoch in range(epochs):
            train_acc, train_loss= train(model, criterion, optimizer, train_loader, epoch, privacy_engine, delta=delta, device=device)
            val_acc, val_loss= val(model, criterion, val_loader, privacy_engine, delta=delta, device=device)
            if val_acc>best_val_acc:
                best_val_acc=val_acc
                torch.save(model.state_dict(), folder_models+f'/{setting}state_dict.pt')
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
        save_and_draw(train_accs, train_losses, val_accs, val_losses, setting)
        model.load_state_dict(torch.load(folder_models+f'/{setting}state_dict.pt'))
        test_acc, test_loss=test(model, criterion, test_loader, privacy_engine, delta=delta, device=device)
        print('Best val_acc: ', best_val_acc)
        print('Test acc:', test_acc)
        with open(folder_log+f'/test_acc_{setting}.txt', 'a') as fp:
            fp.write(f'{test_acc}\n')            
            fp.write(f'{best_val_acc}\n')  