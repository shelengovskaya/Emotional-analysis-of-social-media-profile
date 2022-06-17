import pandas as pd
import numpy as np
import random
import string
import pickle
import time

from tqdm import tqdm
from IPython.display import clear_output

import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.optim import Adam
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import sys
import matplotlib.pyplot as plt

import sys

argv = sys.argv

NUM_CUDA = str(argv[1])  # ?
MAX_LEN = int(argv[2])  # 64
BATCH_SIZE = int(argv[3])  # 64
N_EPOCHS = int(argv[4])  # 20
LR_ = 1e-6

#   no warnings

import warnings

warnings.filterwarnings('ignore')


#   keep seed

def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_random_seed(3407)

#   getting data

df = pd.read_csv('dataset_rus.csv')

#   creating ds class and encoding target

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

labels = {'negative': 0,
          'positive': 1,
          'neutral': 2,
          }


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [labels[label] for label in df['ans']]
        self.texts = [tokenizer(str(text),
                                padding='max_length', max_length=MAX_LEN, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


#   train test split
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                     [int(.8 * len(df)), int(.9 * len(df))])


#   realizing the net

class Net(nn.Module):
    def __init__(self, dropout=0.5):
        super(Net, self).__init__()

        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 128)
        self.tanh = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(128, 3)
        self.softmax = nn.Softmax()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout1(pooled_output)

        linear_output = self.linear1(dropout_output)

        tanh_output = self.tanh(linear_output)
        dropout_output = self.dropout2(tanh_output)

        linear_output = self.linear2(dropout_output)
        final_layer = self.softmax(linear_output)

        return final_layer


train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

def plot_losses(train_losses, test_losses, train_accuracies, test_accuracies):
    # clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))

    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(test_losses) + 1), test_losses, label='test')
    axs[0].set_ylabel('loss')

    axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='train')
    axs[1].plot(range(1, len(test_accuracies) + 1), test_accuracies, label='test')
    axs[1].set_ylabel('accuracy')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()

plot_results = True
def train(model, train_data, val_data, learning_rate, epochs, train_losses, test_losses, train_accuracies,
          test_accuracies):
    best_acc = 0

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + NUM_CUDA if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.to(device)
        criterion = criterion.to(device)

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                # print('val output', output)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        if total_acc_val > best_acc:
            torch.save(model.state_dict(), 'r_best_model_state.pth')
            best_acc = total_acc_val

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

        train_losses += [total_loss_train / len(train_data)]
        test_losses += [total_loss_val / len(val_data)]
        train_accuracies += [total_acc_train / len(train_data)]
        test_accuracies += [total_acc_val / len(val_data)]

        if plot_results:
            plot_losses(train_losses, test_losses, train_accuracies, test_accuracies)

    return train_losses, test_losses, train_accuracies, test_accuracies


model = Net()
LR = 1e-6

train_losses, test_losses, train_accuracies, test_accuracies = train(
    model, df_train, df_val, LR_, N_EPOCHS, train_losses, test_losses, train_accuracies, test_accuracies)

history = {'train_acc': train_accuracies, 'train_loss': train_losses,
           'val_acc': test_accuracies, 'val_loss': test_losses}

with open('roberta_sem_history.pickle', 'wb') as f:
    pickle.dump(history, f)


