import torch
from torch import nn
from transformers import BertTokenizer, BertModel, BertConfig
import sys
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig


class Net(nn.Module):

    def __init__(self, dropout=0.5):

        super(Net, self).__init__()

        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 3)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def predict(text):

    PATH = 'r_best_model_state.pth'

    config = BertConfig.from_pretrained('bert-base-uncased')

    device = torch.device("cpu")
    model = Net().to(device).double()

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    tokens = tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")

    ans = model(tokens['input_ids'].to(device), tokens['attention_mask'].to(device))

    labels = {
            0: 'negative',
            1: 'positive',
            2: 'neutral',
        }

    ans = ans.argmax(dim=1)
    ans = labels[int(ans)]

    return ans