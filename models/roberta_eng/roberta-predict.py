import torch
from torch import nn
from transformers import BertTokenizer, BertModel, BertConfig
import sys
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import net_roberta


def predict(text):

    NUM_CUDA = sys.argv[1]
    PATH = 'r_best_model_state.pth'

    config = BertConfig.from_pretrained('bert-base-uncased')

    device = torch.device('cuda:'+NUM_CUDA if torch.cuda.is_available() else "cpu")

    model = net_roberta.Net().to(device).double()
    model.load_state_dict(torch.load(PATH))

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

if __name__=='__main__':
    text = 'i really like banana'
    result = predict(text)
    print(result)
