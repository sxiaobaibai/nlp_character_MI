import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from nltk.corpus import brown

def n_gram(gram_len,sent):
    '''
    get n gram
    '''
    n_gram_pair = []
    sent_len = len(sent)
    for idx, w in enumerate(sent):
        if  idx+gram_len > sent_len:
            break
        gram = sent[idx:idx+gram_len]
        n_gram_pair.append(gram)
    return n_gram_pair


def get_input_output(n_gram_pair):
    input_list = []
    output_list = []
    for pair in n_gram_pair:
        input_str = pair[:-1]
        output_str = pair[-1]
        input_list.append(' '.join(input_str))
        output_list.append(output_str)
    return (input_list, output_list)

def convert_into_tensor(word):
    look = []
    for c in word:
        look.append(char_to_idx[c])
    np_look = np.array(look)
    return torch.from_numpy(np_look)

def convert_into_numpy(word):
    look = []
    for c in word:
        look.append(char_to_idx[c])
    np_look = np.array(look)
    return np_look#torch.from_numpy(np_look)

n_gram_pair = []
for sent in brown.sents():
    n_gram_pair += n_gram(2,sent)
input_, output_ = get_input_output(n_gram_pair)

full_sent = []
for w in brown.words():
    full_sent.append(w)
full_sent = ' '.join(full_sent)
character_vocab = set(full_sent)
char_len = len(character_vocab)

word_vocab = []
for w in brown.words():
    word_vocab.append(w)
word_vocab = set(word_vocab)
word_len = len(word_vocab)

word_to_idx = {}
idx_to_word = {}
for idx, word in enumerate(word_vocab):
    #print('%d, %s' % (idx,word))
    word_to_idx[word] = idx
    idx_to_word[idx] = word


char_to_idx = {}
idx_to_char = {}
for idx, char in enumerate(character_vocab):
    char_to_idx[char] = idx
    idx_to_char[idx] = char




class MyModel(nn.Module):
    '''
    Character level CNN NNLM
    '''
    def __init__(self):
        super(MyModel,self).__init__()
        self.char_embeds = nn.Embedding(char_len, 30)
        self.word_embeds = nn.Embedding(word_len,100)

        self.features = nn.Sequential(
            nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(1,1,kernel_size=2,stride=1),
        )
        self.linear = nn.Linear(29,1)
        self.sig = nn.Sigmoid()
        self.lstm = nn.LSTM(30, 100)
        self.thresh = nn.Linear(1,1)
    def forward(self,x):
        network_outputs = []
        original = convert_into_tensor(x)
        char_embedding = self.char_embeds(original)
        lstm_out = []
        morph_char = []
        if char_embedding.size()[0] == 1:
            _, out2 = self.lstm(char_embedding.unsqueeze(1))
            lstm_out.append(out2[0].view(1,-1))
            lstm_out = torch.cat(lstm_out,dim=0)
            morph_char.append(x)
            return lstm_out,x
            #summarized_lstm,_ = torch.max(lstm_out, 0)
            #network_outputs.append(summarized_lstm.unsqueeze(0))
            #continue

        a = char_embedding.unsqueeze(0)
        a = a.unsqueeze(0)
        seg = self.features(a)
        seg = self.linear(seg)
        seg = self.sig(seg)

        idx = torch.where(seg >= 0.5)
        boundries = idx[2] + 1
        outputs = []
        outputs.append(torch.tensor(0).unsqueeze(0))

        outputs.append(boundries)
        outputs.append(torch.tensor(original.size()[0]).unsqueeze(0))
        result = torch.cat(outputs, dim=0)
        morph = []
        #morph_char = []
        for cnt, idx in enumerate(result):
            if(cnt == 0):
                continue
            morph.append(char_embedding[result[cnt-1]:result[cnt]].unsqueeze(1))
            morph_char.append(x[result[cnt-1]:result[cnt]])


        for m in morph:
            _, out2 = self.lstm(m)
            lstm_out.append(out2[0].view(1,-1))

        lstm_out = torch.cat(lstm_out,dim=0)
        #return lstm_out
        return lstm_out,morph_char
