#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:50:16 2020

@author: user
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import json
import torch.nn as nn
import os
import dataloader
import eval
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from   tensorboardX import SummaryWriter




"""========================================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
2. Output your results (BLEU-4 score, correction words)
3. Plot loss/score
4. Load/save weights
========================================================================================"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
UKN_token = 2
MAX_LENGTH = 25
#----------Hyper Parameters----------#
hidden_size = 256
#The number of vocabulary
vocab_size = 29
teacher_forcing_ratio = 1
LR = 0.05


#compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)


#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device),torch.zeros(1, 1, self.hidden_size, device=device))

#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=vocab_size):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    #----------sequence to sequence part for encoder----------#

    for di in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[di], encoder_hidden)
        
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden[0]
    decoder_cell   = decoder.initHidden()
    decoder_state  = (decoder_hidden , decoder_cell)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	
    #----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_state = decoder(
                decoder_input, decoder_state)
            loss += criterion(decoder_output, target_tensor[di].view(-1))
            decoder_input = target_tensor[di].view(1,-1)  # Teacher forcing
            if decoder_input.item() == EOS_token:
                break
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_state = decoder(
                decoder_input, decoder_state)
            decoder_input = F.softmax(decoder_output,dim=1).argmax(dim=1).view(1,-1)
            loss += criterion(decoder_output, target_tensor[di].view(-1))
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def trainIters(encoder, decoder, n_iters=75000, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    best_score = 0.
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    # your own dataloader
    '''
    Implement my own dataloader
    '''
    vocabulary = str()
    for voc in range(97 , 97+26):
        vocabulary += chr(voc)
    index2word =  {i+3 : c  for i, c in enumerate(vocabulary)}
    index2word[0] = 'SOS'
    index2word[1] = 'EOS'
    index2word[2] = 'UKN'
    
    training_pairs = dataloader.read_data('./train.json', n_iters)
    criterion = nn.CrossEntropyLoss()
    tb = SummaryWriter(log_dir = './tensorboard/')
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor  = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        output_word = eval.evaluate(encoder, decoder, input_tensor , target_tensor)
        target_chars = []
        input_chars  = []
        print("============================")
        for di in range(len(target_tensor)):
            if target_tensor[di].item() == EOS_token:
               target_chars.append(index2word[target_tensor[di].view(-1).cpu().numpy()[0]])
               break
            elif target_tensor[di].item() == SOS_token :
                pass
            else:
                target_chars.append(index2word[target_tensor[di].view(-1).cpu().numpy()[0]])
        for di in range(len(input_tensor)):
            if input_tensor[di].item() == EOS_token:
               input_chars.append(index2word[input_tensor[di].view(-1).cpu().numpy()[0]])
               break
            elif input_tensor[di].item() == SOS_token :
                pass
            else:
                input_chars.append(index2word[input_tensor[di].view(-1).cpu().numpy()[0]])
        input_word = ""
        input_word = input_word.join(input_chars[:-1])
        target_word = ""
        target_word = target_word.join(target_chars[:-1])       
        print("Input: ",input_word,
              "\nTarget: ",target_word,
              "\nPred: ",output_word)
        if iter % print_every == 0:
            save_name_en = "./checkpoint/encoder/"+str(iter)+".pkl"
            save_name_de = "./checkpoint/decoder/"+str(iter)+".pkl"
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            score = eval.run_evaluate(encoder , decoder, False)
            if score > best_score:
                torch.save(encoder.state_dict(),save_name_en)
                torch.save(decoder.state_dict(),save_name_de)
                best_score = score
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter/ n_iters),
                                         iter, iter/ n_iters * 100, print_loss_avg),
                  'Blue score: %.4f' % (score*100),'%')

            tb.add_scalar("Score", score , str(iter))
            tb.add_scalar("Loss",loss,str(iter))
if __name__ =="__main__":
    encoder1 = EncoderRNN(vocab_size, hidden_size).to(device)
    decoder1 = DecoderRNN(hidden_size, vocab_size).to(device)
    trainIters(encoder1, decoder1, 75000, print_every=100)
