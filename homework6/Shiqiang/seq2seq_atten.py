#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
import math
from io import open

import matplotlib
#if you are running on the gradx/ugradx/ another cluster, 
#you will need the following line
#if you run on a local machine, you can comment it out
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid, 
# it can be very easy to confict with other people's jobs.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15


class Vocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"), 
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab

######################################################################

def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################
class LSTM(nn.Module):
    """the class for the enoder RNN
    """
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        """Initilize a word embedding and bi-directional LSTM encoder
        For this assignment, you should *NOT* use nn.LSTM. 
        Instead, you should implement the equations yourself.
        See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        You should make your LSTM modular and re-use it in the Decoder.
        """
        '''
        i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
        f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
        g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{(t-1)} + b_{hg}) \\
        o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
        c_t = f_t c_{(t-1)} + i_t g_t \\
        h_t = o_t \tanh(c_t)
        '''
        "*** YOUR CODE HERE ***"
        
        self.w_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        
        self.w_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        self.b_i = nn.Parameter(torch.Tensor(hidden_size,1))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size,1))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size,1))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size,1))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def forward(self, input, hidden, cell):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        "*** YOUR CODE HERE ***"
        '''
        i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
        f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
        g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{(t-1)} + b_{hg}) \\
        o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
        c_t = f_t c_{(t-1)} + i_t g_t \\
        h_t = o_t \tanh(c_t)
        '''
        input = input.reshape(-1, 1)
        hidden = hidden.reshape(-1, 1)
        cell = cell.reshape(-1, 1)
        i_t = torch.sigmoid(torch.mm(self.w_ii, input) + torch.mm(self.w_hi, hidden) + self.b_i)
        f_t = torch.sigmoid(torch.mm(self.w_if, input) + torch.mm(self.w_hf, hidden) + self.b_f)
        g_t = torch.sigmoid(torch.mm(self.w_ig, input) + torch.mm(self.w_hg, hidden) + self.b_g)
        o_t = torch.sigmoid(torch.mm(self.w_io, input) + torch.mm(self.w_ho, hidden) + self.b_o)
        c_t = torch.mul(f_t, cell) + torch.mul(i_t, g_t)
        h_t = torch.mul(o_t, torch.tanh(c_t))
        h_t = h_t.view(1,1,-1)
        
        return h_t, h_t, c_t

class EncoderRNN(nn.Module):
    """the class for the enoder RNN
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        """Initilize a word embedding and bi-directional LSTM encoder
        For this assignment, you should *NOT* use nn.LSTM. 
        Instead, you should implement the equations yourself.
        See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        You should make your LSTM modular and re-use it in the Decoder.
        """

        "*** YOUR CODE HERE ***"
        
        self.embedding = nn.Embedding(input_size,hidden_size)
        self.lstm = LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden, cell):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        "*** YOUR CODE HERE ***"
        embed = self.embedding(input)
        output, hidden, cell = self.lstm(embed, hidden, cell)
        
        return output, hidden, cell

    def get_initial_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttDecoder(nn.Module):
    """the class for the decoder 
    """
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.dropout = nn.Dropout(self.dropout_p)
        
        """Initilize your word embedding, decoder LSTM here
        """
        "*** YOUR CODE HERE ***"
        self.embedding=nn.Embedding(output_size,hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.lstm = LSTM(hidden_size, hidden_size)  
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state
        
        Dropout (self.dropout) should be applied to the word embeddings.
        """
        
        "*** YOUR CODE HERE ***"

        embed = self.embedding(input).view(1, 1, -1)
        embed = self.dropout(embed)
        attn_weights = F.softmax(torch.sum(torch.mul(hidden.view(1,-1), encoder_outputs), dim=1))
        attn_hidden = torch.sum(torch.mul(attn_weights.view(-1,1),encoder_outputs), dim=0)
        
        attn_hidden = attn_hidden.view(-1,1)
        output = torch.cat((attn_hidden.view(1,-1), embed.view(1,-1)), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        
        output, hidden, cell = self.lstm(output, hidden, cell)
        output = self.out(output[0])
        log_softmax = self.softmax(output)
        
        
        return log_softmax, hidden, cell

    def get_initial_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


######################################################################

def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.get_initial_hidden_state()

    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()

    "*** YOUR CODE HERE ***"
    

    target_length = target_tensor.size(0)
    input_length = input_tensor.size(0)
    
    encoder_cell = encoder.get_initial_hidden_state()
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
    loss = 0.
    
    for i in range(input_length):
        encoder_output, encoder_hidden, encoder_cell = encoder(input_tensor[i], 
                                                               encoder_hidden, 
                                                               encoder_cell)
        encoder_outputs[i] = encoder_hidden[0,0]
        
    decoder_input = torch.tensor([[SOS_index]], device=device)
    decoder_hidden = encoder_hidden

    for i in range(target_length):
        decoder_output, decoder_hidden,encoder_cell = decoder(decoder_input, 
                                                              decoder_hidden, 
                                                              encoder_cell, 
                                                              encoder_outputs[:input_length-1])
        # topk取出每行指定维度最大数，及其位置
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach() ## detach from history as input
        loss += criterion(decoder_output, target_tensor[i])
        if decoder_input.item() == EOS_token:
            break

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item() 



######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    runs tranlsation, returns the output
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.get_initial_hidden_state()
        encoder_cell = encoder.get_initial_hidden_state()
        
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
             encoder_output, encoder_hidden, encoder_cell = encoder(input_tensor[ei],
                                                                    encoder_hidden,
                                                                    encoder_cell)
             encoder_outputs[ei] = encoder_output[0,0]

        decoder_input = torch.tensor([[SOS_index]], device=device)
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden,encoder_cell = decoder(decoder_input, 
                                                                  decoder_hidden, 
                                                                  encoder_cell, 
                                                                  encoder_outputs[:input_length-1])

            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = topi.detach()

        return decoded_words


######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################



def translate_and_show(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab):
    output_words = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=10000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')

    args = ap.parse_args()

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration 
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttDecoder(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    while iter_num < args.n_iters:
        iter_num += 1
        print(iter_num)
        training_pair = tensors_from_pair(src_vocab, tgt_vocab, random.choice(train_pairs))
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion)
        print_loss_total += loss

        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = 'state_%010d.pt' % iter_num
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
            translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

    # translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing 
    translate_and_show("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab)


if __name__ == '__main__':
    main()
