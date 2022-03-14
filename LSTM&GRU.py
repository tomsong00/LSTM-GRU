import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        #self.hidden_dim = hidden_dim
        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, input_dim):
        #embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(input_dim, 1, -1)
        #tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(lstm_out, dim=1)
        return tag_scores
#定义了一个LSTM网络
#nn.Embedding(vocab_size, embedding_dim) 是pytorch内置的词嵌入工具
#第一个参数词库中的单词数,第二个参数将词向量表示成的维数
#self.lstm LSTM层，nn.LSTM(arg1, arg2) 第一个参数输入的词向量维数，第二个参数隐藏层的单元数
#self.hidden2tag, 线性层

#前向传播的过程，很简单首先词嵌入(将词表示成向量)，然后通过LSTM层，线性层，最后通过一个logsoftmax函数
#输出结果，用于多分类



if __name__ == '__main__':
    #输入通道，input第三个值决定，输出通道数，hidden层数
    rnn = nn.LSTM(2, 2, 1)
    #输入要是三维数据
    input = torch.randn(10,2)
    input=input.view(-1,10,2)
    #第一个值与lstm第三个值一致，第二个值与input第二维一致，第三个值与输出通道数一致
    h0 = torch.randn(1, 10,2)
    c0 = torch.randn(1, 10,2)
    output, (hn, cn) = rnn(input, (h0, c0))
    #rnn = nn.LSTM(10, 20, 2)
    #input = torch.randn(5, 3, 10)
    #h0 = torch.randn(2, 3, 20)
    #c0 = torch.randn(2, 3, 20)
    #output, (hn, cn) = rnn(input, (h0, c0))
    print(h0.view(10,2))
    print(output)

    #赋值与
    #gru = torch.nn.GRU(5,10,2)
    #input = torch.randn(4,2,5)
    #h_0 = torch.randn(2,2,10)
    #output,h1= gru(input,h_0)
   #print(output.shape,h1.shape)

