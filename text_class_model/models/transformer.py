
import copy
import numpy
import torch

class Config(object):
    def __init__(self, base_path, embedding_name):
        self.vocab_path = base_path + '/vocab.pkl'
        self.train_path = base_path + '/train.txt'  # 训练集
        self.dev_path = base_path + '/dev.txt'  # 验证集
        self.test_path = base_path + '/test.txt'
        self.save_path = base_path + '/TextCNN.ckpt'  # 模型训练结果
        class_path = base_path + '/class.txt'
        self.class_list = [x.strip() for x in open(class_path, encoding='utf-8').readlines()]  # 类别名单

        # 预训练词向量
        if embedding_name == 'random':
            self.embedding_pretrained = None
        else:
            # 预训练词向量
            embed_path = base_path + '/' + embedding_name
            self.embedding_pretrained = torch.tensor(numpy.load(embed_path)["embeddings"].astype('float32'))

        self.embedding_size = 300
        if self.embedding_pretrained is not None:
            self.embedding_size = self.embedding_pretrained.size(1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.dropout      = 0.5  # 随机失活
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab     = 0    # 词表大小，在运行时赋值
        self.num_epochs  = 20   # epoch数
        self.batch_size  = 128  # mini-batch大小
        self.pad_size    = 32   # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率

        self.dim_model   = 300
        self.hidden      = 1024
        self.last_hidden = 512
        self.num_head    = 5
        self.num_encoder = 2


class Model(torch.nn.Module):
    def __init__(self, config : Config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = torch.nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = torch.nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.position_embedding = PositionEncoding(
            config.embedding_size,
            config.pad_size,
            config.dropout,
            config.device)
        self.encoder = EncodeLayer(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = torch.nn.ModuleList([copy.deepcopy(self.encoder) for i in range(config.num_encoder)])
        self.full_connect = torch.nn.Linear(config.pad_size * config.dim_model, config.num_classes)

    def forward(self, x):
        out = self.embedding(x[0])
        out = self.position_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        out = self.full_connect(out)
        return out

class PositionEncoding(torch.nn.Module):
    def __init__(self, embedding_size, pad_size, dropout, device):
        super(PositionEncoding, self).__init__()
        self.device = device
        position_embedding = []
        for pos in range(pad_size):# 句子最长32
            vals = [pos / (10000.0 ** (i//2*2.0/embedding_size)) for i in range(embedding_size)]
            position_embedding.append(vals)
        self.position_embedding = torch.tensor(position_embedding)
        self.position_embedding[:, 0::2] = numpy.sin(self.position_embedding[:, 0::2])
        self.position_embedding[:, 1::2] = numpy.cos(self.position_embedding[:, 1::2])
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = x + torch.nn.Parameter(self.position_embedding, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out

####
'''
encode层
'''
class EncodeLayer(torch.nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(EncodeLayer, self).__init__()
        self.attention = MultiHeaderAttention(dim_model, num_head, dropout)
        self.feed_forward = PositionWiseFeedForward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out

class MultiHeaderAttention(torch.nn.Module):
    def __init__(self, dim_model, num_head, droput = 0.0):
        super(MultiHeaderAttention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.full_connent_Q = torch.nn.Linear(dim_model, num_head * self.dim_head)
        self.full_connent_K = torch.nn.Linear(dim_model, num_head * self.dim_head)
        self.full_connent_V = torch.nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = ScaledDotProductAttention()
        self.full_connent = torch.nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = torch.nn.Dropout(droput)
        self.layer_norm = torch.nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.full_connent_Q(x)
        K = self.full_connent_K(x)
        V = self.full_connent_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.full_connent(context)
        out = self.dropout(out)
        out = out + x # 残差连接
        out = self.layer_norm(out)
        return out

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = torch.nn.functional.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, dim_model, hidden, dropout = 0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.full_connect1 = torch.nn.Linear(dim_model, hidden)
        self.full_connect2 = torch.nn.Linear(hidden, dim_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.full_connect1(x)
        out = torch.nn.functional.relu(out)
        out = self.full_connect2(out)
        out = self.dropout(out)
        out = self.layer_norm(out)
        return out