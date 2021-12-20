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

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率

        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2    # lstm层数
        self.hidden_size2 = 64

class Model(torch.nn.Module):
    def __init__(self, config : Config):
        super(Model, self).__init__()
        if config.embedding_pretrained is None:
            self.embedding = torch.nn.Embedding(config.n_vocab, config.embedding_size, padding_idx=config.n_vocab - 1)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)

        self.lstm = torch.nn.LSTM(
            config.embedding_size,
            config.hidden_size,
            config.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout
        )
        self.tanh1 = torch.nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.W = torch.nn.Parameter(torch.zeros(config.hidden_size * 2))  ## * 2 是因为双向？
        self.tanh2 = torch.nn.Tanh()
        self.full_connect  = torch.nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.full_connect2 = torch.nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        x, _ = x
        emb = self.embedding(x) #  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # output + hidden

        M = self.tanh1(H) #  [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = torch.nn.functional.softmax(torch.matmul(M, self.W), dim=1).unsqueeze(-1) # # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = torch.nn.functional.relu(out)
        out = self.full_connect(out)
        out = self.full_connect2(out)
        return out