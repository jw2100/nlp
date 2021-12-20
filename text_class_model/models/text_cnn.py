
import numpy
import torch

class Config(object):
    """模型参数配置"""
    def __init__(self, base_path, embedding_name):

        self.vocab_path = base_path + '/vocab.pkl'
        self.train_path = base_path + '/train.txt'   # 训练集
        self.dev_path   = base_path + '/dev.txt'     # 验证集
        self.test_path  = base_path + '/test.txt'
        self.save_path  = base_path + '/TextCNN.ckpt' # 模型训练结果
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

        self.dropout = 0.5               # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0                 # 词表大小，在运行时赋值
        self.num_epochs = 20             # epoch数
        self.batch_size = 128            # mini-batch大小
        self.pad_size = 32               # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3        # 学习率
        self.filter_sizes = (2, 3, 4)    # 卷积核尺寸
        self.num_filters = 256           # 卷积核数量(channels数)


class Model(torch.nn.Module):
    def __init__(self, config : Config):
        super(Model, self).__init__()
        if config.embedding_pretrained is None:
            self.embedding = torch.nn.Embedding(config.n_vocab, config.embedding_size, padding_idx=config.n_vocab-1)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        self.convs = torch.nn.ModuleList(
            # in_channels, out_channels, kernel_size, stride=1
            [torch.nn.Conv2d(1, config.num_filters, (k, config.embedding_size)) for k in config.filter_sizes]
        )
        self.dropout = torch.nn.Dropout(config.dropout)
        self.full_connect = torch.nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        #squeeze(3) : 1 * 64 * 30 * 1 = 1* 64 * 30
        x = torch.nn.functional.relu(conv(x)).squeeze(3)
        # max_pool1d : 1 * 64 * 30 => 1* 64 *1  (squeeze(2)) => 1* 64
        x = torch.nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
        return x;

    # x torch.LongTensor, int(seq_int)
    def forward(self, x : (torch.LongTensor, int)):
        out = self.embedding(x[0]) # .data.numpy().shape =     1 * seq_len * embedding_size
        out = out.unsqueeze(1)     #                       1 * 1 * seq_len * embedding_size

        # conv1 是 1 * num_filter * kernel_size=(filter_size * embedding_size)
        # conv1(x) == 1 * num_filter * seq_len * 1

        # cat 是拼接在一起
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.full_connect(out)
        return out

