
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
        self.num_filters = 256           # 卷积核数量(channels数) => 250


'''Deep Pyramid Convolutional Neural Networks for Text Categorization'''


class Model(torch.nn.Module):
    def __init__(self, config : Config):
        super(Model, self).__init__()
        if config.embedding_pretrained is None:
            self.embedding = torch.nn.Embedding(config.n_vocab, config.embedding_size, padding_idx=config.n_vocab - 1)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        self.conv_region = torch.nn.Conv2d(1, config.num_filters, (3, config.embedding_size), stride=1)
        self.conv = torch.nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(3,1), stride=2)
        self.padding1 = torch.nn.ZeroPad2d((0, 0, 1, 1)) # top bottom
        self.padding2 = torch.nn.ZeroPad2d((0, 0, 0, 1)) # bottom
        self.relu = torch.nn.ReLU()
        self.full_connect = torch.nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        x = x[0]
        x = self.embedding(x)
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.full_connect(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = torch.nn.functional.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x






















