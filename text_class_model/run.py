
import time
import importlib
import train_eval
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help = "text_cnn")
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
args = parser.parse_args()


if __name__ == "__main__":


    dataset_path = "../Chinese-Text-Classification-Pytorch/THUCNews/data"

    embedding = "embedding_SougouNews.npz"
    if args.embedding == 'random':
        embedding = 'random'

    model_name = args.model
    current_model = importlib.import_module("models." + model_name)

    if model_name == 'fasttext':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif
    config = current_model.Config(dataset_path, embedding)

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = current_model.Model(config).to(config.device)
    if model_name != 'transformer':
        train_eval.init_network(model)
    print(model.parameters)
    train_eval.train(config, model, train_iter, dev_iter, test_iter)