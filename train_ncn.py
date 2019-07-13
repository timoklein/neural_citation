import random
import torch
from ncn.core import SEED, DEVICE
from ncn.training import init_weights, train_model
from ncn.model import NeuralCitationNetwork
from ncn.data_utils import get_bucketized_iterators



if __name__ == '__main__':
    # Set the random seeds before training
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # set up training
    data = get_bucketized_iterators("/home/timo/DataSets/KD_arxiv_CS/arxiv_data.csv")
    PAD_IDX = data.ttl.vocab.stoi['<pad>']
    cntxt_vocab_len = len(data.cntxt.vocab)
    aut_vocab_len = len(data.aut.vocab)
    ttl_vocab_len = len(data.ttl.vocab)
    

    net = NeuralCitationNetwork(context_filters=[4,4,5], context_vocab_size=cntxt_vocab_len,
                                authors=True, author_filters=[1,2], author_vocab_size=aut_vocab_len,
                                title_vocab_size=ttl_vocab_len, pad_idx=PAD_IDX, num_layers=2)
    net.to(DEVICE)
    net.apply(init_weights)

    train_model(net, data.train_iter, data.valid_iter, pad=PAD_IDX)