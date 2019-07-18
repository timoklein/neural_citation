from ncn.evaluation import Evaluator
from ncn.data import get_datasets




if __name__ == '__main__':
    path_to_weights = "/home/timo/Downloads/best_model_bn_TDNN/NCN_7_17_10.pt"
    path_to_data = "/home/timo/DataSets/KD_arxiv_CS/arxiv_data.csv"
    data = get_datasets(path_to_data)
    evaluator = Evaluator(path_to_weights, data, evaluate=True, show_attention=False)
    at_10 = evaluator.recall(10)
    print(at_10)