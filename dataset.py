import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from sampling_dirichlet import clients_indices


def show_clients_data_distribution(dataset, clients_indices: list, num_classes):
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in indices:
            label = dataset[idx][1]
            nums_data[label] += 1

        print(f'{client}: {nums_data}')


def partition_train_teach(list_label2indices: list, num_data_train: int, seed=None):
    #伪随机数序列
    random_state = np.random.RandomState(seed)
    list_label2indices_train = []
    list_label2indices_teach = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_train.append(indices[:num_data_train // 10])
        list_label2indices_teach.append(indices[num_data_train // 10:])

    return list_label2indices_train, list_label2indices_teach


class Indices2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None

    def load(self, indices: list):
        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.dataset[idx]
        return image, label

    def __len__(self):
        return len(self.indices)

#得到是索引
def classify_label(dataset, num_classes: int):
    list_label2indices = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list_label2indices[datum[1]].append(idx)

    return list_label2indices


def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res


#如果全局平衡的话，就是low=high=5000
#如果全局不平衡，就是low、high不一样
def make_imbalance(list_label2indices: list, low: int, high: int, seed=None):
    random_state = np.random.RandomState(seed)
    list_label2num = random_state.randint(low, high + 1, size=len(list_label2indices))

    list_label2indices_imbalance = []
    for indices, num in zip(list_label2indices, list_label2num):
        random_state.shuffle(indices)
        list_label2indices_imbalance.append(indices[:num])

    return list_label2indices_imbalance


def main():
    dataset = datasets.CIFAR10('./data/CIFAR10/', train=False)
    list_label2indices = classify_label(dataset, 10)
    list_label2indices_imbalance = make_imbalance(list_label2indices, 100, 1000, 1234)
    list_client2indices = clients_indices(list_label2indices_imbalance, 10, 100, 1, 1234)
    for client, indices in enumerate(list_client2indices):
        list_num_classes = [0 for _ in range(10)]
        for idx in indices:
            label = dataset[idx][1]
            list_num_classes[label] += 1

        print(f'{client}: {list_num_classes}')


if __name__ == '__main__':
    main()
