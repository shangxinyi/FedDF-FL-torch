import argparse
import numpy as np
from torch import stack, div, max, eq, no_grad, tensor, save, load
from torch.optim import SGD, Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.nn.functional import softmax, log_softmax
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from light_resnet import light_resnet14
from dataset import classify_label, make_imbalance, Indices2Dataset
from dataset import partition_train_teach, show_clients_data_distribution
from sampling_dirichlet import clients_indices
from tqdm import tqdm
import os
import copy


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data/CIFAR10/'))
    parser.add_argument('--path_cifar100', type=str, default=os.path.join(path_dir, 'data/CIFAR100/'))
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_online_clients', type=int, default=8)
    parser.add_argument('--num_rounds', type=int, default=100)

    parser.add_argument('--num_data_train', type=int, default=50000)
    parser.add_argument('--num_data_valid', type=int, default=500)

    parser.add_argument('--num_epochs_local_training', type=int, default=20)
    parser.add_argument('--batch_size_local_training', type=int, default=64)

    parser.add_argument('--total_steps', type=int, default=10000)
    parser.add_argument('--mini_batch_size', type=int, default=128)

    parser.add_argument('--batch_size_test', type=int, default=500)

    parser.add_argument('--lr_global_teaching', type=float, default=0.001)
    parser.add_argument('--lr_local_training', type=float, default=0.1)

    parser.add_argument('--temperature', type=float, default=1.)

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--ratio_imbalance', type=float, default=1.)
    parser.add_argument('--non_iid_alpha', type=float, default=1)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--seed1', type=int, default=987087)
    parser.add_argument('--low', type=int, default=5000)
    parser.add_argument('--high', type=int, default=5000)

    args = parser.parse_args()

    return args


class Global(object):
    def __init__(self,
                 unlabeled_data,
                 num_classes: int,
                 total_steps: int,
                 mini_batch_size: int,
                 lr_global_teaching: float,
                 temperature: float,
                 device: str,
                 seed,
                 num_online_clients: int):
        self.model = light_resnet14(num_classes)
        self.model.to(device)
        self.model1 = light_resnet14(num_classes)
        self.model1.to(device)
        self.init_server_student = light_resnet14(num_classes)
        self.init_server_student.to(device)
        self.validate_model = light_resnet14(num_classes)
        self.validate_model.to(device)

        self.dict_global_params = self.model.state_dict()
        self.unlabeled_data = unlabeled_data
        self.total_steps = total_steps
        self.mini_batch_size = mini_batch_size
        self.ce_loss = CrossEntropyLoss()
        self.kld_loss = KLDivLoss(reduction="batchmean")
        self.optimizer = Adam(self.model.parameters(), lr=lr_global_teaching)
        self.lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 20], gamma=0.1)
        self.temperature = temperature
        self.device = device
        self.random_state = np.random.RandomState(seed)
        self.num_online_clients = num_online_clients
        self.epoch_acc = []
        self.epoch_loss = []

    def _early_stop(self, step, list_acc):
        idx_max = int(np.argmax(list_acc))
        if (step+1) - ((idx_max + 1) * 100) >= 1000:
            return True
        else:
            return False

    def update(self, list_dicts_local_params: list, list_nums_local_data: list, data_valid):
        # Prepare for validation data
        images_valid = []
        labels_valid = []
        for datum_valid in data_valid:
            image, label = datum_valid
            images_valid.append(image)
            label = tensor(label)
            labels_valid.append(label)
        images_valid = stack(images_valid, dim=0)
        labels_valid = stack(labels_valid, dim=0)
        images_valid = images_valid.to(self.device)
        labels_valid = labels_valid.to(self.device)
        # fedavg
        self._initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
        self.model.load_state_dict(self.dict_global_params)
        self.init_server_student = copy.deepcopy(self.model)
        #init_acc = self._compute_acc(images_valid, labels_valid)
        self.init_server_student.eval()
        with no_grad():
            num_corrects = 0
            outputs = self.init_server_student(images_valid)
            _, predicts = max(outputs, -1)
            num_corrects += sum(eq(predicts.cpu(), labels_valid.cpu())).item()
            init_acc = num_corrects / len(images_valid)


        # teach
        self.model.train()
        list_params = []
        list_acc = []
        for step in tqdm(range(self.total_steps)):
            total_indices = [i for i in range(len(self.unlabeled_data))]
            batch_indices = self.random_state.choice(total_indices, self.mini_batch_size, replace=False)
            images = []
            for idx in batch_indices:
                image, _ = self.unlabeled_data[idx]
                images.append(image)
            images = stack(images, dim=0)
            images = images.to(self.device)

            logits_teacher = self._avg_logits(images, list_dicts_local_params)
            # for step1 in range(self.server_steps):
            self._teach_one_step(images, logits_teacher)

            # validate
            if (step + 1) % 100 == 0:
                list_params.append(self.model.state_dict())
                self.validate_model = copy.deepcopy(self.model)
                self.validate_model.eval()
                with no_grad():
                    num_corrects = 0
                    outputs = self.validate_model(images_valid)
                    _, predicts = max(outputs, -1)
                    num_corrects += sum(eq(predicts.cpu(), labels_valid.cpu())).item()
                    acc = num_corrects / 500
                list_acc.append(acc)
                judge_bool = self._early_stop(step, list_acc)
                if judge_bool:
                    break
        idx_max = int(np.argmax(list_acc))
        print(f'Optimal step: {(idx_max + 1) * 100}')
        print(f'Initial acc: {init_acc}')
        print(f'Largest acc: {list_acc[idx_max]}')
        if list_acc[idx_max] > init_acc:
            print(f'FedDF win!')
            self.dict_global_params = list_params[idx_max]
        else:
            print(f'FedAvg win!')
            self.dict_global_params = self.init_server_student.state_dict()

    def _teach_one_step(self, images, logits_teacher):
        logits_student = self.model(images)
        x = log_softmax(div(logits_student, self.temperature), dim=1)
        y = softmax(div(logits_teacher, self.temperature), dim=1)
        loss = self.kld_loss(x, y.detach())
        #loss = self.kld_loss(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        for name_param in self.dict_global_params:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            self.dict_global_params[name_param] = value_global_param

    def _avg_logits(self, images, list_dicts_local_params: list):
        list_logits = []
        weights = ([1.0 / self.num_online_clients] * self.num_online_clients)
        for dict_local_params in list_dicts_local_params:
            self.model1.load_state_dict(dict_local_params)
            self.model1.eval()
            with no_grad():
                list_logits.append(self.model1(images))
        teacher_avg_logits = sum(
                [
                    teacher_logit * weight
                    for teacher_logit, weight in zip(list_logits, weights)
                ]
            )
        return teacher_avg_logits

    def eval(self, data_test, batch_size_test: int):
        self.model.load_state_dict(self.dict_global_params)
        self.model.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            list_loss = []
            for data_batch in tqdm(test_loader):
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
                loss_batch = self.ce_loss(outputs, labels)
                list_loss.append(loss_batch.cpu().item())
            accuracy = num_corrects / len(data_test)
            loss = sum(list_loss) / len(list_loss)
            self.epoch_acc.append(accuracy)
            self.epoch_loss.append(loss)

    def download_params(self):
        return self.dict_global_params


class Local(object):
    def __init__(self,
                 dict_global_params,
                 data_client,
                 num_classes: int,
                 num_epochs_local_training: int,
                 batch_size_local_training: int,
                 lr_local_training: float,
                 device: str):
        self.model = light_resnet14(num_classes)
        self.model.to(device)
        self.model.load_state_dict(dict_global_params)
        self.data_client = data_client
        self.num_epochs = num_epochs_local_training
        self.batch_size = batch_size_local_training
        self.ce_loss = CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=lr_local_training)
        self.device = device

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            data_loader = DataLoader(dataset=self.data_client,
                                     batch_size=self.batch_size,
                                     shuffle=True)
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.ce_loss(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def upload_params(self):
        return self.model.state_dict()


def main():
    args = args_parser()
    random_state = np.random.RandomState(args.seed)
    random_state1 = np.random.RandomState(args.seed1)
    # Load data
    data_local_training = datasets.CIFAR10(args.path_cifar10, transform=ToTensor())
    data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=ToTensor())
    unlabeled_data = datasets.CIFAR100(args.path_cifar100, transform=ToTensor())
    # Sample global validation data
    list_label2indices_test = classify_label(data_global_test, args.num_classes)
    indices_valid = []
    for indices in list_label2indices_test:
        indices_selected = random_state.choice(indices, args.num_data_valid // args.num_classes, replace=False)
        indices_valid.extend(indices_selected)

    data_valid = []
    for idx in indices_valid:
        data_valid.append(data_global_test[idx])
    # Distribute data
    list_label2indices = classify_label(data_local_training, args.num_classes)
    list_label2indices_train, _ = partition_train_teach(list_label2indices, args.num_data_train, args.seed)
    list_label2indices_imbalance = make_imbalance(list_label2indices_train, args.low, args.high, args.seed)
    list_client2indices = clients_indices(list_label2indices_imbalance, args.num_classes, args.num_clients,
                                          args.non_iid_alpha, args.seed)
    show_clients_data_distribution(data_local_training, list_client2indices, args.num_classes)
    indices2data = Indices2Dataset(data_local_training)

    global_model = Global(unlabeled_data=unlabeled_data,
                          num_classes=args.num_classes,
                          total_steps=args.total_steps,
                          mini_batch_size=args.mini_batch_size,
                          lr_global_teaching=args.lr_global_teaching,
                          temperature=args.temperature,
                          device=args.device,
                          seed=args.seed,
                          num_online_clients=args.num_online_clients)
    total_clients = list(range(args.num_clients))
    for r in range(args.num_rounds):
        dict_global_params = global_model.download_params()
        online_clients = random_state1.choice(total_clients, args.num_online_clients, replace=False)
        print(online_clients)
        list_dicts_local_params = []
        list_nums_local_data = []
        # local training
        for client in tqdm(online_clients, desc='local_training'):
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            list_nums_local_data.append(len(data_client))
            local_model = Local(dict_global_params=dict_global_params,
                                data_client=data_client,
                                num_classes=args.num_classes,
                                num_epochs_local_training=args.num_epochs_local_training,
                                batch_size_local_training=args.batch_size_local_training,
                                lr_local_training=args.lr_local_training,
                                device=args.device)
            local_model.train()
            dict_local_params = local_model.upload_params()
            list_dicts_local_params.append(dict_local_params)
        # global update
        print(f'Round: [{r + 1}/{args.num_rounds}] Global Updating')
        global_model.update(list_dicts_local_params, list_nums_local_data, data_valid)
        # global valuation
        print(f'Round: [{r + 1}/{args.num_rounds}] Global Testing')
        global_model.eval(data_global_test, args.batch_size_test)
        print('Accuracy')
        print(global_model.epoch_acc)
        print('Loss')
        print(global_model.epoch_loss)
    # 保存
    save(global_model, '\ feddf_validate_acc,epoch=20,batchsize=64,round=32.pkl')
    # 加载
    # model = load('\model.pkl')


if __name__ == '__main__':
    main()
