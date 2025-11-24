import os

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return torch.true_divide(torch.sum(preds == labels), len(labels))


def eval_fn(model, loader, device):
    """
    Evaluation method
    :param model: model to evaluate
    :param loader: data loader for either training or testing set
    :param device: torch device
    :return: accuracy on the data
    """
    score = AverageMeter()
    model.eval()

    t = tqdm(loader)
    with torch.no_grad():  # no gradient needed
        for images, labels in t:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            acc = accuracy(outputs, labels)
            score.update(acc.item(), images.size(0))

            t.set_description('(=> Test) Score: {:.4f}'.format(score.avg))

    return score.avg


def eval_model(model, saved_model_file, test_data_dir, data_augmentations):
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models', saved_model_file)))
    data = ImageFolder(test_data_dir, transform=data_augmentations)

    test_loader = DataLoader(dataset=data,
                             batch_size=1,
                             num_workers=0,
                             pin_memory=False,
                             shuffle=False)

    score = eval_fn(model, test_loader, device)

    print("Avg accuracy:", str(score * 100) + "%")
    with open("score.txt", "w") as f:
        f.write(str(score * 100) + "%")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
