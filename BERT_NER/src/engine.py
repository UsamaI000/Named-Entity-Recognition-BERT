import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import accuracy_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t==y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    accuracy = []
    f1_scor = []
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        tag, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
        acc = accuracy_score(data['target_tag'].cpu().numpy(), tag.argmax(2).cpu().numpy())
        accuracy.append(np.mean(acc))
        f1 = f1_score(data['target_tag'].cpu().numpy(), tag.argmax(2).cpu().numpy(), average='weighted', zero_division=0)
        f1_scor.append(f1)
        
    final_acc = np.mean(np.array(accuracy))
    final_f1 = np.mean(np.array(f1_scor))
    return final_loss / len(data_loader), final_acc, final_f1


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    accuracy = []
    f1_scor = []
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        tag, loss = model(**data)
        final_loss += loss.item()
        acc = accuracy_score(data['target_tag'].cpu().numpy(), tag.argmax(2).cpu().numpy())
        accuracy.append(np.mean(acc))
        f1 = f1_score(data['target_tag'].cpu().numpy(), tag.argmax(2).cpu().numpy(), average='weighted', zero_division=0)
        f1_scor.append(f1)

    final_acc = np.mean(np.array(accuracy))
    final_f1 = np.mean(np.array(f1_scor))
    return final_loss / len(data_loader), final_acc, final_f1
