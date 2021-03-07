import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import preprocessing

# GIVE LABEL TO EACH WORD IN SENTENCE (DRUG: 1, ELSE 0)
def labels(sent, word):
  label = []
  sent = sent.split()
  for i, w in enumerate(sent):
    if word in w:
      label.append(1)
    else:
      label.append(0)
  return label

# METHOD TO PLOT ACCURACY AND LOSS CURVES
def plotCurves(stats):
    fig = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec2[0, 0])
    for i in ['train_loss', 'test_loss']: 
      ax1.plot(stats[i], label=i)
    ax1.legend()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    plt.show()
    fig.savefig('Metrics.png')


if __name__ == "__main__":
    
    print('Loading and pre-processing data....')
    f = pd.read_csv(config.TRAINING_FILE, error_bad_lines=False, low_memory=False)   ## Mudasser Path
    df = f.copy()
    df = df.drop(['Unnamed: 0', 'condition', 'date', 'usefulCount', 'rating'], axis=1)

    # Converting in lower case
    df.loc[:, 'review'] = df['review'].apply(lambda x: x.lower().replace('"','').replace(',','').replace('.','').replace(";",'').replace("&",'').replace("#",'').replace("%",""))
    df.loc[:, 'drugName'] = df['drugName'].apply(lambda x: x.lower())

    # Making a list of tokens and removing stopwords and other stuff
    stop_words = set(stopwords.words('english')) 
    df.loc[:,'review'] = df['review'].apply(lambda x: word_tokenize(x))
    df.loc[:,'review'] = df.loc[:,'review'].apply(lambda x: ' '.join([w for w in x if not w in stop_words]))
    df.loc[:,'review'] = df.loc[:,'review'].apply(lambda x: "".join([i for i in x if not i.isdigit()]))
    df.loc[:,'drugName'] = df['drugName'].apply(lambda x: "".join([i for i in x if not i.isdigit()]))

    # Finding drugname in corresponding review
    match = []
    for i, drug in enumerate(df['drugName']):
      if drug in df['review'][i]:
        match.append(1)
      else: match.append(0)
    len(match)
    df['match'] = match

    # Taking only those reviews that have drug name in them
    new_df = df[df.match ==True]
    new_df = new_df.reset_index().drop(['index'], axis=1)
    # Getting labels for each word for each review and storing in the dataframe
    label = []
    for i, sentences in enumerate(new_df['review']):
      drug = new_df['drugName'][i]
      label.append(labels(sentences, drug))
    new_df['labels'] = label

    new_df.loc[:, 'review'] = new_df['review'].apply(lambda x: x.split())
    sentences = new_df["review"].apply(list).values
    tag = new_df["labels"].apply(list).values
    print('\nDone')
    print(f'Length of dataset is: {len(new_df)}\n')
    
    num_tag = 2
    (
        train_sentences,
        test_sentences,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(sentences, tag, random_state=4242, test_size=0.25)

    train_dataset = dataset.EntityDataset(
        texts=train_sentences, tags=train_tag
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.EntityDataset(
        texts=test_sentences, tags=test_tag
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-7)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    train_losses, test_losses = [], []
    stats = []
    for epoch in range(1, config.EPOCHS+1):
        train_loss, train_acc, train_f1 = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)     
        train_losses.append(train_loss)

        test_loss, test_acc, test_f1 = engine.eval_fn(valid_data_loader, model, device)
        test_losses.append(test_loss)
        print(f"Epoch # {epoch}\n\tTrain Loss = {train_loss}, Test Loss = {test_loss},\n\tTrain Acc = {train_acc}, Test Acc = {test_acc},\n\tTrain F1 score = {train_f1}, Test F1 score = {test_f1}\n")

        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss
        history = {
          'train_loss': train_losses,
          'test_loss': test_losses
        }
        stat = pd.DataFrame(history, columns=['train_loss', 'test_loss'])
        if epoch % 3 == 0: plotCurves(stat)
