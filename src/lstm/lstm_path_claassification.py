import ast
import os
import pickle
import re

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def ohe_df(df, column_name, multi_label=False):
    if multi_label is True:
        df['label'] = df['label'].apply(ast.literal_eval)
        enc = MultiLabelBinarizer()
        labels_enc = enc.fit_transform(df['label'].values)
        labels = list(enc.classes_)
    else: 
        enc = OneHotEncoder(handle_unknown = 'ignore',drop=None)
        labels_enc = enc.fit_transform(df[['label']]).toarray()
        labels = list(enc.categories_[0])
        
    labels_enc = pd.DataFrame(labels_enc, columns=labels)
    df = pd.concat([df, labels_enc], axis=1)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df = df.loc[:, [column_name] + labels]

    return df, enc, labels


def ohe_transform(df, column_name, enc, multi_label=False):
    
    if multi_label is True:
        df['label'] = df['label'].apply(ast.literal_eval)
        labels_enc = enc.transform(df['label'].values)
        labels = list(enc.classes_)
    else:
        labels_enc = enc.transform(df[['label']]).toarray()
        labels = list(enc.categories_[0])
        
    labels_enc = pd.DataFrame(labels_enc, columns=labels)
    df = pd.concat([df, labels_enc], axis=1)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df = df.loc[:, [column_name] + labels]
    
    return df


def preprocess(sentences):
    r = "[^0-9A-Za-z. ]"
    preprocessed = []
    for sentence in sentences:
        # print(sentence)
        sentence = sentence.lower()
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = re.sub(r, ' ', sentence)
        sentence = re.sub("[+=——,$%^，。？、~@#￥%……&*《》<>「」{}【】() ./]", ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        if sentence.startswith(' ') or sentence.endswith(' '):
            sentence = re.sub(r"^(\s+)|(\s+)$", "", sentence)
        preprocessed.append(sentence)
    return preprocessed


def train_dev_test(train, dev_set, test_set, text_col_name, multi_label=False):
    ohe_df_train, enc, labels = ohe_df(train, text_col_name, multi_label)
    ohe_df_train = ohe_df_train[~ohe_df_train.index.duplicated(keep='last')]
    ohe_df_train = ohe_df_train.sample(n=len(ohe_df_train))
    
    df_dev_ohe = ohe_transform(dev_set, text_col_name, enc, multi_label)
    df_test_ohe = ohe_transform(test_set, text_col_name, enc, multi_label)

    preprocessed_train = preprocess(ohe_df_train[text_col_name])
    preprocessed_dev = preprocess(df_dev_ohe[text_col_name])
    preprocessed_test = preprocess(df_test_ohe[text_col_name])
    labels_train = np.array(ohe_df_train.iloc[:,1:])
    labels_dev = np.array(df_dev_ohe.iloc[:,1:])
    labels_test = np.array(df_test_ohe.iloc[:,1:])

    return preprocessed_train, preprocessed_dev, preprocessed_test, labels_train, labels_dev, labels_test, labels


def tokenize_(sentences_train, sentences_dev, sentences_test, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS):
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token='<OOV>', lower=True)

    tokenizer.fit_on_texts(sentences_train)

    word_index = tokenizer.word_index

    train_seq = tokenizer.texts_to_sequences(sentences_train)
    dev_seq = tokenizer.texts_to_sequences(sentences_dev)
    test_seq = tokenizer.texts_to_sequences(sentences_test)

    padd_train = pad_sequences(train_seq, maxlen = MAX_SEQUENCE_LENGTH)
    padd_dev = pad_sequences(dev_seq, maxlen = MAX_SEQUENCE_LENGTH)
    padd_test = pad_sequences(test_seq, maxlen = MAX_SEQUENCE_LENGTH)

    return padd_train, padd_dev, padd_test, word_index


def pipeline_LSTM(train_set, dev_set, test_set, text_col_name,
                  MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, EMBEDDING_DIM, multi=False,
                  femb='../data/emb.pkl'):
    
    preprocessed_train, preprocessed_dev, preprocessed_test, \
    labels_train, labels_dev, labels_test, labels = train_dev_test(train_set, dev_set, test_set,
                                                                   text_col_name, multi_label=multi)
    padd_train, padd_dev, padd_test, word_index = tokenize_(preprocessed_train, preprocessed_dev, preprocessed_test,
                                                 MAX_SEQUENCE_LENGTH, MAX_NB_WORDS)
    
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
    
    with open(femb, 'rb') as f:
        embeddings_index = pickle.load(f)

    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = tf.keras.layers.Embedding(num_words + 1,
                                                EMBEDDING_DIM,
                                                weights=[embedding_matrix],
                                                input_length=MAX_SEQUENCE_LENGTH,
                                                trainable=False)
    
    X_train = embedding_layer(padd_train).numpy()
    X_dev = embedding_layer(padd_dev).numpy()
    X_test = embedding_layer(padd_test).numpy()

    X_train = torch.Tensor(X_train).type(torch.float32)
    X_dev = torch.Tensor(X_dev).type(torch.float32)
    X_test = torch.Tensor(X_test).type(torch.float32)

    labels_train = torch.Tensor(labels_train).type(torch.float32)
    labels_dev = torch.Tensor(labels_dev).type(torch.float32)
    labels_test = torch.Tensor(labels_test).type(torch.float32)

    return X_train, X_dev, X_test, labels_train, labels_dev, labels_test, labels


class Torch_Model_Single(nn.Module):

    def __init__(self, n_classes, embed, lstm_size, lstm_layers, drop_):
        super().__init__()
        
        self.ls = nn.LSTM(embed, lstm_size, lstm_layers, bidirectional=True, batch_first =True, dropout=drop_)
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(lstm_size*2))
        self.fc1 = nn.Linear(lstm_size*2, lstm_size//2)
        self.fc2 = nn.Linear(lstm_size//2, n_classes)
        
    def forward(self, X):
        
        emb = X
        H,_ = self.ls(emb)
        M = self.tanh(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, 1)  
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out

    def classify(self, X, device, multi=False):
        
        outs = self.forward(X)

        if multi == False:
            labels = F.softmax(outs, dim=1)
            a = labels.argmax(1)
            m = torch.zeros(labels.shape).to(device).scatter(1, a.unsqueeze(1), 1.0)
        elif multi ==True:
            labels = torch.sigmoid(outs).data>0.5
            m = labels.to(torch.float32)

        return m


class AsymmetricLoss(nn.Module):
    
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
    
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

def train(model, model_path, device, X_train, Y_train, X_val, Y_val, epochs, batch_size, lr_,
          loss_f, all_labels=None, multi_=False):
    torch.manual_seed(21)
    torch.cuda.manual_seed(21)
    
    model.to(device)
    X_val = X_val.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_, weight_decay=1e-5)
    
    lambda1 = lambda epoch: 0.99 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [lambda1],)

    loss_fn = loss_f

    batches = X_train.shape[0]//batch_size

    losses = []
    training_accs = []
    validation_accs = []
    val_f1s = []
    best_score = 0

    for epoch in range(epochs):
        total_loss = 0
        tr_accs = 0
        f1_scs = 0
        for i in range(batches):
            X_batch = X_train[i*batch_size: i*batch_size + batch_size, :, :].to(device)
            y_batch = Y_train[i*batch_size: i*batch_size + batch_size, :].to(device)

            logits = model(X_batch)
            
            loss = loss_fn(logits, y_batch)

            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                tmp_logits = model.classify(X_batch, device, multi=multi_)
                tmp_logits = tmp_logits.cpu().detach().numpy()
                tmp_labels = y_batch.cpu().detach().numpy()

                tr_accs += accuracy_score(tmp_labels, tmp_logits)
                f1_scs += f1_score(tmp_labels, tmp_logits, average='macro', labels=range(len(all_labels)),
                                   zero_division=False)
            
            
        
        scheduler.step()

        #validate
        tmp_logits_val = model.classify(X_val, device, multi=multi_)
        tmp_logits_val = tmp_logits_val.cpu().detach().numpy()

        tmp_labels_val = Y_val.cpu().detach().numpy()
        
        f1 = f1_score(tmp_labels_val, tmp_logits_val, average='macro', labels=range(len(all_labels)),
                      zero_division=False)

        if f1 > best_score:
            best_score = f1
            torch.save(model.state_dict(), model_path)

        if not multi_:
            validation_acc = accuracy_score(tmp_labels_val, tmp_logits_val)
            print(f"Epoch {epoch}:\tloss {total_loss/batches} & training accuracy {tr_accs/batches} & validation accuracy {validation_acc} & validation f1 {f1}")
            validation_accs.append(validation_acc)
        else:
            print(f"Epoch {epoch}:\tloss {total_loss/batches} & training f1 {f1_scs/batches} & validation f1 {f1}")
        
        losses.append(total_loss/batches)
        training_accs.append(tr_accs/batches)


def eval(data_name, labels, model, model_path, preds_path, report_path, device, X_eval, Y_eval, multi_):

    X_eval = X_eval.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    best_model = model.eval()

    final_logits_eval = best_model.classify(X_eval, device, multi=multi_)
    final_logits_eval = final_logits_eval.cpu().detach().numpy()

    final_labels_eval = Y_eval.cpu().detach().numpy()

    report = classification_report(final_labels_eval, final_logits_eval, labels=range(len(labels)), zero_division=False)
    eval_f1_result = f1_score(final_labels_eval, final_logits_eval, average='macro', labels=range(len(labels)),
                              zero_division=False)

    with open(report_path, 'w') as f:
        f.write(f'\n{data_name}')
        f.write(f'\n{report}')
        f.write(f'\n{str(labels)}')
        f.write(f'\n final test f1 score: {eval_f1_result}')

    # Write predictions
    if not multi_:
        labels = [ast.literal_eval(label).pop() for label in labels]

    if not multi_:
        preds = [labels[i] for row in final_logits_eval for i, val in enumerate(row) if val == 1]
    else:
        preds = [{labels[i] for i, val in enumerate(row) if val == 1} for row in final_logits_eval]
    preds = pd.DataFrame({'predictions': preds})
    preds.to_csv(preds_path, index=False)

    return report


def main(tasks, data_dir='../data/', output_dir='../output/single_tasks/'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    single_labels = ['path_type', 'grade', 'biopsy', 'er', 'her2', 'lvi', 'pr', 'ln_involvement', 'margins',
                     'dcis_margins']
    multi_labels = ['histo', 'site_disease', 'site_examined']

    for task in tasks:
        if task in single_labels:
            multi_data = False
            loss = nn.CrossEntropyLoss()

        elif task in multi_labels:
            multi_data = True
            loss = AsymmetricLoss()

        else:
            raise ValueError("Unsupported task: ", task)

        train_df = pd.read_csv(os.path.join(data_dir, 'train_all.csv'), usecols=['idx', 'full_text', 'label_'+task])
        train_df = train_df.rename(columns={'label_' + task: 'label'})

        dev = pd.read_csv(os.path.join(data_dir, 'dev_all.csv'), usecols=['idx', 'full_text', 'label_'+task])
        dev = dev.rename(columns={'label_' + task: 'label'})

        test = pd.read_csv(os.path.join(data_dir, 'test_all.csv'), usecols=['idx', 'full_text', 'label_'+task])
        test = test.rename(columns={'label_' + task: 'label'})

        text_col_name = 'full_text'

        EMBEDDING_DIM = 250
        MAX_NB_WORDS = 5200
        MAX_SEQUENCE_LENGTH = 4600

        X_train, X_dev, X_test, labels_train, labels_dev, labels_test, labels = pipeline_LSTM(train_df, dev, test,
                                                                                              text_col_name,
                                                                                              MAX_SEQUENCE_LENGTH,
                                                                                              MAX_NB_WORDS,
                                                                                              EMBEDDING_DIM,
                                                                                              multi=multi_data)

        n_classes = labels_train.shape[1]
        embed = EMBEDDING_DIM
        lstm_size = 128
        lstm_layers = 2
        drop_ = 0.5
        epochs = 70
        batch_size = 16

        if task in ['path_type', 'grade', 'her2', 'er']:
            lr_ = 5e-4
        elif task in ['biopsy', 'lvi', 'pr', 'margins', 'dcis_margins', 'ln_involvement']:
            lr_ = 5e-3
        else:
            lr_ = 1e-3

        model = Torch_Model_Single(n_classes, embed, lstm_size, lstm_layers, drop_)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print("Task: ", task)

        model_dir = os.path.join(output_dir, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'best_' + task + '.pth')
        train(model, model_path, device, X_train, labels_train, X_dev, labels_dev, epochs, batch_size, lr_, loss,
              all_labels=labels, multi_=multi_data)

        results_dir = os.path.join(output_dir, 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        print("Dev set results for the best model: ")
        dev_preds_path = os.path.join(results_dir, 'dev_preds_' + task + '.log')
        dev_report_path = os.path.join(results_dir, 'dev_' + task + '.log')
        report = eval(task, labels, model, model_path, dev_preds_path, dev_report_path, device, X_dev, labels_dev, multi_=multi_data)

        print(report)
        print(labels)
        print('----------------------------------------------------------------------------------------------------')

        print("Test set results for the task: ", task)
        test_preds_path = os.path.join(results_dir, 'test_preds_' + task + '.log')
        test_report_path = os.path.join(results_dir, 'test_' + task + '.log')
        report = eval(task, labels, model, model_path, test_preds_path, test_report_path, device, X_test, labels_test, multi_=multi_data)

        print(report)
        print(labels)


if __name__ == "__main__":
    tasks = ['path_type', 'grade', 'biopsy', 'er', 'her2', 'lvi', 'pr', 'ln_involvement', 'margins',
             'dcis_margins', 'histo', 'site_disease', 'site_examined'
            ]

    main(tasks)
