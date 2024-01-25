import ast
import os

import numpy as np
import pandas as pd

from art import labelingsignificance, precision_recall_fscore_macro


# Helper function for preprocessing
def _remove_set_from_single_labels(label, task_name):
    label = ast.literal_eval(label)
    if task_name not in {'histo', 'site_disease', 'site_examined'}:
        label = label.pop()
    elif isinstance(label, str):
        label = [label]
    elif isinstance(label, set):
        label = list(label)
    return label


def get_gold_labels(task_name, data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, 'train_all.csv'),
                           usecols=['idx', 'full_text', 'label_'+task_name]
                           )
    train_df = train_df.rename(columns={'label_' + task_name: 'label'})
    train_df['label'] = train_df.apply(lambda x: _remove_set_from_single_labels(x['label'], task_name), axis=1)
    if task_name not in {'histo', 'site_disease', 'site_examined'}:
        label_dict = {label: i for i, label in enumerate(train_df['label'].unique())}
    else:
        label_dict = {label for labels in train_df['label'].tolist() for label in labels}
        label_dict = {label: i for i, label in enumerate(label_dict)}

    df = pd.read_csv(os.path.join(data_dir, 'test_all.csv'), usecols=['idx', 'full_text', 'label_'+task_name])
    df = df.rename(columns={'label_' + task_name: 'label'})
    df['label'] = df.apply(lambda x: _remove_set_from_single_labels(x['label'], task_name), axis=1)

    labels = np.zeros((df.shape[0], len(label_dict)))
    for i, cur_label in enumerate(df['label'].tolist()):
        if task_name not in {'histo', 'site_disease', 'site_examined'}:
            labels[i, label_dict[cur_label]] = 1
        else:
            for cur_sub_label in cur_label:
                labels[i, label_dict[cur_sub_label]] = 1

    return labels, label_dict


def read_preds_bert(fpath, label_dict):
    df = pd.read_csv(fpath, sep='\t', header=None, usecols=[1], index_col=False)
    print(df)
    # TODO: support multilabel
    preds = [label_dict[pred] for pred in df[0].tolist()]
    print(preds)
    return preds


def read_preds_gpt(fpath, task_name, label_dict, model_name='gpt-4'):
    df = pd.read_csv(fpath)
    labels = np.zeros((df.shape[0], len(label_dict)))
    for i, cur_label in enumerate(df[model_name+'_'+task_name].tolist()):
        if task_name not in {'histo', 'site_disease', 'site_examined'}:
            labels[i, label_dict[cur_label]] = 1
        else:
            cur_label = ast.literal_eval(cur_label)
            for cur_sub_label in cur_label:
                labels[i, label_dict[cur_sub_label]] = 1

    return labels


def read_preds_lstm(fpath, task_name, label_dict):
    df = pd.read_csv(fpath, index_col=False)

    labels = np.zeros((df.shape[0], len(label_dict)))
    for i, cur_label in enumerate(df['predictions'].tolist()):
        if task_name not in {'histo', 'site_disease', 'site_examined'}:
            labels[i, label_dict[cur_label]] = 1
        else:
            cur_label = ast.literal_eval(cur_label)
            for cur_sub_label in cur_label:
                labels[i, label_dict[cur_sub_label]] = 1

    return labels


def get_significance(task, data_dir, fpred1, fpred2, num_shuffles):
    labels, label_dict = get_gold_labels(task, data_dir)

    preds1 = read_preds_gpt(fpred1, task, label_dict)
    preds2 = read_preds_lstm(fpred2, task, label_dict)

    p_vals = labelingsignificance(gold=labels, system1=preds1, system2=preds2, absolute=True,
                                 n=num_shuffles, scoring=precision_recall_fscore_macro,
                                 return_distribution=False)
    f_score_p = p_vals[-1]
    return f_score_p


def main():
    tasks = ['path_type', 'grade', 'pr', 'her2', 'biopsy', 'er', 'lvi', 'dcis_margins', 'margins', 'ln_involvement',
             'histo', 'site_examined', 'site_disease'
             ]
    data_dir = '../data/'
    fpreds_gpt4 = '../output/test_gpt-4_out_all.csv'
    dir_lstm_preds = '../output/single_tasks/results/'
    n_shuffles = 100000

    for task in tasks:
        p = get_significance(task, data_dir, fpreds_gpt4,
                             dir_lstm_preds + 'test_preds_' + task + '.log',
                             n_shuffles)

        if p <= 0.05:
            print("Significantly different for task: ", task, " p value: ", p)
        else:
            print("Not significantly different for task: ", task, " p value: ", p)


if __name__ == '__main__':
    main()