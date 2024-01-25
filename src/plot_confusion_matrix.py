import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix


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


def get_all_labels(task_name, data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, 'train_' + task_name + '.csv'))
    train_df['label'] = train_df.apply(lambda x: _remove_set_from_single_labels(x['label'], task_name), axis=1)
    if task_name not in {'histo', 'site_disease', 'site_examined'}:
        labels = {label for label in train_df['label'].unique()}
    else:
        labels = {label for labels in train_df['label'].tolist() for label in labels}
        labels = {label for label in labels}

    labels = sorted(list(labels))
    return labels


def load_golds_preds(fname, out_dir):
    return pd.read_csv(os.path.join(out_dir, fname))


def get_confusion_matrix(df, task_name, labels, model_name='gpt-4'):
    if task_name in ['histo', 'site_examined', 'site_disease']:
        oh_golds = np.zeros((df.shape[0], len(labels)))
        oh_preds = np.zeros((df.shape[0], len(labels)))
        for i, (cur_labels, cur_preds) in enumerate(zip(df['label_' + task_name].tolist(),
                                                        df[model_name + '_' + task_name].tolist()
                                                        )
                                                    ):

            for cur_label in ast.literal_eval(cur_labels):
                oh_golds[i, labels.index(cur_label)] = 1

            for cur_pred in ast.literal_eval(cur_preds):
                oh_preds[i, labels.index(cur_pred)] = 1

        cm = multilabel_confusion_matrix(y_true=oh_golds, y_pred=oh_preds)
    else:
        golds = df['label_' + task_name].tolist()
        preds = df[model_name + '_' + task_name].tolist()

        cm = confusion_matrix(y_true=golds, y_pred=preds, labels=labels)

    return cm


def visualize_cm(ax, cm, labels, title, font_size=20, x_rotation=45):
    sns.heatmap(cm, annot=True, fmt=".0f", ax=ax, cmap="Blues")

    ax.set_title(title)

    # Rotate y labels
    ax.set_xticklabels(labels, rotation=x_rotation, fontsize=font_size)
    ax.set_yticklabels(labels, rotation='horizontal', fontsize=font_size)

    # Label the axes with true and predicted
    ax.set_xlabel("Predicted label", fontsize=font_size)
    ax.set_ylabel("True label", fontsize=font_size)


def plot_single_label(fgpt='test_gpt-4_out_all.csv', out_dir='../output/', data_dir='../data/'):
    fig, axs = plt.subplots(2, 5, figsize=(40, 11))
    sns.set(font_scale=1.4)

    tasks = ['path_type', 'grade', 'pr', 'her2', 'biopsy', 'er', 'lvi', 'dcis_margins', 'margins', 'ln_involvement']
    plot_titles = {
        'path_type': 'Pathology type',
        'biopsy': 'Biopsy type',
        'er': 'Estrogen receptor status',
        'pr': 'Progesterone receptor status',
        'her2': 'HER-2 receptor status',
        'grade': 'Grade',
        'ln_involvement': 'Lymph node involvement',
        'lvi': 'Lymphovascular invasion',
        'margins': 'Margins',
        'dcis_margins': 'DCIS Margins',
    }

    for i, task in enumerate(tasks):
        labels = get_all_labels(task, data_dir)
        df = load_golds_preds(fgpt, out_dir)
        cm = get_confusion_matrix(df, task, labels)

        visualize_cm(axs[i // 5, i % 5], cm, labels, plot_titles[task])

    for i in range(len(tasks), 10, 1):
        fig.delaxes(axs[i // 5, i % 5])

    # Show the plot
    plt.tight_layout(h_pad=1.5, w_pad=1.5)
    plt.savefig(os.path.join(out_dir, 'gpt4_cm_single_label.png'), dpi=800)


def plot_multilabel(fgpt='test_gpt-4_out_all.csv', out_dir='../output/', data_dir='../data/'):
    fig, axs = plt.subplots(4, 5, figsize=(20, 12))
    sns.set(font_scale=1.1)

    tasks = ['histo', 'site_examined', 'site_disease']
    plot_titles = {
        'histo': 'Histology',
        'site_examined': 'Sites examined',
        'site_disease': 'Sites of disease',
    }

    plot_idx = 0
    for task in tasks:
        labels = get_all_labels(task, data_dir)
        df = load_golds_preds(fgpt, out_dir)
        cm = get_confusion_matrix(df, task, labels)

        for i, label in enumerate(labels):
            cm_label = cm[i]
            visualize_cm(axs[plot_idx // 5, plot_idx % 5], cm_label, ['Absent', 'Present'],
                         plot_titles[task]+': '+label, font_size=12, x_rotation=0)
            plot_idx += 1

    for i in range(plot_idx, 20, 1):
        fig.delaxes(axs[i // 5, i % 5])

    # Show the plot
    plt.tight_layout(h_pad=1.5, w_pad=1.5)
    plt.savefig(os.path.join(out_dir, 'gpt4_cm_multilabel.png'), dpi=1200)


if __name__ == '__main__':
    plot_single_label()
    plot_multilabel()