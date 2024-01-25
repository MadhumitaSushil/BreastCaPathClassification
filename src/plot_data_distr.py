import ast
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Distribution:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_df = self.load_data('train')
        self.dev_df = self.load_data('dev')
        self.test_df = self.load_data('test')

    def _remove_set_from_single_labels(self, label, col_name):
        label = ast.literal_eval(label)
        if col_name not in {'label_histo', 'label_site_disease', 'label_site_examined'}:
            label = label.pop()
        elif isinstance(label, str):
            label = set([label])
        return label

    def load_data(self, split):
        df = pd.read_csv(os.path.join(self.data_dir, split + '_all.csv'), index_col=False)

        if 'label_path_type_new' in df.columns:
            df = df.drop(columns=['label_path_type_new'])
        if 'label_biopsy_fixed' in df.columns:
            df = df.drop(columns=['label_biopsy_fixed'])

        label_cols = {col for col in df.columns if col.startswith('label_')}
        for col in label_cols:
            df[col] = df.apply(lambda x: self._remove_set_from_single_labels(x[col], col), axis=1)
        return df

    def get_class_distribution(self):
        tasks = {col[6:] for col in self.train_df.columns if col.startswith('label_')}
        print("Tasks: ", tasks)

        n_classes = defaultdict(lambda: defaultdict(int))

        for task in tasks:
            cur_labels = self.train_df['label_'+task].tolist()
            for label in cur_labels:
                if task in ['histo', 'site_disease', 'site_examined']:
                    for sub_label in label:
                        n_classes[task][sub_label] += 1
                else:
                    n_classes[task][label] += 1

        return n_classes


def plot(data):

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
        'histo': 'Histology',
        'site_examined': 'Sites examined',
        'site_disease': 'Sites of disease',
    }
    # Set Seaborn style
    sns.set_style("whitegrid", {'axes.grid': False})

    # Create a 4x4 grid of subplots
    fig, axs = plt.subplots(4, 4, figsize=(14, 8))

    # Flatten the axes for easier iteration
    axs = axs.flatten()

    # Iterate through tasks and create horizontal bar plots
    for i, task in enumerate(plot_titles.keys()):
        class_counts = data[task]
        # Sort class_counts by values in ascending order
        sorted_counts = dict(sorted(class_counts.items(), key=lambda item: item[1]))

        class_labels = list(sorted_counts.keys())
        class_values = list(sorted_counts.values())

        sns.barplot(x=class_values, y=class_labels, ax=axs[i], palette="Blues")
        axs[i].set_title(task)

        # Annotate each bar with its value
        for j, val in enumerate(class_values):
            axs[i].text(val + 1, j, str(val), va="center", fontsize=10)

        # Set the title for the subplot
        axs[i].set_title(f'{plot_titles[task]}', fontsize=10)
        axs[i].set_xlim([0, 500])
        axs[i].set_ylabel(None)

    fig.delaxes(axs[13])
    fig.delaxes(axs[14])
    fig.delaxes(axs[15])

    # Adjust subplot layout
    plt.tight_layout()

    # save the plot
    plt.savefig('../output/class_distr.png', dpi=600)


def main():
    distr = Distribution('../data/')
    class_distr = distr.get_class_distribution()
    plot(class_distr)

if __name__ == '__main__':
    main()

