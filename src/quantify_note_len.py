import ast
import os

import pandas as pd
from scipy import stats


def load_data(split, data_dir='../data/'):
    df = pd.read_csv(os.path.join(data_dir, split + '_all.csv'), index_col=False)

    if 'label_path_type_new' in df.columns:
        df = df.drop(columns=['label_path_type_new'])
    if 'label_biopsy_fixed' in df.columns:
        df = df.drop(columns=['label_biopsy_fixed'])

    label_cols = {col for col in df.columns if col.startswith('label_')}
    for col in label_cols:
        df[col] = df.apply(lambda x: _remove_set_from_single_labels(x[col], col), axis=1)
    return df


def _remove_set_from_single_labels(label, col_name):
    label = ast.literal_eval(label)
    if col_name not in {'label_histo', 'label_site_disease', 'label_site_examined'}:
        label = label.pop()
    elif isinstance(label, str):
        label = set([label])
    return label


def main():
    train_df = load_data('train')
    dev_df = load_data('dev')
    test_df = load_data('test')

    all_data = pd.concat([train_df, dev_df, test_df])
    print(all_data.columns)

    f = lambda x: len(x['full_text'].split()) - 1
    all_data['n_words'] = all_data.apply(f, axis=1)

    print(stats.iqr(all_data['n_words'].tolist()))


if __name__ == '__main__':
    main()
