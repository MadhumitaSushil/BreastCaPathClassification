import ast
import os
import re

import pandas as pd

from src.evaluate import get_f1_score, get_accuracy
from src.openai_api import OpenaiApiCall
from src.prompts.bc_path_prompts import BreastCancerPrompt

from sklearn.preprocessing import MultiLabelBinarizer

def preprocess(text):
    r = "[^0-9A-Za-z. ]"
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r, ' ', text)
    text = re.sub("[+=——,$%^，。？、~@#￥%……&*《》<>「」{}【】() ./]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    if text.startswith(' ') or text.endswith(' '):
        text = re.sub(r"^(\s+)|(\s+)$", "", text)

    return text


class BreastCancerPathGpt4Inference:
    def __init__(self, data_dir, output_dir, model_name='gpt-35-turbo'):
        self.data_dir = data_dir
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.gpt_label_mapping = {
            'path_type': {1: 'Cytology', 2: 'Histopathology', 3: 'Irrelevant note', 4: 'Unknown'},
            'biopsy': {1: 'Biopsy', 2: 'Lumpectomy', 3: 'Mastectomy', 4: 'Unknown'},
            'sites_examined': {1: 'Left Breast', 2: 'Left LN', 3: 'Other tissues', 4: 'Right Breast', 5: 'Right LN', 6: 'Unknown'},
            'sites_cancer': {1: 'Left Breast', 2: 'Left LN', 3: 'None', 4: 'Other tissues', 5: 'Right Breast', 6: 'Right LN', 7: 'Unknown'},
            'histology': {1: 'DCIS', 2: 'Invasive ductal', 3: 'Invasive lobular', 4: 'No malignancy', 5: 'Others', 6: 'Unknown'},
            'lymph_nodes_involved': {1: '1-3 involved', 2: '10+ involved', 3: '4-9 involved', 4: '0 involved', 5: 'Unknown'},
            'er': {1: 'Negative', 2: 'Positive', 3: 'Unknown'},
            'pr': {1: 'Negative', 2: 'Positive', 3: 'Unknown'},
            'her2': {1: 'Equivocal', 2: 'FISH_Negative', 3: 'FISH_Positive', 4: 'Negative', 5: 'Positive', 6: 'Unknown'},
            'grade': {1: '1 (Low)', 2: '2 (Intermediate)', 3: '3 (High)', 4: 'Unknown'},
            'lvi': {1: 'Absent', 2: 'Present', 3: 'Unknown'},
            'margins': {1: 'Less than 2mm', 2: 'More than/eq to 2mm', 3: 'Positive margin', 4: 'Unknown'},
            'dcis_margins': {1: 'Less than 2mm', 2: 'More than/eq to 2mm', 3: 'Positive margin', 4: 'Unknown'},
        }

        self.gpt_to_task_name_mapping = {
            'path_type': 'path_type',
            'biopsy': 'biopsy',
            'sites_examined': 'site_examined',
            'sites_cancer': 'site_disease',
            'histology': 'histo',
            'lymph_nodes_involved': 'ln_involvement',
            'er': 'er',
            'pr': 'pr',
            'her2': 'her2',
            'grade': 'grade',
            'lvi': 'lvi',
            'margins': 'margins',
            'dcis_margins': 'dcis_margins',
        }

        self.model_name = model_name
        self.output_file = self.model_name + '_out_all.csv'

    def _remove_set_from_single_labels(self, label, col_name):
        label = ast.literal_eval(label)
        if col_name not in {'label_histo', 'label_site_disease', 'label_site_examined'}:
            label = label.pop()
        elif isinstance(label, str):
            label = set([label])
        return label

    def load_data(self, split):
        df = pd.read_csv(os.path.join(self.data_dir, split + '_all.csv'), index_col=False)
        df = df.drop(columns=['label_path_type_new'])

        label_cols = {col for col in df.columns if col.startswith('label_')}
        for col in label_cols:
            df[col] = df.apply(lambda x: self._remove_set_from_single_labels(x[col], col), axis=1)
        df['full_text_preproc'] = df.apply(lambda x: preprocess(x['full_text']), axis=1)
        return df

    def process_path_data(self, split):
        df = self.load_data(split)

        prompt_obj = BreastCancerPrompt()
        sys_prompt = prompt_obj.system_prompt
        user_prompt = prompt_obj.user_prompt

        api = OpenaiApiCall(model=self.model_name)

        outputs = list()

        for i, row in enumerate(df.itertuples()):
            print(row)

            # fetch completion
            cur_user_prompt = '"' + row.full_text_preproc + '"\n\n' + user_prompt
            result = api.get_response_text(cur_user_prompt, sys_prompt)

            if result is None: # content filter
                continue

            result = ast.literal_eval(result)

            # reformat results to append to a new df
            gpt_output = dict()
            for gpt_name, task_name in self.gpt_to_task_name_mapping.items():
                if task_name not in {'site_examined', 'site_disease', 'histo'}:
                    gpt_output[self.model_name + '_' + task_name] = self.gpt_label_mapping[gpt_name][result[gpt_name]]
                else:
                    gpt_output[self.model_name + '_' + task_name] = {self.gpt_label_mapping[gpt_name][label]
                                                                     for label in result[gpt_name]
                                                                     }
            gpt_output['idx'] = row.idx
            outputs.append(gpt_output)

            # if i == 2:
            #     break

        # save new dataframe as csv
        output_df = pd.DataFrame(outputs)
        output_df = df.set_index('idx').join(output_df.set_index('idx'))

        output_df.to_csv(os.path.join(self.output_dir, split+'_'+self.output_file))

    def evaluate_path_output(self, split):
        df = pd.read_csv(os.path.join(self.output_dir, split+'_'+self.output_file))

        for task, all_labels in self.gpt_label_mapping.items():
            task = self.gpt_to_task_name_mapping[task]
            all_labels = list(all_labels.values())

            task_labels = df['label_'+task].tolist()
            gpt_outputs = df[self.model_name+'_'+task].tolist()

            if task in {'histo', 'site_examined', 'site_disease'}:
                mlb = MultiLabelBinarizer(classes=all_labels)
                # Multilabel
                task_labels = [ast.literal_eval(label) for label in task_labels]
                task_labels = mlb.fit_transform(task_labels)

                gpt_outputs = [ast.literal_eval(label) for label in gpt_outputs]
                gpt_outputs = mlb.transform(gpt_outputs)

                macro_f1 = get_f1_score(y_true=task_labels, y_pred=gpt_outputs, average='macro',
                                        labels=range(len(all_labels))
                                        )

            else:
                print(all_labels)
                macro_f1 = get_f1_score(y_true=task_labels, y_pred=gpt_outputs, average='macro',
                                        labels=all_labels
                                        )

            print("Task: ", task)
            print("Macro F1: ", macro_f1, "\n")


if __name__ == '__main__':
    for model_name in ['gpt-35-turbo-16k', 'gpt-4']:
        print("Model: ", model_name)
        bc_inf_obj = BreastCancerPathGpt4Inference('../data/', '../output/', model_name=model_name)
        for split in ['dev', 'test']:
            print("Split: ", split)
            bc_inf_obj.process_path_data(split)
            bc_inf_obj.evaluate_path_output(split)