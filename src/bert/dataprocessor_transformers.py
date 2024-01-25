# coding=utf-8
# Copyright 2020- The Google AI Language Team Authors and The HuggingFace Inc. team and Facebook Inc.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""processors and helpers for classification"""

import ast
import logging
import os

import pandas as pd

from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, \
    confusion_matrix, multilabel_confusion_matrix

logger = logging.getLogger(__name__)


def convert_examples_to_features(
    processor,
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if task is not None and processor is not None:
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))
        inputs = tokenizer.encode_plus(
            example.text_a, example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        elif output_mode == 'multilabel_classification':
            label = [label_map[l] for l in example.label]
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            if output_mode == 'multilabel_classification':
                logger.info("label: %s (id = %s)" % (example.label, str(label)))
            else:
                logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    return features


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


def _read_csv(input_file, task_name, quotechar='"'):
    """Reads a comma separated value file."""
    cols_to_load = ['idx', 'full_text', 'label_'+task_name]
    # if task_name != 'path_type':
    #     cols_to_load.append('label_path_type')

    df = pd.read_csv(input_file, delimiter=",", quotechar=quotechar, usecols=cols_to_load)
    df = df.rename(columns={'label_'+task_name: 'label'})
    df['label'] = df.apply(lambda x: _remove_set_from_single_labels(x['label'], task_name), axis=1)

    # # removing irrelevant notes for all tasks except path type
    # if task_name != 'path_type':

    return df


class PathProcessor(DataProcessor):
    """Processor for the breast cancer pathology preprocessed dataset"""

    def __init__(self, task_name, data_dir='../../data/'):
        self.task_name = task_name

        train_df = _read_csv(os.path.join(data_dir, 'train_all.csv'), self.task_name)
        if 'multilabel' in output_modes[task_name]:
            self.labels = {l for label_set in train_df['label'].tolist() for l in label_set}
            self.labels = list(self.labels)
        else:
            self.labels = train_df['label'].unique()

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, 'train_all.csv'), self.task_name))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, 'dev_all.csv'), self.task_name))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, 'test_all.csv'), self.task_name))

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, df):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, row) in enumerate(df.itertuples(index=False)):
            if i != 0:
                guid = str(i)
                text_a = row.full_text
                label = row.label
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


output_modes = {
    "path_type": "classification",
    "grade": "classification",
    "pr": "classification",
    "her2": "classification",
    "biopsy": "classification",
    "er": "classification",
    "lvi": "classification",
    "dcis_margins": "classification",
    "margins": "classification",
    "ln_involvement": "classification",
    "histo": "multilabel_classification",
    "site_examined": "multilabel_classification",
    "site_disease": "multilabel_classification",
}

stopping_metrics = {
    "path_type": "macro_f1",
    "grade": "macro_f1",
    "pr": "macro_f1",
    "her2": "macro_f1",
    "biopsy": "macro_f1",
    "er": "macro_f1",
    "lvi": "macro_f1",
    "dcis_margins": "macro_f1",
    "margins": "macro_f1",
    "ln_involvement": "macro_f1",
    "histo": "macro_f1",
    "site_examined": "macro_f1",
    "site_disease": "macro_f1",
}


def multiclass_metrics(preds, labels, is_multilabel=False):
    acc = accuracy_score(y_pred=preds, y_true=labels)
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    macro_weighted_f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    macro_precision = precision_score(y_true=labels, y_pred=preds, average='macro')
    macro_weighted_precision = precision_score(y_true=labels, y_pred=preds, average='weighted')
    macro_recall = recall_score(y_true=labels, y_pred=preds, average='macro')
    macro_weighted_recall = recall_score(y_true=labels, y_pred=preds, average='weighted')
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    if not is_multilabel:
        confusion = confusion_matrix(y_true=labels, y_pred=preds)
    else:
        confusion = multilabel_confusion_matrix(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        'micro_f1': micro_f1,
        "macro_f1": macro_f1,
        "macro_weighted_f1": macro_weighted_f1,
        "macro_precision": macro_precision,
        "macro_weighted_precision": macro_weighted_precision,
        "macro_recall": macro_recall,
        "macro_weighted_recall": macro_weighted_recall,
        "confusion_matrix": confusion,
    }


def compute_metrics(preds, labels, examples, is_multilabel=False):
    assert len(preds) == len(labels) == len(examples)

    results = multiclass_metrics(preds, labels, is_multilabel)
    return results
