# Breast Cancer Pathology Classification
This repository contains the source code for breast cancer pathology classification with supervised and zero-shot methods.
769 breast cancer pathology reports at the University of California, San Francisco were manually annotated with 13 categories of information including cancer sites, hitology, disease grade and spread, biomarkers, and surgical margins. 569 reports were used for supervised machine learning model training, 99 reports were used for validation, and 100 reports for testing. Random forests classifier, LSTM with attention, and UCSF-BERT model performance was compared with the GPT-4 model performance and the GPT-3.5 model performance. The GPT-4 model performed either significantly better than or as well as the best supervised model, the LSTM model with Attention. 

Please cite the following manuscript preprint if you use this code base in your study: 

[A comparative study of zero-shot inference with large language models and supervised modeling in breast cancer pathology classification](https://arxiv.org/abs/2401.13887).

The dataset for this study is coming soon.
