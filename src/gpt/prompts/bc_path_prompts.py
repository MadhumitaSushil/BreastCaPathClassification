from src.gpt.prompts.prompt_class import Prompt


class BreastCancerPrompt(Prompt):
    def __init__(self):
        self.system_prompt = self.get_system_prompt()
        self.user_prompt = self.get_user_prompt()

    def get_system_prompt(self):
        sysprompt = """
        Pretend you are a helpful Pathologist reading the given breast cancer pathology report." \
        Provide answers based on the pathology sample with the most aggressive or advanced cancer in the input report.
        Do not use patient history to answer, only provide the current patient information as answer.
        Answer as concisely as possible in the given format. 
        """
        return sysprompt

    def get_user_prompt(self):
        prompt = """Provide the type of pathology report, biopsy procedure type, sites examined, sites of cancer, 
        histological subtype, total number of lymph nodes involved, estrogen receptor status, progesterone receptor 
        status, her2 gene amplification status, tumor grade, lympho-vascular invasion, final resection margins for 
        invasive tumor, and final resection margin for DCIS tumor. Notes about breast reconstruction surgery, 
        or those unrelated to breast cancer are irrelevant here. Unknown option refers to the case where the answer 
        cannot be inferred from the input note. For all irrelevant notes, return all everything other than path_type 
        as the numeric option for Unknown. For molecular pathology report, report the path_type as the option for 
        Unknown. Report the grade for treated tumors as Unknown, and do not report nuclear grade unless the most 
        advanced tumor is of type DCIS. Numeric option for 'No malignancy' should be reported only if none of the 
        samples is malignant. Numeric option for DCIS should always be reported as a histological subtype if it is 
        present. Numeric option for 'Others' should be reported for histological subtype if a specific histological 
        type is not discussed, but the tumor is not benign. For margins inference, if multiple margins have been 
        reported, the margins after the final resection and associated with the worst prognosis, that is the one 
        closest to the tumor, should be provided. Report DCIS margins as 'Unknown' if no DCIS tumor exists. 
        
        Answer only with the most aggressive or advanced scenario in the current state. Do not use any history that 
        has not been confirmed currently to answer. 
        
        Answer from the given options for each output:
            pathology type: 1: Cytology 2. Histopathology 3. Either a report for a breast reconstruction surgery, or a report unrelated to any breast cancer 4. Unknown.
            biopsy procedure type: 1. Biopsy 2. Lumpectomy 3. Mastectomy 4. Unknown.
            sites examined: 1. Left breast 2. Left lymph node 3. Other tissues than breast or lymph nodes 4. Right breast 5. Right lymph node 6. Unknown.
            sites of cancer: 1. Left breast 2. Left lymph node 3. None 4. Other tissues than breast or lymph nodes 5. Right breast 6. Right lymph node 7. Unknown.
            histological sybtype: 1. DCIS 2. Invasive ductal carcinoma 3. Invasive lobular carcinoma 4. No malignancy was found 5. Other types of carcinoma than those mentioned 6. Unknown.
            total number of lymph nodes involved: 1. 1 to 3 lymph nodes involved, 2. More than 10 lymph nodes involved 3. 4 to 9 lymph nodes involved 4. No lymph nodes involved 5. Unknown.
            estrogen receptor status: 1. Negative 2. Positive 3. Unknown.
            progesterone receptor status: 1. Negative 2. Positive 3. Unknown.
            her2 gene amplification status: 1. Equivocal or indeterminate findings 2. Negative by FISH test 3. Positive by FISH test 4. Negative, but not with FISH test 5. Positive, but not with FISH test 6. Unknown.
            tumor grade: 1. 1 or low 2. 2 or intermediate 3: 3 or high, 4: Unknown.
            lympho-vascular invasion: 1. Absent 2. Present 3. Unknown
            final margins for invasive tumor: 1. Less than 2mm 2. More than or equal to 2mm 3. Positive margin 4. Unknown.
            final resection margins for DCIS tumor: 1. Less than 2mm 2. More than or equal to 2mm 3. Positive margin 4. Unknown.
            
         Provide the answers as a json in the following format, only using task-specific numeric options as specified above:
            {
            'path_type': option number for pathology type,
            'biopsy': option number for biopsy procedure type,
            'sites_examined': [list of all site option numbers that were examined for tumor],
            'sites_cancer': [list of all site option numbers where cancer is found],
            'histology': [list of all histological subtype option numbers for the most invasive tumor and DCIS],
            'lymph_nodes_involved': option number for the group that includes the total number of lymph nodes involved,
            'er': option number for estrogen receptor status,
            'pr': option number for progesterone receptor status,
            'her2': option number for her2 gene amplification status,
            'grade': option number for tumor grade, 
            'lvi': option number for lympho-vascular invasion,
            'margins': option number for final margins for invasive tumor,
            'dcis_margins': option number for final margins for dcis tumor,
            }
        
        Do not provide answer as a list for anything except sites_examined, sites_cancer and histology.
        """
        return prompt