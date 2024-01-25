import os
import pandas as pd

from utils import CdwRetriever

pd.set_option('max_columns', None)


class Cohort:
    def __init__(self,
                 dir_data='../data/',
                 config_cdw='../config/deid_cdw.yaml',
                 ):
        self.dir_data = dir_data
        if not os.path.exists(self.dir_data):
            os.makedirs(self.dir_data)

        self.cdw_retriever = CdwRetriever(self.dir_data, config_cdw)

    def query_bc_cohort(self):
        qry_bc_diag = """
            SELECT A.PatientDurableKey, A.EncounterKey, C.PatientEpicId FROM deid_uf.DiagnosisEventFact as A
                     JOIN deid_uf.PatDurableDim as C
                     ON A.PatientDurableKey = C.PatientDurableKey
                     JOIN deid_uf.DiagnosisTerminologyDim as B
                     ON A.DiagnosisKey = B.DiagnosisKey
                     WHERE (C.Sex = 'Female')
                     AND ((B.Type = 'ICD-10-CM' AND 
                     B.Value LIKE 'C50%')
                     OR
                     (B.Type = 'ICD-10-CM' AND 
                     B.Value LIKE 'D05%')
                     OR
                     (B.Type = 'ICD-10-CM' AND 
                     B.Value LIKE 'Z85.3%')
                     OR
                     (B.Type = 'ICD-9-CM' AND 
                     B.Value LIKE '174%')
                     OR
                     (B.Type = 'ICD-9-CM' AND 
                     B.Value LIKE '175%')
                     OR
                     (B.Type = 'ICD-9-CM' AND 
                     B.Value LIKE '233.0%')
                     OR
                     (B.Type = 'ICD-9-CM' AND 
                     B.Value LIKE 'V10.3%'))
                AND (CONVERT(date, [StartDateKeyValue]) < '01-04-2021')
        """
        bc_pts = self.cdw_retriever.query_cdw(qry_bc_diag)
        print("Number of breast cancer patients: ", len(bc_pts['PatientEpicId'].unique()))
        self.cdw_retriever.serialize_results(bc_pts, 'bc_pts.csv')
        return bc_pts

    def query_path_notes_for_pts(self, enc_keys):
        qry_path_meta = """
            SELECT PatientEpicId, EncounterKey, deid_note_key FROM deid_uf.note_metadata
            WHERE note_type LIKE '%Pathology%' AND EncounterKey IN ({})
            AND CONVERT(date, [deid_service_date]) < '01-04-2021'
        """
        bc_path_notes_meta = self.cdw_retriever.query_cdw(qry_path_meta.format("'" + "','".join(enc_keys) + "'"))
        print("Number of breast cancer patients with pathology reports: ",
              len(bc_path_notes_meta['PatientEpicId'].unique()),
              "Number of pathology reports: ", len(bc_path_notes_meta['deid_note_key'].unique()))
        self.cdw_retriever.serialize_results(bc_path_notes_meta, 'bc_path_notes_meta.csv')
        return bc_path_notes_meta

    def query_note_text(self, deid_note_keys):
        qry_path_text = """
                    SELECT * FROM deid_uf.note_text
                    WHERE deid_note_key IN ({}) AND LEN(note_text) > 300
                    AND note_text NOT LIKE '%vaginal%'
                    AND note_text NOT LIKE '%cervical%'
                    AND note_text NOT LIKE '%cervix%'
                """
        bc_path_notes = self.cdw_retriever.query_cdw(qry_path_text.format("'" + "','".join(deid_note_keys) + "'"))
        print("Number of breast cancer patients with pathology reports > 300 characters "
              "after removing pap smear-related keywords: ",
              len(bc_path_notes['PatientEpicId'].unique()),
              "Number of pathology reports > 300 characters "
              "after removing pap smear-related keywords: ",
              len(bc_path_notes['deid_note_key'].unique()))

        self.cdw_retriever.serialize_results(bc_path_notes, 'bc_path_notes.csv')
        return bc_path_notes


def main():
    cohort_obj = Cohort()
    bc_pts = cohort_obj.query_bc_cohort()
    enc_keys = bc_pts['EncounterKey'].unique()
    bc_path_notes_meta = cohort_obj.query_path_notes_for_pts(enc_keys)
    deid_note_keys = bc_path_notes_meta['deid_note_key'].unique()
    bc_path_notes = cohort_obj.query_note_text(deid_note_keys)



if __name__=='__main__':
    main()