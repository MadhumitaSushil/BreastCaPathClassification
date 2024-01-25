import os
import pandas as pd

import pyodbc
from credbl import get_mssql_connection_string


class CdwRetriever:
    def __init__(self, data_dir='../data/', config_cdw='../config/deid_cdw.yaml'):
        self.data_dir = data_dir
        self.config_cdw = config_cdw

        try:
            # Please note that on my local system, this connection string can only be created from the installed
            # Python 3.10 virtual environment called env, not from conda (due to the ARM64 processor).
            # If running on any different system, please change the driver accordingly.
            connection_str = get_mssql_connection_string(self.config_cdw, database='CDW_NEW',
                                                         driver='/opt/homebrew/Cellar/freetds/1.3.16/lib/libtdsodbc.0.so'
                                                         )
            self.conn = pyodbc.connect(connection_str)
        except Exception as e:
            print("Couldn't create a connection. Error: ", e)
            self.conn = None

    def query_cdw(self, query):
        if not self.conn:
            print("Error in obtaining connection string, aborting")
            return None

        df = pd.read_sql_query(query, self.conn)
        return df

    def serialize_results(self, df, fout):
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        with open(os.path.join(self.data_dir, fout), 'w') as f:
            df.to_csv(f, escapechar='\\', index=False)