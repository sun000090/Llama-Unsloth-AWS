from datasets import Dataset
import pandas as pd
import glob
from logger import Logger

logs = Logger.logger_init()

class csv_creator:
    def read_table(csv_file):
        try:
            table = pd.read_csv(csv_file)
            columns_list = table.columns.tolist()
            table_list = table.fillna(0).to_numpy().tolist()
            fin_table = columns_list + table_list
            logs.info('Financial data read')
            return fin_table
        except Exception as e:
            logs.info(f'Error in reading csv files: {e}')
            return None
    
    def read_response(response_file):
        try:
            with open(response_file, "r") as file:
                content = file.read()
            logs.info('Response data read')
            return content
        except Exception as e:
            logs.info(f'Error in reading text files: {e}')
            return None
    
    def data_creator(loc1,loc2):
        try:
            file_read1 = []
            for filepath in glob.glob(loc1):
                file_read1.append(csv_creator.read_table(filepath))
            logs.info('All finance tables files read')
            
            file_read2 = []
            for filepath in glob.glob(loc2):
                file_read2.append(csv_creator.read_response(filepath))
            logs.info('All response files read')

        except Exception as e:
            logs.info(f'Error in data creation {e}')
            return None

        try:
            data_fin = []
            for i in range(len(file_read1)):
                data_fin.append({'prompt':f'Analyze the company from investor prespective. Financial data {file_read1[i]}','response':file_read2[i]})
            data_fin = Dataset.from_list(data_fin).train_test_split(test_size=0.1)
            logs.info('Dataset creation successful')
            return data_fin
        except Exception as e:
            logs.info(f'Error in dataset creation {e}')
            return None