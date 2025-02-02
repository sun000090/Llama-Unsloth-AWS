from dataset import csv_creator
import glob
from logger import Logger
from dataset import Dataset
from train_gpu_struct import TrainModel

logs = Logger.logger_init()

### Read all data from location
loc1 = 'Training_Context/*.csv'
loc2 = 'Training_Answer/*.txt'
training_data = csv_creator.data_creator(loc1=loc1,loc2=loc2)
dataset = training_data.map(TrainModel.add_text)

### Initialize and train model
model_name1 = 'unsloth/Llama-3.2-3B-Instruct' #Llama
model_name2 = 'unsloth/DeepSeek-R1-Distill-Llama-8B' #Deepseek
trainingSave = TrainModel.train_save_model(model_name_=model_name1, dataset = dataset)
