#This file will prepare raw files for NMT model.
import utils
test_or_train='train'
preprocessed_data_file_path="./data/processed/hi/" + test_or_train + ".json"
raw_file_path="./data/processed/hi/" + test_or_train + ".raw.txt"
with open(raw_file_path,'w+') as raw_file:
    for features in utils.load_json(preprocessed_data_file_path):
        text = " ".join(features["words"])
        raw_file.write(text + '\n')
print('Successfully run.')