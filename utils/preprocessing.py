import torch
from torch.utils.data import Dataset

# Define custom dataset Class
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]["message"]
        labels_multiclass = self.data.iloc[index]["multiclass_label"]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_index": labels_multiclass,
            "labels_multiclass": torch.tensor(labels_multiclass, dtype=torch.long),
        }


def assert_data_size(dataframe, batch_size):
    # Assert the data split w.r.t batch size
    num_to_delete = len(dataframe) % batch_size
    if not num_to_delete == 1:
        return dataframe

    # Get the count of unique values in column 'message'
    unique_counts = dataframe['message'].value_counts()
    
    # Identify the values that have more than one occurrence
    duplicates = unique_counts[unique_counts > 1].index.tolist()

    for duplicate in duplicates:
        indices = dataframe[dataframe['message'] == duplicate].index.tolist()

        if len(indices) >= num_to_delete:
            drop_idx = indices[:num_to_delete]
            dataframe = dataframe.drop(indices[:num_to_delete])
            # Loop over the duplicate, and delete 1 instance of
            # each, listed in duplicates
            dataframe = dataframe.reset_index(drop=True)
            return dataframe
    
    return dataframe


def label_binary_mapping(label):
        if label in ["low", "medium", "high"]:
            return "anomaly"
        else:
            return label


def label_multiclass_mapping_101(label):
    # normal    = [0, 0, 0, 0]
    # low       = [0, 1, 0, 1]
    # medium    = [0, 0, 1, 1]  -> means that this is an anomaly and the anomaly is medium
    # high      = [1, 0, 0, 1]
    normal = 0 if label == "normal" else 1
    low_anomaly = 1 if label == "low" else 0
    medium_anomaly = 1 if label == "medium" else 0
    high_anomaly = 1 if label == "high" else 0
    return [high_anomaly, low_anomaly, medium_anomaly, normal]
    
def label_multiclass_mapping_111(label):
    # normal    = [0, 0, 0, 0]
    # low       = [0, 1, 0, 1]  -> means that this is an anomaly and the anomaly is low
    # medium    = [0, 1, 1, 1]
    # high      = [1, 1, 1, 1]
    normal = 0 if label == "normal" else 1
    low_anomaly = 0 if label == "normal" else 1
    medium_anomaly = 1 if label == "medium" or label == "high" else 0
    high_anomaly = 1 if label == "high" else 0
    return [high_anomaly, low_anomaly, medium_anomaly, normal]