import torch
from torch.utils.data import Dataset


class BaselineDataset(Dataset):
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


# Define custom dataset Class
class MultiheadDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]["message"]
        binary_name_label = self.data.iloc[index]["binary_name_label"]
        multi_name_label = self.data.iloc[index]["multi_name_label"]
        binary_label = self.data.iloc[index]["binary_label"]
        one_hot_label = self.data.iloc[index]["one_hot_label"]
        multi_101_label = self.data.iloc[index]["multi_101_label"]
        multi_111_label = self.data.iloc[index]["multi_111_label"]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "binary_name_label": binary_name_label,
            "multi_name_label": multi_name_label,
            "binary_index": binary_label,
            "one_hot_index": one_hot_label,
            "multi_101_index": multi_101_label,
            "multi_111_index": multi_111_label,
            "binary_tensor": torch.tensor(binary_label, dtype=torch.long),
            "one_hot_tensor": torch.tensor(one_hot_label, dtype=torch.long),
            "multi_101_tensor": torch.tensor(multi_101_label, dtype=torch.float32),
            "multi_111_tensor": torch.tensor(multi_111_label, dtype=torch.float32)
        }



# Define custom dataset
class MultitaskDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]["message"]
        labels_multitask = self.data.iloc[index]["multi_task_label"]
        class_label = self.data.iloc[index]["label"]

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
            "labels_multitask_index": labels_multitask,
            "class_label": class_label,
            "labels_multitask": torch.tensor(labels_multitask, dtype=torch.float32),
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


def label2multitask110(label):
    # To comply with the class weights from utils.preprocessing.inverse_freq
    # and sklearn.utils.class_weight import compute_class_weight,
    # the order made alphabetically.
    # [high_anomaly, low_anomaly, medium_anomaly, normal]
    # normal    = [0, 0, 0, 1]
    # low       = [0, 1, 0, 1]
    # medium    = [0, 1, 1, 0]  -> means that the medium shared common features with low
    # high      = [1, 0, 1, 0]
    if label == 'normal':
        return [0, 0, 0, 1]
    elif label == 'low':
        return [0, 1, 0, 1]
    elif label == 'medium':
        return [0, 1, 1, 0]
    else:
        return [1, 0, 1, 0]
    
    
def label2multitask101(label):
    # normal    = [0, 0, 0, 0]
    # low       = [0, 1, 0, 1]
    # medium    = [0, 0, 1, 1]  -> means that this is an anomaly and the anomaly is medium
    # high      = [1, 0, 0, 1]
    normal = 0 if label == "normal" else 1
    low_anomaly = 1 if label == "low" else 0
    medium_anomaly = 1 if label == "medium" else 0
    high_anomaly = 1 if label == "high" else 0
    # To comply with the class weights from utils.preprocessing.inverse_freq
    # and sklearn.utils.class_weight import compute_class_weight,
    # the order made alphabetically.
    return [high_anomaly, low_anomaly, medium_anomaly, normal]
    # return [normal, low_anomaly, medium_anomaly, high_anomaly]
    
    
def label2multitask111(label):
    # normal    = [0, 0, 0, 0]
    # low       = [0, 1, 0, 1]  -> means that this is an anomaly and the anomaly is low
    # medium    = [0, 1, 1, 1]
    # high      = [1, 1, 1, 1]
    normal = 0 if label == "normal" else 1
    low_anomaly = 0 if label == "normal" else 1
    medium_anomaly = 1 if (label == "medium" or label == "high") else 0
    high_anomaly = 1 if label == "high" else 0
    # To comply with the class weights from utils.preprocessing.inverse_freq
    # and sklearn.utils.class_weight import compute_class_weight,
    # the order made alphabetically.
    return [high_anomaly, low_anomaly, medium_anomaly, normal]
    # return [normal, low_anomaly, medium_anomaly, high_anomaly]
    
    
def label2multitask0101(label):
    # normal    = [0, 0, 0, 0]
    # low       = [0, 1, 0, 1]
    # medium    = [0, 0, 1, 1]  -> means that this is an anomaly and the anomaly is medium
    # high      = [1, 0, 0, 1]
    # normal = 1 if label == "normal" else 0
    # low_anomaly = 1 if label == "low" else 0
    # medium_anomaly = 1 if label == "medium" else 0
    # high_anomaly = 1 if label == "high" else 0
    # To comply with the class weights from utils.preprocessing.inverse_freq
    # and sklearn.utils.class_weight import compute_class_weight,
    # the order made alphabetically.
    if label == 'normal':
        return [0, 0, 0, 1]
    elif label == 'low':
        return [0, 1, 0, 1]
    elif label == 'medium':
        return [0, 0, 1, 1]
    else:
        return [1, 0, 0, 1]
    # return [high_anomaly, low_anomaly, medium_anomaly, normal]
    # return [normal, low_anomaly, medium_anomaly, high_anomaly]
    
    
def label2multitask0111(label):
    # normal    = [0, 0, 0, 0]
    # low       = [0, 1, 0, 1]  -> means that this is an anomaly and the anomaly is low
    # medium    = [0, 1, 1, 1]
    # high      = [1, 1, 1, 1]
    normal = 0 if label == "normal" else 1
    low_anomaly = 0 if label == "normal" else 1
    medium_anomaly = 1 if (label == "medium" or label == "high") else 0
    high_anomaly = 1 if label == "high" else 0
    # To comply with the class weights from utils.preprocessing.inverse_freq
    # and sklearn.utils.class_weight import compute_class_weight,
    # the order made alphabetically.
    if label == 'normal':
        return [0, 0, 0, 1]
    elif label == 'low':
        return [0, 1, 0, 1]
    elif label == 'medium':
        return [0, 1, 1, 1]
    else:
        return [1, 1, 1, 1]
    return [high_anomaly, low_anomaly, medium_anomaly, normal]
    # return [normal, low_anomaly, medium_anomaly, high_anomaly]
    