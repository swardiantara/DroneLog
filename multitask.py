import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
# from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import os
import copy
import datetime
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
from sklearn.utils.class_weight import compute_class_weight
from utils.preprocessing import assert_data_size
from utils.losses import inverse_freq, FocalLoss
from utils.preprocessing import label2multitask101, label2multitask111
from utils.postprocessing import label2class
from utils.assert_scenario import assert_baseline
from utils.preprocessing import MultitaskDataset, assert_data_size 
from utils.losses import inverse_freq, FocalLoss
from models.dronelog import DroneLog

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='filtered',
                    choices=['filtered', 'unfiltered'])
parser.add_argument('--output_dir', type=str, default='multitask',
                    help="Folder to store the experimental results. Default: multitask")
parser.add_argument('--word_embed', type=str, choices=[
                    'bert'], default='bert', help='Type of Word Embdding used. Default: BERT-base')
parser.add_argument('--encoder', type=str, choices=['transformer', 'lstm', 'gru', 'none'], default='none',
                    help="Encoder Architecture used to perform computation. Default: None.")
parser.add_argument('--pooling', type=str, choices=['cls', 'max', 'min', 'mean'], default='cls',
                    help="Pooling mechanism to get final representation for non-RNN models. Default: BERT [CLS]")
parser.add_argument('--freeze_embedding', action='store_true',
                    help="Wether to freeze the pre-trained embedding's parameter.")
parser.add_argument('--bidirectional', action='store_true',
                    help="Wether to use Bidirectionality for LSTM and GRU.")
parser.add_argument('--save_best_model', action='store_true',
                    help="Wether to save best model for each encoder type.")
parser.add_argument('--class_weight', choices=['uniform', 'balanced', 'inverse'], default='uniform',
                    help="Wether to weigh the class based on the class frequency. Default: Uniform")
parser.add_argument('--loss', choices=['logloss'], default='logloss',
                    help="Loss function to use. Default: logloss")
parser.add_argument('--label_schema', choices=['101', '111'], default='101',
                    help="Target label schema. Default: 101.")
parser.add_argument('--n_heads', type=int, default=1,
                    help='Number of attention heads')
parser.add_argument('--n_layers', type=int, default=1,
                    help='Number of encoder layers')
parser.add_argument('--n_epochs', type=int, default=10,
                    help='Number of training iterations')

args = parser.parse_args()


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    args = parser.parse_args()
    embedding_type = args.word_embed
    class_weight = args.class_weight
    loss_fc = args.loss
    pooling = args.pooling
    encoder_type = args.encoder
    freeze_embedding = True if args.freeze_embedding else False
    n_heads = args.n_heads
    n_layers = args.n_layers
    bidirectional = True if args.bidirectional else False
    label_schema = args.label_schema
    n_eopochs = args.n_epochs
    save_best_model = True if args.save_best_model else False
    output_dir = args.output_dir
    
    # Assert the scenario arguments
    assert_baseline(args)

    # Prepare the experiment scenario directory to store the results and logs
    root_workdir = os.path.join('experiments', output_dir, args.dataset)
    if not os.path.exists(root_workdir):
        os.makedirs(root_workdir)

    scenario_dir = os.path.join(encoder_type, class_weight, loss_fc, pooling, str(
        n_layers), str(n_heads), 'bidirectional' if bidirectional else 'unidirectional', label_schema)
    workdir = os.path.join(root_workdir, scenario_dir)
    print('[multitask] - Current Workdir: ', workdir)
    if not os.path.exists(workdir):
        os.makedirs(workdir)
        
    if os.path.exists(os.path.join(workdir, 'scenario_arguments.json')):
        print('The scenario has been executed')
        return 0

    # Set global seed for reproducibility
    set_seed(42)

    idx2label = {
        1: 'normal',
        2: 'low',
        3: 'medium',
        4: 'high'
    }

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    if args.dataset == 'filtered':
        dataset_path = 'dataset/merged-manual-unique.csv'
        dataset = pd.read_csv(dataset_path)
        dataset["label"] = dataset['label'].map(idx2label)
        label_encoder_multi = LabelEncoder()
        dataset["multiclass_label"] = label_encoder_multi.fit_transform(
            dataset["label"].to_list())
        train_path = 'dataset/fold_5_train.csv'
        test_path = 'dataset/fold_5_test.csv'
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        if label_schema == '101':
            train_df['multi_task_label'] = train_df["label"].apply(label2multitask101)
            test_df['multi_task_label'] = test_df["label"].apply(label2multitask101)
        elif label_schema == '111':
            train_df['multi_task_label'] = train_df["label"].apply(label2multitask111)
            test_df['multi_task_label'] = test_df["label"].apply(label2multitask111)
    elif args.dataset == 'unfiltered':
        dataset_path = 'dataset/merged-manual-unfiltered.csv'
        dataset = pd.read_csv(dataset_path)
        dataset["label"] = dataset['label'].map(idx2label)
        label_encoder_multi = LabelEncoder()
        dataset["multiclass_label"] = label_encoder_multi.fit_transform(
            dataset["label"].to_list())
        if label_schema == '101':
            dataset["multi_task_label"] = dataset["label"].apply(label2multitask101)
        elif label_schema == '111':
            dataset["multi_task_label"] = dataset["label"].apply(label2multitask111)
    else:
        raise SystemExit("Dataset is invalid. Please use valid dataset.")

    # Compute class weights
    if class_weight == 'balanced':
        class_weights = compute_class_weight('balanced', classes=[0, 1, 2, 3], y=dataset["multiclass_label"].to_list())
    elif class_weight == 'inverse':
        class_weights = inverse_freq(dataset["multiclass_label"].to_list())
    else:   # uniform
        class_weights = np.ones([4])

    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Split the data into train and test sets
    if args.dataset != 'filtered':
        train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)

    # Check the split results, if the last batch contains only 1 instance
    train_df = assert_data_size(train_df, 8)
    test_df = assert_data_size(test_df, 8)
    
    bert_model_name = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name).to(device)

    # Define the custom dataset and dataloaders
    max_seq_length = 64
    batch_size = 8
    train_dataset = MultitaskDataset(train_df, tokenizer, max_seq_length)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MultitaskDataset(test_df, tokenizer, max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize the model
    num_classes_multiclass = len(label_encoder_multi.classes_)  # Number of classes for multiclass classification
    
    # Instantiate the model based on passed arguments
    lstm_hidden_size = int(bert_model.config.hidden_size /
                           2) if bidirectional else bert_model.config.hidden_size
    model = DroneLog(bert_model, encoder_type,
                                n_heads, n_layers, freeze_embedding, bidirectional, lstm_hidden_size, pooling, num_classes_multiclass).to(device)

    # Define loss functions and optimizer
    if loss_fc == 'cross_entropy':
        criterion_multiclass = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction='mean')
    elif loss_fc == 'focal':
        criterion_multiclass = FocalLoss(alpha=class_weights.to(device), gamma=2)
    elif loss_fc == 'logloss':
        criterion_multiclass = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device), reduction='sum')
    else:
        raise SystemExit("The loss function is not supported.")
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # Lists to store training and evaluation metrics
    train_loss_history = []
    train_accuracy_history = []
    train_f1_history = []
    val_loss_history = []
    val_accuracy_history = []
    val_f1_history = []

    # Training loop
    num_epochs = n_eopochs  # You can adjust this
    train_started_at = datetime.datetime.now()
    print(f"[multitask] - {train_started_at} - Start Training...\n")
    best_model_state = None  # Initialize as None
    best_validation_metric = float('-inf')
    best_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        train_epoch_labels = []
        train_epoch_preds = []
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            class_label_train = batch["class_label"]
            labels_multiclass_train = batch["labels_multitask"].to(device)

            optimizer.zero_grad()

            logits_multiclass = model(input_ids, attention_mask)

            # Calculate losses
            loss_multiclass_train = criterion_multiclass(logits_multiclass, labels_multiclass_train)
            loss_multiclass_train.backward()
            optimizer.step()

            # Calculate training accuracy
            preds_multiclass_train = []
            for logits in logits_multiclass:
                after_sigmoid = [1 if (torch.sigmoid(element) >= 0.5) else 0 for element in logits]
                string_label = label2class(after_sigmoid)
                preds_multiclass_train.append(string_label)
            
            train_epoch_labels.extend(class_label_train)
            train_epoch_preds.extend(preds_multiclass_train)
            total_train_loss += loss_multiclass_train.item()
            
        train_loss_epoch = total_train_loss / len(train_loader)
        train_acc_epoch = accuracy_score(train_epoch_labels, train_epoch_preds)
        train_f1_epoch = f1_score(train_epoch_labels, train_epoch_preds, average='micro')
        train_loss_history.append(train_loss_epoch)
        train_accuracy_history.append(train_acc_epoch)
        train_f1_history.append(train_f1_epoch)

        # In training Evaluation
        model.eval()
        total_val_loss = 0.0
        val_epoch_labels = []
        val_epoch_preds = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                val_labels_class = batch["class_label"]
                labels_multiclass_val = batch["labels_multitask"].to(device)

                logits_multiclass_val = model(input_ids, attention_mask)
                loss_multiclass_val = criterion_multiclass(
                    logits_multiclass_val.squeeze(), labels_multiclass_val)
                total_val_loss += loss_multiclass_val.item()
                preds_multiclass_val = []
                for logits in logits_multiclass_val:
                    after_sigmoid = [1 if (torch.sigmoid(element) >= 0.5) else 0 for element in logits]
                    string_label = label2class(after_sigmoid)
                    preds_multiclass_val.append(string_label)
                    
                val_epoch_labels.extend(val_labels_class)
                val_epoch_preds.extend(preds_multiclass_val)
                
        val_loss_epoch = total_val_loss / len(test_loader)
        val_acc_epoch = accuracy_score(val_epoch_labels, val_epoch_preds)
        val_f1_epoch = f1_score(val_epoch_labels, val_epoch_preds, average='micro')
        val_loss_history.append(val_loss_epoch)
        val_accuracy_history.append(val_acc_epoch)
        val_f1_history.append(val_f1_epoch)
        print(f"{epoch+1}/{num_epochs}: train_loss: {total_train_loss} - val_loss: {val_loss_epoch} - train_f1: {train_f1_epoch} - val_f1: {val_f1_epoch}")

        # Check if the current model is the best
        if val_f1_epoch > best_validation_metric:
            best_validation_metric = val_f1_epoch
            # Save the model's state (weights and other parameters)
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
    
    # Save the train and validation logs to files
    # Plot and save the training and evaluation metrics as PDF files
    train_finished_at = datetime.datetime.now()
    epochs = range(1, num_epochs + 1)
    
    # Training and test loss plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss_history, label="Train Loss")
    plt.plot(epochs, val_loss_history, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plot_loss = plt.gca()
    plot_loss.get_figure().savefig(os.path.join(workdir, "train_val_loss.pdf"), format='pdf', bbox_inches='tight')

    # Training and test accuracy plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_accuracy_history, label="Train Accuracy")
    plt.plot(epochs, val_accuracy_history, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plot_accuracy = plt.gca()
    plot_accuracy.get_figure().savefig(os.path.join(workdir, "train_val_acc.pdf"), format='pdf', bbox_inches='tight')

    # Training and test F1 plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_f1_history, label="Train F1 score")
    plt.plot(epochs, val_f1_history, label="Val F1 score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 score")
    plt.title("Training and Validation F1 score")
    plt.legend()
    plt.tight_layout()
    # Save the plots as PDF files
    plot_f1 = plt.gca()
    plot_f1.get_figure().savefig(os.path.join(workdir, "train_val_f1.pdf"), format='pdf', bbox_inches='tight')

    # Evaluation
    best_model = DroneLog(bert_model, encoder_type,
                                     n_heads, n_layers, freeze_embedding, bidirectional, lstm_hidden_size, pooling, num_classes_multiclass).to(device)
    best_model.load_state_dict(best_model_state)
    best_model.eval()
    all_labels_multiclass = []
    all_preds_multiclass = []
    eval_started_at = datetime.datetime.now()
    print(f"\n[multitask] - {eval_started_at} - Start evaluation...\n")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_index = batch["class_label"]

            logits_multiclass_test = best_model(input_ids, attention_mask)
            predicted_labels_multiclass_test = []
            for logits in logits_multiclass_test:
                after_sigmoid = [1 if (torch.sigmoid(element) >= 0.5) else 0 for element in logits]
                string_label = label2class(after_sigmoid)
                predicted_labels_multiclass_test.append(string_label)

            all_labels_multiclass.extend(labels_index)
            all_preds_multiclass.extend(predicted_labels_multiclass_test)
            
    # Calculate multiclass classification accuracy and report
    # Save the input, label, and preds for error analysis
    prediction_df = pd.DataFrame()
    prediction_df["message"] = test_df["message"]
    prediction_df["label"] = list(all_labels_multiclass)
    prediction_df["pred"] = list(all_preds_multiclass)
    prediction_df.to_csv(os.path.join(
        workdir, "prediction.csv"), index=False)

    accuracy = accuracy_score(all_labels_multiclass, all_preds_multiclass)
    evaluation_report = classification_report(
        all_labels_multiclass, all_preds_multiclass, digits=5)
    # Calculate multiclass classification report
    classification_report_result = classification_report(
        all_labels_multiclass, all_preds_multiclass, digits=5, output_dict=True)
    classification_report_result['macro_avg'] = classification_report_result.pop('macro avg')
    classification_report_result['weighted_avg'] = classification_report_result.pop('weighted avg')
    micro_pre, micro_rec, micro_f1, support = precision_recall_fscore_support(all_labels_multiclass, all_preds_multiclass, average='micro')
    classification_report_result['micro_avg'] = {
        "precision": micro_pre,
        "recall": micro_rec,
        "f1-score": micro_f1
        }

    # Export the dictionary to a JSON file
    with open(os.path.join(workdir, "evaluation_report.json"), 'w') as json_file:
        json.dump(classification_report_result, json_file, indent=4)
    with open(os.path.join(workdir, "evaluation_report.txt"), "w") as text_file:
        text_file.write(evaluation_report)

    # print("Binary Classification Accuracy:", binary_accuracy)
    print("Classification Report:\n", evaluation_report)
    eval_finished_at = datetime.datetime.now()
    print(f"[multitask] - {eval_finished_at} - Finish...\n")

    arguments_dict = vars(args)
    arguments_dict['scenario_dir'] = workdir
    arguments_dict['best_epoch'] = best_epoch
    arguments_dict['train_started_at'] = str(train_started_at)
    arguments_dict['train_finished_at'] = str(train_finished_at)
    train_duration = train_finished_at - train_started_at
    arguments_dict['train_duration'] = str(train_duration.total_seconds()) + ' seconds'
    arguments_dict['eval_started_at'] = str(eval_started_at)
    arguments_dict['eval_finished_at'] = str(eval_finished_at)
    eval_duration = eval_finished_at - eval_started_at
    arguments_dict['eval_duration'] = str(eval_duration.total_seconds()) + ' seconds'
    
    with open(os.path.join(workdir, 'scenario_arguments.json'), 'w') as json_file:
        json.dump(arguments_dict, json_file, indent=4)
        
    # Save the model
    if save_best_model:
        best_model_dir = os.path.join('best_models', output_dir, args.dataset, args.encoder)
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
            
            # Save the experimental logs
            plot_loss.get_figure().savefig(os.path.join(best_model_dir, "train_val_loss.pdf"), format='pdf', bbox_inches='tight')
            plot_accuracy.get_figure().savefig(os.path.join(best_model_dir, "train_val_acc.pdf"), format='pdf', bbox_inches='tight')
            plot_f1.get_figure().savefig(os.path.join(best_model_dir, "train_val_f1.pdf"), format='pdf', bbox_inches='tight')
            prediction_df.to_csv(os.path.join(best_model_dir, "prediction.csv"), index=False)
            with open(os.path.join(best_model_dir, "evaluation_report.json"), 'w') as json_file:
                json.dump(classification_report_result, json_file, indent=4)
            with open(os.path.join(best_model_dir, "evaluation_report.txt"), "w") as text_file:
                text_file.write(evaluation_report)
            with open(os.path.join(best_model_dir, 'scenario_arguments.json'), 'w') as json_file:
                json.dump(arguments_dict, json_file, indent=4)
                
            # Save the model's file
            torch.save(best_model_state, os.path.join(best_model_dir, 'pytorch_model.pt'))
        else:
            # Check the previous best and compare to current model's performance
            eval_report_path = os.path.join(best_model_dir, "evaluation_report.json")
            with open(eval_report_path) as eval_report_file:
                eval_report = json.load(eval_report_file)
                if accuracy > eval_report['accuracy']:
                    # Save the experimental logs
                    plot_loss.get_figure().savefig(os.path.join(best_model_dir, "train_val_loss.pdf"), format='pdf', bbox_inches='tight')
                    plot_accuracy.get_figure().savefig(os.path.join(best_model_dir, "train_val_acc.pdf"), format='pdf', bbox_inches='tight')
                    plot_f1.get_figure().savefig(os.path.join(best_model_dir, "train_val_f1.pdf"), format='pdf', bbox_inches='tight')
                    prediction_df.to_csv(os.path.join(best_model_dir, "prediction.csv"), index=False)
                    with open(os.path.join(best_model_dir, "evaluation_report.json"), 'w') as json_file:
                        json.dump(classification_report_result, json_file, indent=4)
                    with open(os.path.join(best_model_dir, "evaluation_report.txt"), "w") as text_file:
                        text_file.write(evaluation_report)
                    with open(os.path.join(best_model_dir, 'scenario_arguments.json'), 'w') as json_file:
                        json.dump(arguments_dict, json_file, indent=4)
                        
                    # Save the model's file
                    torch.save(best_model_state, os.path.join(best_model_dir, 'pytorch_model.pt'))
    
    return 0


if __name__ == "__main__":
    main()
