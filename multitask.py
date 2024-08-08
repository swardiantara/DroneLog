import os
import copy
import json
import torch
import random
import datetime
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.dronelog import DroneLog
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from utils.losses import inverse_freq, FocalLoss
from utils.assert_scenario import assert_baseline
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from utils.postprocessing import visualize_projection, label2class, visualize_prediction
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_recall_fscore_support
from utils.preprocessing import BaselineDataset, MultitaskDataset, assert_data_size, label2multitask110, label2multitask101, label2multitask111, label2multitask0101, label2multitask0111

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='filtered',
                    choices=['filtered', 'unfiltered'])
parser.add_argument('--output_dir', type=str, default='multi_dronelog',
                    help="Folder to store the experimental results. Default: multitask")
parser.add_argument('--word_embed', type=str, choices=['bert', 'drone-severity'], default='bert', help='Type of Word Embdding used. Default: BERT-base')
parser.add_argument('--encoder', type=str, choices=['transformer', 'lstm', 'gru', 'none'], default='none',
                    help="Encoder Architecture used to perform computation. Default: none.")
parser.add_argument('--pooling', type=str, choices=['cls', 'max', 'avg', 'last'], default='avg',
                    help="Pooling mechanism to get final representation. Default: avg")
parser.add_argument('--bidirectional', action='store_true',
                    help="Wether to use Bidirectionality for LSTM and GRU.")
parser.add_argument('--save_best_model', action='store_true',
                    help="Wether to save best model for each encoder type.")
parser.add_argument('--viz_projection', action='store_true',
                    help="Wether to visualize the encoder's output.")
parser.add_argument('--class_weight', choices=['uniform', 'balanced', 'inverse'], default='uniform',
                    help="Wether to weigh the class based on the class frequency. Default: Uniform")
parser.add_argument('--label_schema', choices=['110', '101', '111', '0101', '0111'], default='110',
                    help="Target label schema. Default: 101.")
parser.add_argument('--loss', choices=['logloss'], default='logloss',
                    help="Loss function to use. Default: logloss")
parser.add_argument('--n_heads', type=int, default=1,
                    help='Number of attention heads')
parser.add_argument('--n_layers', type=int, default=1,
                    help='Number of encoder layers')
parser.add_argument('--n_epochs', type=int, default=15,
                    help='Number of training iterations')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Number of samples in a batch')


# Arguments for Ablation study
parser.add_argument('--exclude_cls_before', action='store_true', help="Wether to include CLS token representation before encoder.")
parser.add_argument('--exclude_cls_after', action='store_true', help="Wether to include CLS token representation after encoder.")
parser.add_argument('--freeze_embedding', action='store_true', help="Wether to freeze the pre-trained embedding's parameter.")
parser.add_argument('--normalize_logits', action='store_true', help="Wether to normalize the logits during training.")

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
    # print(f"Random seed set as {seed}")


def main():
    # Set global seed for reproducibility
    set_seed(42)
    
    # Parse and load the arguments
    args = parser.parse_args()
    embedding_type = args.word_embed
    viz_projection = args.viz_projection
    class_weight = args.class_weight
    loss_fc = args.loss
    pooling = args.pooling
    encoder_type = args.encoder
    freeze_embedding = True if args.freeze_embedding else False 
    n_heads = args.n_heads
    n_layers = args.n_layers
    n_epochs = args.n_epochs
    bidirectional = True if args.bidirectional else False
    label_schema = args.label_schema
    save_best_model = True if args.save_best_model else False
    output_dir = args.output_dir + '_' + str(n_epochs)
    normalize_logits = True if args.normalize_logits else False
    exclude_cls_before = True if args.exclude_cls_before else False
    exclude_cls_after = True if args.exclude_cls_after else False

    # Assert the scenario arguments
    assert_baseline(args)

    # Prepare the experiment scenario directory to store the results and logs
    root_workdir = os.path.join('experiments', output_dir, args.dataset)
    if not os.path.exists(root_workdir):
        os.makedirs(root_workdir)

    scenario_dir = os.path.join(encoder_type, class_weight, loss_fc, pooling, str(
        n_layers), str(n_heads), 'bidirectional' if bidirectional else 'unidirectional', label_schema)
    workdir = os.path.join(root_workdir, scenario_dir)
    print(f'[multitask-{label_schema}] - Current Workdir: {workdir}')
    if not os.path.exists(workdir):
        os.makedirs(workdir)
        
    if os.path.exists(os.path.join(workdir, 'scenario_arguments.json')):
        print('The scenario has been executed')
        return 0

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
        train_path = 'dataset/filtered_train.csv'
        test_path = 'dataset/filtered_test.csv'
    elif args.dataset == 'unfiltered':
        dataset_path = 'dataset/merged-manual-unfiltered.csv'
        train_path = 'dataset/unfiltered_train.csv'
        test_path = 'dataset/unfiltered_test.csv'
    else:
        raise SystemExit("The dataset option is not valid.")
    label_encoder_multi = LabelEncoder()
    dataset = pd.read_csv(dataset_path)
    dataset["label"] = dataset['label'].map(idx2label)
    dataset["multiclass_label"] = label_encoder_multi.fit_transform(dataset["label"].to_list())
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if label_schema == '101':
        train_df['multi_task_label'] = train_df["label"].apply(label2multitask101)
        test_df['multi_task_label'] = test_df["label"].apply(label2multitask101)
    elif label_schema == '110':
        train_df['multi_task_label'] = train_df["label"].apply(label2multitask110)
        test_df['multi_task_label'] = test_df["label"].apply(label2multitask110)
    elif label_schema == '111':
        train_df['multi_task_label'] = train_df["label"].apply(label2multitask111)
        test_df['multi_task_label'] = test_df["label"].apply(label2multitask111)
    elif label_schema == '0101':
        train_df['multi_task_label'] = train_df["label"].apply(label2multitask0101)
        test_df['multi_task_label'] = test_df["label"].apply(label2multitask0101)
    elif label_schema == '0111':
        train_df['multi_task_label'] = train_df["label"].apply(label2multitask0111)
        test_df['multi_task_label'] = test_df["label"].apply(label2multitask0111)
    else:
        raise NotImplementedError("label_schema is invalid!")

    # Compute the class weights
    if class_weight == 'balanced':
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2, 3]), y=dataset["multiclass_label"].to_list())
    elif class_weight == 'inverse':
        class_weights = inverse_freq(dataset["multiclass_label"].to_list())
    else:   # uniform
        class_weights = np.ones([4])
    
    # Convert class weights to a PyTorch tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Check the dataset, if the last batch contains only 1 instance
    train_df = assert_data_size(train_df, args.batch_size)
    test_df = assert_data_size(test_df, args.batch_size)
    
    if embedding_type == 'bert':
        bert_model_name = "bert-base-cased"
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        bert_model = BertModel.from_pretrained(bert_model_name).to(device)
    elif embedding_type == 'drone-severity':
        bert_model_name = "swardiantara/drone-severity-embedding"
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert_model = AutoModel.from_pretrained(bert_model_name).to(device)
    else:
        raise SystemExit("The embedding is not supported.")

    # Define the custom dataset and dataloaders
    max_seq_length = 64
    batch_size = args.batch_size

    merged_dataset = BaselineDataset(dataset, tokenizer, max_seq_length)
    merged_loader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=False)

    train_dataset = MultitaskDataset(train_df, tokenizer, max_seq_length)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MultitaskDataset(test_df, tokenizer, max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    num_classes_multiclass = len(label_encoder_multi.classes_)
    # Instantiate the model based on passed arguments
    lstm_hidden_size = int(bert_model.config.hidden_size /
                           2) if bidirectional else bert_model.config.hidden_size
    model = DroneLog(bert_model, encoder_type,
                                n_heads, n_layers, freeze_embedding, bidirectional, lstm_hidden_size, pooling, exclude_cls_before, exclude_cls_after, num_classes_multiclass, None, normalize_logits, loss_fc).to(device)

    # Define loss functions and optimizer
    if loss_fc == 'cross_entropy':
        criterion_multiclass = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction='mean')
    elif loss_fc == 'focal':
        criterion_multiclass = FocalLoss(alpha=class_weights.to(device), gamma=2)
    elif loss_fc == 'logloss':
        criterion_multiclass = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device), reduction='mean')
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
    num_epochs = n_epochs
    train_started_at = datetime.datetime.now()
    print(f"[multitask-{label_schema}] - {train_started_at} - Start Training...\n")
    best_model_state = None  # Initialize as None
    best_acc_epoch = float('-inf')
    best_f1_epoch = float('-inf')
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

            _, logits_multiclass, _ = model(input_ids, attention_mask)
            # if normalize_logits == True:
            #     logits_multiclass = torch.softmax(logits_multiclass, axis=1)
            loss_multiclass_train = criterion_multiclass(logits_multiclass, labels_multiclass_train)
            loss_multiclass_train.backward()
            optimizer.step()

            # Calculate training accuracy
            preds_multiclass_train = []
            for logits in logits_multiclass:
                after_sigmoid = torch.sigmoid(logits)
                # after_sigmoid = [1 if (torch.sigmoid(element) >= 0.5) else 0 for element in logits]
                string_label = label2class([1 if element >= 0.5 else 0 for element in after_sigmoid])
                preds_multiclass_train.append(string_label)
            train_epoch_labels.extend(class_label_train)
            train_epoch_preds.extend(preds_multiclass_train)

            total_train_loss += loss_multiclass_train.item()

        # Logs the train loss, acc, and f1
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

                _, logits_multiclass_val, _ = model(input_ids, attention_mask)
                # if normalize_logits == True:
                #     logits_multiclass_val = torch.softmax(logits_multiclass_val, axis=1)
                loss_multiclass_val = criterion_multiclass(
                    logits_multiclass_val, labels_multiclass_val)
                
                # Log the val preds
                total_val_loss += loss_multiclass_val.item()
                preds_multiclass_val = []
                for logits in logits_multiclass_val:
                    after_sigmoid = [1 if (torch.sigmoid(element) >= 0.5) else 0 for element in logits]
                    string_label = label2class(after_sigmoid)
                    preds_multiclass_val.append(string_label)
                val_epoch_labels.extend(val_labels_class)
                val_epoch_preds.extend(preds_multiclass_val)
        
        # Logs the val loss, acc, and f1
        val_loss_epoch = total_val_loss / len(test_loader)
        val_acc_epoch = accuracy_score(val_epoch_labels, val_epoch_preds)
        val_f1_epoch = f1_score(val_epoch_labels, val_epoch_preds, average='weighted')
        val_loss_history.append(val_loss_epoch)
        val_accuracy_history.append(val_acc_epoch)
        val_f1_history.append(val_f1_epoch)
        print(f"{epoch+1}/{num_epochs}: train_loss: {total_train_loss} - val_loss: {val_loss_epoch} - train_f1: {train_f1_epoch} - val_f1: {val_f1_epoch}")

        # Check if the current epoch is the best
        if (val_f1_epoch > best_f1_epoch and val_acc_epoch > best_acc_epoch) or (val_f1_epoch > best_f1_epoch and val_acc_epoch >= best_acc_epoch) or (val_f1_epoch >= best_f1_epoch and val_acc_epoch > best_acc_epoch):
            best_f1_epoch = val_f1_epoch
            best_acc_epoch = val_acc_epoch
            # Save the model's state (weights and other parameters)
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
    
    train_finished_at = datetime.datetime.now()
    
    # Save the train and validation logs to files
    # Plot and save the training and evaluation metrics as PDF files
    epochs = range(1, num_epochs + 1)

    # Train and Val loss plot
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
    plot_f1 = plt.gca()
    plot_f1.get_figure().savefig(os.path.join(workdir, "train_val_f1.pdf"), format='pdf', bbox_inches='tight')

    # Evaluation
    best_model = DroneLog(bert_model, encoder_type,
                                     n_heads, n_layers, freeze_embedding, bidirectional, lstm_hidden_size, pooling, exclude_cls_before, exclude_cls_after, num_classes_multiclass, None, normalize_logits, loss_fc).to(device)
    best_model.load_state_dict(best_model_state)
    best_model.eval()
    label_order = ['high', 'low','medium', 'normal']
    all_labels_multiclass = []
    all_preds_multiclass = []
    all_preds_probs_multiclass = []
    pred_probs_multiclass = []
    eval_started_at = datetime.datetime.now()
    print(f"\n[multitask-{label_schema}] - {eval_started_at} - Start evaluation...\n")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_index = batch["class_label"]

            _, logits_multiclass_test, _ = best_model(input_ids, attention_mask)

            predicted_labels_multiclass_test = []
            predicted_probs_multiclass_test = []
            pred_probs_batch = []
            for logits in logits_multiclass_test:
                after_sigmoid = torch.sigmoid(logits)
                string_label = label2class([1 if element >= 0.5 else 0 for element in after_sigmoid])
                pred_prob_sample = after_sigmoid[label_order.index(string_label)]
                predicted_labels_multiclass_test.append(string_label)
                predicted_probs_multiclass_test.append(after_sigmoid.cpu().numpy().tolist())
                pred_probs_batch.append(pred_prob_sample.cpu().item())
            all_labels_multiclass.extend(labels_index)
            all_preds_multiclass.extend(predicted_labels_multiclass_test)
            all_preds_probs_multiclass.extend(predicted_probs_multiclass_test)
            pred_probs_multiclass.extend(pred_probs_batch)

    # Calculate multiclass classification accuracy and report
    # Save the input, label, and preds for error analysis
    prediction_df = pd.DataFrame()
    prediction_df["message"] = test_df["message"]
    prediction_df["label"] = list(all_labels_multiclass)
    prediction_df["pred"] = list(all_preds_multiclass)
    prediction_df["verdict"] = [label == pred for label, pred in zip(all_labels_multiclass, all_preds_multiclass)]
    prediction_df["prob"] = list(pred_probs_multiclass)
    for idx, label_name in enumerate(label_order):
        prediction_df[f"prob_{label_name}"] = [row[idx] for row in all_preds_probs_multiclass]
    prediction_df.to_excel(os.path.join(workdir, "prediction.xlsx"), index=False)

    accuracy = accuracy_score(all_labels_multiclass, all_preds_multiclass)
    evaluation_report = classification_report(
        all_labels_multiclass, all_preds_multiclass, digits=5)
    # Calculate multiclass classification report
    f1_weighted = f1_score(all_labels_multiclass, all_preds_multiclass, average='weighted')
    classification_report_result = classification_report(
        all_labels_multiclass, all_preds_multiclass, digits=5, output_dict=True)
    classification_report_result['macro_avg'] = classification_report_result.pop('macro avg')
    classification_report_result['weighted_avg'] = classification_report_result.pop('weighted avg')
    micro_pre, micro_rec, micro_f1, _ = precision_recall_fscore_support(all_labels_multiclass, all_preds_multiclass, average='micro')
    classification_report_result['micro_avg'] = {
        "precision": micro_pre,
        "recall": micro_rec,
        "f1-score": micro_f1
        }

    # Logs the evaluation results into files
    with open(os.path.join(workdir, "evaluation_report.json"), 'w') as json_file:
        json.dump(classification_report_result, json_file, indent=4)
    with open(os.path.join(workdir, "evaluation_report.txt"), "w") as text_file:
        text_file.write(evaluation_report)
    print("Best epoch: ", best_epoch)
    print("Accuracy:", accuracy)
    print("F1-score:", f1_weighted)
    print("Classification Report:\n", evaluation_report)
    eval_finished_at = datetime.datetime.now()
    print(f"[multitask-{label_schema}] - {eval_finished_at} - Finish...\n")

    arguments_dict = vars(args)
    arguments_dict['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    arguments_dict['scenario_dir'] = workdir
    arguments_dict['best_epoch'] = best_epoch 
    arguments_dict['best_val_acc'] = best_acc_epoch
    arguments_dict['best_val_f1'] = best_f1_epoch
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
    
    if viz_projection:
        # Save the model's hidden state to a 2D plot
        visualize_projection(merged_loader, label_encoder_multi, best_model.to(device), device, workdir)
        visualize_prediction(test_loader, best_model.to(device), device, workdir, prediction_df)
        # visualize_projection(merged_loader, label_encoder_multi, best_model.to(device), device, workdir, test_loader, prediction_df)
    
    # Save the best model for each dataset and encoder type
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
            
            if viz_projection:
                # Save the model's hidden state to a 2D plot
                visualize_projection(merged_loader, label_encoder_multi, best_model.to(device), device, best_model_dir)
                visualize_prediction(test_loader, best_model.to(device), device, best_model_dir, prediction_df)
                # visualize_projection(merged_loader, label_encoder_multi, best_model.to(device), device, best_model_dir, test_loader, prediction_df)
            # Save the model's file
            torch.save(best_model_state, os.path.join(best_model_dir, 'pytorch_model.pt'))
        else:
            # Check the previous best and compare to current model's performance
            eval_report_path = os.path.join(best_model_dir, "evaluation_report.json")
            with open(eval_report_path) as eval_report_file:
                eval_report = json.load(eval_report_file)
                if (accuracy > eval_report['accuracy'] and f1_weighted > eval_report['weighted_avg']['f1-score']) or (accuracy >= eval_report['accuracy'] and f1_weighted > eval_report['weighted_avg']['f1-score']) or (accuracy > eval_report['accuracy'] and f1_weighted >= eval_report['weighted_avg']['f1-score']):
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
                        
                    if viz_projection:
                        # Save the model's hidden state to a 2D plot
                        visualize_projection(merged_loader, label_encoder_multi, best_model.to(device), device, best_model_dir)
                        visualize_prediction(test_loader, best_model.to(device), device, best_model_dir, prediction_df)
                    # Save the model's file
                    torch.save(best_model_state, os.path.join(best_model_dir, 'pytorch_model.pt'))

        
    return 0


if __name__ == "__main__":
    main()
