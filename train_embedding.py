import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Step 1: Load a pre-trained model
model_name = 'all-mpnet-base-v2'  # or 'hkunlp/instructor-xl'
model = SentenceTransformer(model_name).to(device)

# Step 2: Prepare the dataset
class DroneLogsDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return InputExample(texts=[row['message']], label=row['cluster_id'])

# Load your dataset
df = pd.read_csv(os.path.join('dataset', 'filtered_train.csv'))  # Assume the CSV has 'message' and 'cluster_id' columns

# Create pairs for contrastive learning
def create_pairs(df):
    examples = []
    for label in df['label'].unique():
        cluster_df = df[df['label'] == label]
        other_df = df[df['label'] != label]
        for i, row in cluster_df.iterrows():
            for j, other_row in cluster_df.iterrows():
                if i != j:
                    examples.append(InputExample(texts=[row['message'], other_row['message']], label=1.0))
            for j, other_row in other_df.iterrows():
                examples.append(InputExample(texts=[row['message'], other_row['message']], label=0.0))
    return examples

examples = create_pairs(df)
# Step 3: Create DataLoader
train_dataloader = DataLoader(examples, shuffle=True, batch_size=64)
# print(train_dataloader)

# Step 4: Define the contrastive loss
train_loss = losses.ContrastiveLoss(model=model)

# Optional: Define evaluator for validation
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(examples, name='drone-log-eval')

# Step 5: Train the model
num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
output_path = os.path.join('experiments', 'embeddings')
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=output_path
)

# Save the model
model.save(output_path, 'drone_log_semantic')
