# Assume Task 2 code (MultiTaskSentenceTransformer and SentenceTransformerModel)

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel # For base model components
import torch.nn.functional as F # For base model components / normalization
import logging
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score
import random # For generating dummy data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SentenceTransformerModel(torch.nn.Module):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', pooling_strategy: str = 'mean'):
        super(SentenceTransformerModel, self).__init__()
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.transformer_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_dim = self.transformer_model.config.hidden_size
        if pooling_strategy not in ['mean', 'cls', 'max']:
            raise ValueError("pooling_strategy must be one of 'mean', 'cls', or 'max'")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * input_mask_expanded
        sum_embeddings = torch.sum(masked_embeddings, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _cls_pooling(self, model_output):
        return model_output.last_hidden_state[:, 0]

    def _max_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]

    # Modified forward for potential fine-tuning (removes internal no_grad)
    def forward_for_tuning(self, sentences: List[str], normalize: bool = False) -> torch.Tensor:
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
        # Assume tensors on correct device
        model_output = self.transformer_model(**encoded_input)

        if self.pooling_strategy == 'mean':
            sentence_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
        elif self.pooling_strategy == 'cls':
            sentence_embedding = self._cls_pooling(model_output)
        elif self.pooling_strategy == 'max':
            sentence_embedding = self._max_pooling(model_output, encoded_input['attention_mask'])
        else:
             raise ValueError("Invalid pooling strategy")

        if normalize:
            sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)

        return sentence_embedding
# --- MultiTaskSentenceTransformer from Task 2 (Modified to allow backbone training) ---
class MultiTaskSentenceTransformer(torch.nn.Module):
    def __init__(self,
                 base_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 pooling_strategy: str = 'mean',
                 num_classes_task_a: int = 4,
                 num_classes_task_b: int = 3,
                 freeze_backbone: bool = False): # Added freeze option
        super(MultiTaskSentenceTransformer, self).__init__()
        self.freeze_backbone = freeze_backbone

        # Shared Backbone
        self.shared_backbone = SentenceTransformerModel(
            model_name=base_model_name,
            pooling_strategy=pooling_strategy
        )
        embedding_dim = self.shared_backbone.embedding_dim

        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in self.shared_backbone.parameters():
                param.requires_grad = False
            logging.info("Shared backbone parameters frozen.")
        else:
             logging.info("Shared backbone parameters are trainable.")

        # Task Heads
        self.classification_head = torch.nn.Linear(embedding_dim, num_classes_task_a)
        self.num_classes_task_a = num_classes_task_a
        self.sentiment_head = torch.nn.Linear(embedding_dim, num_classes_task_b)
        self.num_classes_task_b = num_classes_task_b

    def forward(self, sentences: List[str]) -> Dict[str, torch.Tensor]:
        shared_embedding = self.shared_backbone.forward_for_tuning(sentences, normalize=False)
        logits_classification = self.classification_head(shared_embedding) # Task A
        logits_sentiment = self.sentiment_head(shared_embedding)         # Task B

        return {
            'classification': logits_classification,
            'sentiment': logits_sentiment
        }

# --- Helper Function for Accuracy ---
def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculates accuracy given logits and labels."""
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    accuracy = correct / labels.size(0) if labels.size(0) > 0 else 0.0
    return accuracy

# --- Task 4: Training Loop Implementation ---

def train_loop_demo(model: MultiTaskSentenceTransformer,
                    data: List[Tuple[List[str], torch.Tensor, torch.Tensor]],
                    loss_fn_A: nn.Module,
                    loss_fn_B: nn.Module,
                    optimizer: optim.Optimizer,
                    loss_weight_A: float = 1.0,
                    loss_weight_B: float = 1.0,
                    device: torch.device = torch.device("cpu")):
    model.to(device)
    model.train() # Set model to training mode

    total_loss_epoch = 0.0
    total_acc_A_epoch = 0.0
    total_acc_B_epoch = 0.0
    num_batches = len(data)

    logging.info("--- Starting Training Loop Demo (1 Epoch) ---")

    for batch_idx, (sentences, labels_A, labels_B) in enumerate(data):
        # Move labels to device
        labels_A = labels_A.to(device)
        labels_B = labels_B.to(device)

        optimizer.zero_grad()

        outputs = model(sentences)
        logits_A = outputs['classification']
        logits_B = outputs['sentiment']

        loss_B = loss_fn_B(logits_B, labels_B)

        total_loss = (loss_weight_A * loss_A) + (loss_weight_B * loss_B)

        total_loss.backward()

        optimizer.step()

        acc_A = calculate_accuracy(logits_A, labels_A)
        acc_B = calculate_accuracy(logits_B, labels_B)

        total_loss_epoch += total_loss.item()
        total_acc_A_epoch += acc_A
        total_acc_B_epoch += acc_B

        if (batch_idx + 1) % 1 == 0: # Print every batch for demo
             logging.info(f"  Batch {batch_idx + 1}/{num_batches} | "
                          f"Loss A: {loss_A.item():.4f} | Loss B: {loss_B.item():.4f} | "
                          f"Total Loss: {total_loss.item():.4f} | "
                          f"Acc A: {acc_A:.4f} | Acc B: {acc_B:.4f}")

    # --- Epoch Summary ---
    avg_loss = total_loss_epoch / num_batches
    avg_acc_A = total_acc_A_epoch / num_batches
    avg_acc_B = total_acc_B_epoch / num_batches

    logging.info("--- Epoch Demo Complete ---")
    logging.info(f"Average Loss: {avg_loss:.4f}")
    logging.info(f"Average Accuracy Task A: {avg_acc_A:.4f}")
    logging.info(f"Average Accuracy Task B: {avg_acc_B:.4f}")


# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Assumptions and Decisions ---
    print("--- Task 4: Training Loop Assumptions ---")
    print("1. Hypothetical Data: Using randomly generated labels and sample sentences.")
    print("2. Loss Combination: Using a simple weighted sum (defaulting to equal weights).")
    print("3. Optimizer: Using AdamW (common choice for transformers).")
    print("4. Metrics: Using basic Accuracy for demonstration.")
    print("5. Training Mode: Model set to 'train' mode (`model.train()`).")
    print("6. Backbone State: Demonstrating with a *trainable* backbone (freeze_backbone=False).")
    print("7. Device: Assuming CPU for simplicity.")
    print("8. Training Duration: Simulating only *one* pass over the data (one epoch).")
    print("-" * 40)

    LEARNING_RATE = 2e-5
    EPOCHS = 1 # Demo only one epoch
    BATCH_SIZE = 2 # Small batch size for demo
    NUM_CLASSES_A = 4
    NUM_CLASSES_B = 3
    LOSS_WEIGHT_A = 1.0 # Equal weighting example
    LOSS_WEIGHT_B = 1.0
    FREEZE_BACKBONE_DEMO = False # Set to True to test frozen backbone training

    model = MultiTaskSentenceTransformer(
        num_classes_task_a=NUM_CLASSES_A,
        num_classes_task_b=NUM_CLASSES_B,
        freeze_backbone=FREEZE_BACKBONE_DEMO
    )

    loss_fn_A = nn.CrossEntropyLoss()
    loss_fn_B = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    sample_sentences_all = [
        "How do I install this library?",
        "The interface is really intuitive!",
        "Can you explain multi-task learning?",
        "This service keeps crashing.",
        "What time is it?",
        "Your documentation needs improvement.",
        "This is fantastic!",
        "Just Browse around.",
    ]
    num_samples = len(sample_sentences_all)

    dummy_labels_A = torch.randint(0, NUM_CLASSES_A, (num_samples,))
    dummy_labels_B = torch.randint(0, NUM_CLASSES_B, (num_samples,))

    demo_data = []
    for i in range(0, num_samples, BATCH_SIZE):
        batch_sentences = sample_sentences_all[i:i+BATCH_SIZE]
        batch_labels_A = dummy_labels_A[i:i+BATCH_SIZE]
        batch_labels_B = dummy_labels_B[i:i+BATCH_SIZE]
        if not batch_sentences: continue # Skip empty batch if num_samples not divisible by BATCH_SIZE
        demo_data.append((batch_sentences, batch_labels_A, batch_labels_B))

    # --- Run the Training Loop Demo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    train_loop_demo(model, demo_data, loss_fn_A, loss_fn_B, optimizer, LOSS_WEIGHT_A, LOSS_WEIGHT_B, device)