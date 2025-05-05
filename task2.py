import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SentenceTransformerModel(torch.nn.Module):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', pooling_strategy: str = 'mean'):
        super(SentenceTransformerModel, self).__init__()
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        logging.info(f"Loading base model: {self.model_name}")
        self.transformer_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_dim = self.transformer_model.config.hidden_size # Get embedding dimension
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

    def forward(self, sentences: List[str], normalize: bool = True) -> torch.Tensor:
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)

        with torch.no_grad(): # Usually no grad needed if just using the base for embeddings
            model_output = self.transformer_model(**encoded_input)

        if self.pooling_strategy == 'mean':
            sentence_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
        elif self.pooling_strategy == 'cls':
             if 'bert' not in self.model_name.lower() and 'xlm' not in self.model_name.lower():
                 logging.warning(f"CLS pooling chosen for model '{self.model_name}', which might not be optimal unless fine-tuned for CLS tasks.")
             sentence_embedding = self._cls_pooling(model_output)
        elif self.pooling_strategy == 'max':
            sentence_embedding = self._max_pooling(model_output, encoded_input['attention_mask'])

        if normalize:
            sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)

        return sentence_embedding

# --- Task 2: Multi-Task Learning Expansion ---

class MultiTaskSentenceTransformer(torch.nn.Module):
    def __init__(self,
                 base_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 pooling_strategy: str = 'mean', # Or 'cls' based on Task 1 choice
                 num_classes_task_a: int = 4,     # Example: Sentence Classification
                 num_classes_task_b: int = 3):    # Example: Sentiment Analysis
        super(MultiTaskSentenceTransformer, self).__init__()

        # 1. Shared Backbone (Instantiate the Task 1 Model)
        # We don't normalize embeddings here, as heads are trained on unnormalized pooled features usually.
        self.shared_backbone = SentenceTransformerModel(
            model_name=base_model_name,
            pooling_strategy=pooling_strategy
        )
        # Freeze the backbone initially? This depends on Task 3 discussion.
        # For now, assume it might be trainable or partially frozen later.

        # Get the embedding dimension from the shared backbone
        embedding_dim = self.shared_backbone.embedding_dim

        # 2. Task-Specific Heads
        # Task A: Sentence Classification Head
        # Example Classes: "Technical Question", "General Inquiry", "Feedback/Suggestion", "Chit-chat"
        self.classification_head = torch.nn.Linear(embedding_dim, num_classes_task_a)
        self.num_classes_task_a = num_classes_task_a
        logging.info(f"Initialized Task A Head (Classification) with {num_classes_task_a} classes.")

        # Task B: Sentiment Analysis Head
        # Example Classes: "Positive", "Negative", "Neutral"
        self.sentiment_head = torch.nn.Linear(embedding_dim, num_classes_task_b)
        self.num_classes_task_b = num_classes_task_b
        logging.info(f"Initialized Task B Head (Sentiment Analysis) with {num_classes_task_b} classes.")

    def forward(self, sentences: List[str]) -> Dict[str, torch.Tensor]:
   
        encoded_input = self.shared_backbone.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
        model_output = self.shared_backbone.transformer_model(**encoded_input)

        # Apply chosen pooling strategy (without normalization yet)
        if self.shared_backbone.pooling_strategy == 'mean':
            shared_embedding = self.shared_backbone._mean_pooling(model_output, encoded_input['attention_mask'])
        elif self.shared_backbone.pooling_strategy == 'cls':
             shared_embedding = self.shared_backbone._cls_pooling(model_output)
        elif self.shared_backbone.pooling_strategy == 'max':
            shared_embedding = self.shared_backbone._max_pooling(model_output, encoded_input['attention_mask'])
        else:
             raise ValueError("Invalid pooling strategy") # Should not happen if init check passes

        logits_classification = self.classification_head(shared_embedding) # Task A
        logits_sentiment = self.sentiment_head(shared_embedding)         # Task B

        return {
            'classification': logits_classification,
            'sentiment': logits_sentiment
        }

# --- Testing the Multi-Task Implementation ---
if __name__ == "__main__":
    NUM_CLASSES_A = 4 # "Technical Question", "General Inquiry", "Feedback/Suggestion", "Chit-chat"
    NUM_CLASSES_B = 3 # "Positive", "Negative", "Neutral"

    mtl_model = MultiTaskSentenceTransformer(
        base_model_name='sentence-transformers/all-MiniLM-L6-v2',
        pooling_strategy='mean',
        num_classes_task_a=NUM_CLASSES_A,
        num_classes_task_b=NUM_CLASSES_B
    )

    mtl_model.eval()

    sample_sentences = [
        "How do I reset my password?", # Likely Technical Question, Neutral Sentiment
        "I really love using this software, it's great!", # Likely Feedback, Positive Sentiment
        "Tell me about your company history.", # Likely General Inquiry, Neutral Sentiment
        "This is taking too long to load and crashes often.", # Likely Feedback, Negative Sentiment
        "Just saying hello!", # Likely Chit-chat, Neutral Sentiment
    ]

    logging.info(f"\nPerforming forward pass on {len(sample_sentences)} sample sentences...")

    # Get the multi-task outputs (run in no_grad context for inference)
    with torch.no_grad():
        outputs = mtl_model(sample_sentences)

    logging.info("Forward pass completed.")

    # Showcase the obtained outputs (logits)
    print("\n--- Multi-Task Model Outputs (Logits) ---")

    print(f"\nOutput dictionary keys: {outputs.keys()}")

    logits_task_a = outputs['classification']
    logits_task_b = outputs['sentiment']

    print(f"\nTask A (Classification) Logits Shape: {logits_task_a.shape}")
    # Expected shape: (number_of_sentences, NUM_CLASSES_A) -> (5, 4)
    print(f"Sample Logits Task A (Sentence 1): {logits_task_a[0].tolist()}")

    print(f"\nTask B (Sentiment) Logits Shape: {logits_task_b.shape}")
    # Expected shape: (number_of_sentences, NUM_CLASSES_B) -> (5, 3)
    print(f"Sample Logits Task B (Sentence 1): {logits_task_b[0].tolist()}")

    # To get probabilities, you would apply softmax
    probs_task_a = torch.softmax(logits_task_a, dim=1)
    probs_task_b = torch.softmax(logits_task_b, dim=1)
    print(f"\nSample Probabilities Task A (Sentence 1): {probs_task_a[0].tolist()}")
    print(f"Sample Probabilities Task B (Sentence 1): {probs_task_b[0].tolist()}")