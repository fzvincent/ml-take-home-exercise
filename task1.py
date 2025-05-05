# requirements.txt:
# torch
# transformers

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SentenceTransformerModel(torch.nn.Module):
    def __init__(self, model_name = 'sentence-transformers/all-MiniLM-L6-v2', pooling_strategy = 'mean'):
    
        super(SentenceTransformerModel, self).__init__()
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy

        logging.info(f"Loading pre-trained model: {self.model_name}")
        # Load the pre-trained transformer model from Hugging Face
        self.transformer_model = AutoModel.from_pretrained(model_name)
        # Load the corresponding tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info(f"Model and tokenizer loaded successfully.")

        assert pooling_strategy  in ['mean', 'cls', 'max']

    def _mean_pooling(self, model_output, attention_mask):
        """
        Performs mean pooling on the last hidden state.
        It takes the average of all token embeddings, ignoring padding tokens.
        """
        # Extract the last hidden state (batch_size, sequence_length, hidden_size)
        token_embeddings = model_output.last_hidden_state
        # Expand attention mask from (batch_size, sequence_length) to
        # (batch_size, sequence_length, hidden_size) to match token_embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Zero out embeddings corresponding to padding tokens
        masked_embeddings = token_embeddings * input_mask_expanded
        # Sum embeddings across the sequence length dimension
        sum_embeddings = torch.sum(masked_embeddings, 1)
        # Sum the attention mask elements to get the number of actual tokens (not padding)
        # Clamp minimum to 1e-9 to avoid division by zero
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # Divide the sum of embeddings by the number of actual tokens
        mean_pooled_embeddings = sum_embeddings / sum_mask
        return mean_pooled_embeddings

    def _cls_pooling(self, model_output):
        # The CLS token embedding is typically the first token's embedding
        return model_output.last_hidden_state[:, 0] # (batch_size, hidden_size)

    def _max_pooling(self, model_output, attention_mask):
        """
        Performs max pooling on the last hidden state.
        Takes the element-wise maximum across the sequence dimension, ignoring padding.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Set padding tokens to a very small value so they are not chosen by max pooling
        token_embeddings[input_mask_expanded == 0] = -1e9
        # Compute the max over the sequence length dimension
        max_pooled_embeddings = torch.max(token_embeddings, 1)[0] # [0] to get the values, not indices
        return max_pooled_embeddings

    def forward(self, sentences: list[str]):
     
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)

        with torch.no_grad():
            model_output = self.transformer_model(**encoded_input)

        if self.pooling_strategy == 'mean':
            sentence_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
        elif self.pooling_strategy == 'cls':
             if 'bert' not in self.model_name.lower() and 'xlm' not in self.model_name.lower():
                 logging.warning(f"CLS pooling chosen for model '{self.model_name}', which might not be optimal "
                                 "unless fine-tuned specifically for CLS-based sentence tasks.")
             sentence_embedding = self._cls_pooling(model_output)
        elif self.pooling_strategy == 'max':
            sentence_embedding = self._max_pooling(model_output, encoded_input['attention_mask'])
        else:
            # This case should ideally be caught in __init__, but included for completeness
             raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")

        normalized_embedding = F.normalize(sentence_embedding, p=2, dim=1)

        return normalized_embedding # Or return sentence_embedding if normalization is not desired

# --- Testing the Implementation ---
if __name__ == "__main__":
    model = SentenceTransformerModel(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        pooling_strategy='cls'
    )
    # Set the model to evaluation mode (disables dropout, etc.)
    model.eval()

    # Sample sentences to test
    sample_sentences = [
        "This is the first sample sentence.",
        "Here is another sentence for testing.",
        "Sentence embeddings are useful for semantic search.",
        "A short one."
    ]

    logging.info(f"\nEncoding {len(sample_sentences)} sample sentences...")

    # Get the embeddings
    embeddings = model(sample_sentences)

    logging.info("Embeddings obtained successfully.")

    # Showcase the obtained embeddings
    print("\n--- Obtained Embeddings ---")
    print(f"Shape of embeddings tensor: {embeddings.shape}")


    print("\nFirst few dimensions of each sentence embedding:")
    for i, sentence in enumerate(sample_sentences):
        print(f"Sentence {i+1}: '{sentence}'")
        # Print first 5 dimensions and last 5 dimensions
        embedding_str = (f"{embeddings[i, :5].tolist()} ... {embeddings[i, -5:].tolist()}")
        print(f"  Embedding (vector dims [:5]...[-5:]): {embedding_str}\n")

