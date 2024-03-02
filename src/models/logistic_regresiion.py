import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Load the pre-trained transformer model and tokenizer
model_name = "allmpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer_model = AutoModel.from_pretrained(model_name)

# Define the popularity prediction model
class PopularityPredictionModel(nn.Module):
    def __init__(self, transformer_model):
        super(PopularityPredictionModel, self).__init__()
        self.transformer = transformer_model
        self.fc_layer = nn.Linear(self.transformer.config.hidden_size, self.transformer.config.hidden_size)
        self.activation = nn.Tanh()
        self.regressor = nn.Linear(self.transformer.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extract the CLS token embedding
        fc_output = self.activation(self.fc_layer(cls_embedding))  # Apply fully-connected layer with activation
        popularity_score = self.regressor(fc_output)  # Predict the popularity score
        return popularity_score

# Initialize the popularity prediction model
popularity_model = PopularityPredictionModel(transformer_model)

# Example usage:
input_headline = "Your news headline goes here"
input_encoding = tokenizer(input_headline, return_tensors="pt", padding="max_length", truncation=True)
popularity_score = popularity_model(input_encoding["input_ids"], input_encoding["attention_mask"])
