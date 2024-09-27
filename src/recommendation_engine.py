import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from config.config import Config
config = Config()

# IBM Watsonx credentials

embeddings = WatsonxEmbeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
    url=config.WATSONX_URL,
    apikey=config.WATSONX_API_KEY,
    project_id=config.WATSONX_PROJECT_ID
)

from sklearn.model_selection import train_test_split
class PhoneRecommendationEngine:
    def __init__(self):
        # Configuration parameters
        self.config = {
            "threshold_limit": config.THRESHOLD_LIMIT,
            "recommendations_count": config.RECOMMENDATIONS_COUNT
        }

        # Path for data
        self.data_file = config.FILE_PATH

        # Load and prepare data
        self.data = self.load_data()
        self.existing_embeddings = self.generate_embeddings()

    def load_data(self):
        """Load mobile data and prepare it for recommendation."""
        df = pd.read_csv(self.data_file)
        # Perform a train-test split on 'product_label'
        unique_labels = df['product_label'].unique()
        train_labels, test_labels = train_test_split(unique_labels, test_size=0.2, random_state=42)

        # Create train and test sets based on the split labels
        train_df = df[df['product_label'].isin(train_labels)]
        test_df = df[df['product_label'].isin(test_labels)]

        # Extract relevant columns
        texts = train_df['Description'].astype(str)
        product_labels = train_df['product_label'].astype(str)
        ratings = train_df['ratings']
        prices = train_df['price']
        img_urls = train_df['imgURL']
        
        return pd.DataFrame({
            'text': texts,
            'product_label': product_labels,
            'ratings': ratings,
            'price': prices,
            'imgURL': img_urls
        })

    def generate_embeddings(self):
        """Generate embeddings for mobile descriptions."""
        texts = self.data['text'].tolist()
        return embeddings.embed_documents(texts)

    def calculate_similarity(self, input_text, similarity_threshold=None):
        """Calculate similarity between a new query and existing mobile data."""
        if similarity_threshold is None:
            similarity_threshold = config.THRESHOLD_LIMIT
        
        new_embedding = np.array(embeddings.embed_query(input_text))
        existing_embeddings_np = np.array(self.existing_embeddings)
        similarities = cosine_similarity([new_embedding], existing_embeddings_np)[0]

        similarity_percentage = np.round(similarities * 100, 2)
        product_labels = self.data['product_label'].tolist()
        additional_data = self.data[['ratings', 'price', 'imgURL']].values.tolist()

        # Get top recommendations based on similarity
        unique_labels = set()
        unique_top_indices = []
        for idx in np.argsort(similarities)[::-1]:
            if similarities[idx] >= similarity_threshold and product_labels[idx] not in unique_labels:
                unique_labels.add(product_labels[idx])
                unique_top_indices.append(idx)
            if len(unique_top_indices) == config.RECOMMENDATIONS_COUNT:
                break

        # Handle case where no recommendations are found
        if not unique_top_indices:
            return {"message": "No similar data found", "recommendations": []}

        # Construct the result data
        top_data = [(*additional_data[idx], similarity_percentage[idx]) for idx in unique_top_indices]
        return {
            "recommendations": [
                {
                    "product_label": product_labels[idx],
                    "ratings": data[0],
                    "price": data[1],
                    "imgURL": data[2],
                    "similarity_score": f"{data[3]}%"
                } for idx, data in zip(unique_top_indices, top_data)
            ]
        }
    def get_dataset(self):
        df = pd.read_csv(self.data_file)
        return df
