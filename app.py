import streamlit as st
import time
import math
import pandas as pd

from src.recommendation_engine import PhoneRecommendationEngine

# Initialize the PhoneRecommendationEngine object in session state if it doesn't exist
if 'recommendation_engine' not in st.session_state:
    st.session_state.recommendation_engine = PhoneRecommendationEngine()

st.title("Product Recommendation System")
st.caption('powered by IBM® watson:blue[x]™ embedding model')

# Add a collapsible accordion section to view the dataset
with st.expander("View Dataset"):
    dataset = st.session_state.recommendation_engine.get_dataset()
    if dataset is not None:
        st.dataframe(dataset)
    else:
        st.write("Unable to load dataset.")

input_text = st.text_input("Enter your requirement description here")

if st.button("Get Recommendations"):
    with st.spinner("invoking watsonx.ai embedding model please wait..."):
        recommendations = st.session_state.recommendation_engine.calculate_similarity(input_text=input_text)
        if recommendations and 'recommendations' in recommendations:
            recommendations = recommendations['recommendations']
            
            # Define number of columns in the grid
            num_columns = 3
            
            # Calculate number of rows needed
            num_rows = math.ceil(len(recommendations) / num_columns)
            
            for row in range(num_rows):
                cols = st.columns(num_columns)
                for col in range(num_columns):
                    idx = row * num_columns + col
                    if idx < len(recommendations):
                        with cols[col]:
                            recommendation = recommendations[idx]
                            with st.container():
                                st.subheader(recommendation['product_label'])
                                st.image(recommendation['imgURL'], width=200)
                                st.write(f"Price: ₹{recommendation['price']}")
                                st.write(f"Rating: {recommendation['ratings']} ⭐")
                                st.metric("Similarity", recommendation['similarity_score'])
                            

