import streamlit as st
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px
import openai
import random
import time
from typing import List, Dict


""" load wine data from wine_data 
"""
def load_data(file_path='data/wine_data.csv'):
    try:
        df = pd.read_csv(file_path)  # Read CSV file
        return df
    except FileNotFoundError:
        st.error(f"Data file not found: {file_path}")
        return pd.DataFrame()


def preprocess_data(df):
    """Preprocess the wine dataset."""
    # Convert quality to binary classification (good vs bad wine)
    # Quality <= 5: Low Quality, Quality > 5: High Quality
    df['quality_label'] = df['quality'].apply(lambda x: 'High Quality' if x > 5 else 'Low Quality')

    # Features and target
    feature_columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
                       'density', 'pH', 'sulphates', 'alcohol']

    return feature_columns, df


def create_few_shot_examples(df, n_examples_per_class=10):
    """Create balanced few-shot examples for the LLM prompt."""
    few_shot_examples = []

    # Get examples for each class
    for label in ['Low Quality', 'High Quality']:
        class_examples = df[df['quality_label'] == label].sample(
            n=min(n_examples_per_class, len(df[df['quality_label'] == label])),
            random_state=42
        )

        for _, row in class_examples.iterrows():
            example_features = {
                'fixed_acidity': row['fixed_acidity'],
                'volatile_acidity': row['volatile_acidity'],
                'citric_acid': row['citric_acid'],
                'residual_sugar': row['residual_sugar'],
                'chlorides': row['chlorides'],
                'free_sulfur_dioxide': row['free_sulfur_dioxide'],
                'total_sulfur_dioxide': row['total_sulfur_dioxide'],
                'density': row['density'],
                'pH': row['pH'],
                'sulphates': row['sulphates'],
                'alcohol': row['alcohol']
            }
            few_shot_examples.append({
                'features': example_features,
                'label': label
            })

    # Shuffle the examples
    random.shuffle(few_shot_examples)
    return few_shot_examples


def build_few_shot_prompt(few_shot_examples: List[Dict], target_features: Dict) -> str:
    """Build a few-shot prompt for the LLM."""

    prompt = """You are a wine quality classifier that categorizes wines as either "Low Quality" or "High Quality".

Based on the chemical properties of wine, classify it as:
- "Low Quality": Quality score ≤ 5
- "High Quality": Quality score > 5

Here are some examples:

"""

    # Add few-shot examples
    for example in few_shot_examples:
        features_str = ", ".join([f"{k}: {v}" for k, v in example['features'].items()])
        prompt += f'Wine Properties: {features_str}\nClassification: {example["label"]}\n\n'

    # Add the target features
    target_str = ", ".join([f"{k}: {v}" for k, v in target_features.items()])
    prompt += f'Wine Properties: {target_str}\nClassification:'

    return prompt


def predict_with_llm(client, few_shot_examples: List[Dict], features_dict: Dict,
                     model_name: str = "gpt-3.5-turbo") -> Dict:
    """Make prediction using LLM with few-shot prompting."""
    try:
        # Build the prompt
        prompt = build_few_shot_prompt(few_shot_examples, features_dict)

        # Make API call
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system",
                 "content": "You are a helpful wine quality classifier. Respond with exactly 'Low Quality' or 'High Quality' only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )

        prediction = response.choices[0].message.content.strip()

        # Clean up the prediction
        if "High Quality" in prediction:
            prediction = "High Quality"
        elif "Low Quality" in prediction:
            prediction = "Low Quality"
        else:
            # Default to Low Quality if unclear
            prediction = "Low Quality"

        result = {
            'prediction': prediction,
            'features': features_dict
        }

        return result

    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return {
            'prediction': "Low Quality",
            'features': features_dict
        }


def evaluate_llm_model(client, few_shot_examples: List[Dict], df_test,
                       model_name: str = "gpt-3.5-turbo"):
    """Evaluate the LLM model on test set."""
    # Sample a smaller subset for evaluation to save costs
    test_sample_size = min(30, len(df_test))
    df_sample = df_test.sample(n=test_sample_size, random_state=42)

    y_true = []
    y_pred = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (_, row) in enumerate(df_sample.iterrows()):
        features_dict = {
            'fixed_acidity': row['fixed_acidity'],
            'volatile_acidity': row['volatile_acidity'],
            'citric_acid': row['citric_acid'],
            'residual_sugar': row['residual_sugar'],
            'chlorides': row['chlorides'],
            'free_sulfur_dioxide': row['free_sulfur_dioxide'],
            'total_sulfur_dioxide': row['total_sulfur_dioxide'],
            'density': row['density'],
            'pH': row['pH'],
            'sulphates': row['sulphates'],
            'alcohol': row['alcohol']
        }

        result = predict_with_llm(client, few_shot_examples, features_dict, model_name)

        y_true.append(row['quality_label'])
        y_pred.append(result['prediction'])

        # Update progress
        progress_bar.progress((idx + 1) / test_sample_size)
        status_text.text(f"Evaluating: {idx + 1}/{test_sample_size}")

        # Small delay to avoid rate limits
        time.sleep(0.5)

    progress_bar.empty()
    status_text.empty()

    return y_true, y_pred


def main():
    st.set_page_config(
        page_title="Wine Quality Classifier",
        page_icon="🍷",
        layout="wide"
    )

    st.title("🍷 Wine Quality Classifier")
    st.markdown("*Powered by OpenAI GPT with Few-Shot Learning*")
    st.markdown("---")

    # API Configuration
    st.sidebar.header("🔧 API Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password",
                                    help="Enter your OpenAI API key")
    model_choice = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"], index=0)

    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        return

    # Initialize OpenAI client
    try:
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return

    # Load data
    with st.spinner("Loading data..."):
        df = load_data()

    if df.empty:
        st.error("No data available. Please check the data files.")
        return

    # Preprocess data
    feature_columns, df_processed = preprocess_data(df)

    # Sidebar with dataset info
    st.sidebar.header("📊 Dataset Information")
    st.sidebar.metric("Total Examples", len(df_processed))

    class_counts = df_processed['quality_label'].value_counts()
    for label, count in class_counts.items():
        st.sidebar.metric(f"{label}", count)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 Classification:\n- **High Quality**: Score > 5\n- **Low Quality**: Score ≤ 5")

    # Configuration
    st.sidebar.header("⚙️ Model Configuration")
    n_examples_per_class = st.sidebar.slider("Examples per class in prompt", min_value=3,
                                             max_value=15, value=5,
                                             help="Number of examples for each class to include in the few-shot prompt")

    # Display data overview
    with st.expander("📊 View Dataset Overview", expanded=False):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Sample Data")
            st.dataframe(df_processed.head(10), use_container_width=True)

        with col2:
            st.subheader("Quality Distribution")
            quality_dist = df_processed['quality'].value_counts().sort_index()
            fig = px.bar(
                x=quality_dist.index,
                y=quality_dist.values,
                labels={'x': 'Quality Score', 'y': 'Count'},
                title='Wine Quality Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Setup LLM
    st.header("🤖 LLM Setup")

    if st.button("🔄 Prepare Few-Shot Examples", type="primary"):
        with st.spinner("Preparing few-shot examples..."):
            # Create few-shot examples
            few_shot_examples = create_few_shot_examples(df_processed, n_examples_per_class)

            # Store in session state
            st.session_state.client = client
            st.session_state.few_shot_examples = few_shot_examples
            st.session_state.df_processed = df_processed
            st.session_state.model_name = model_choice
            st.session_state.feature_columns = feature_columns

            st.success(f"✅ Few-shot setup complete!")

            # Show few-shot examples info
            st.info(f"""
            **Few-Shot Examples**: {len(few_shot_examples)} total
            - {len([ex for ex in few_shot_examples if ex['label'] == 'Low Quality'])} Low Quality examples
            - {len([ex for ex in few_shot_examples if ex['label'] == 'High Quality'])} High Quality examples
            """)

            # Show sample few-shot examples
            with st.expander("🔍 View Sample Few-Shot Examples"):
                for example in few_shot_examples[:4]:  # Show first 4
                    features_str = ", ".join([f"{k}: {v}" for k, v in example['features'].items()])
                    if example['label'] == 'Low Quality':
                        st.error(f"**{example['label']}**: {features_str}")
                    else:
                        st.success(f"**{example['label']}**: {features_str}")

    st.markdown("---")

    # Show LLM evaluation if available
    if 'few_shot_examples' in st.session_state:
        with st.expander("🧪 Evaluate LLM Performance", expanded=False):
            st.warning(
                "⚠️ This will make API calls to evaluate performance. Estimated cost: ~$0.05-0.30")

            if st.button("🔬 Run LLM Evaluation"):
                with st.spinner("Evaluating LLM performance on test set..."):
                    try:
                        # Create test set
                        test_size = min(30, len(st.session_state.df_processed))
                        df_test = st.session_state.df_processed.sample(n=test_size, random_state=42)

                        y_true, y_pred = evaluate_llm_model(
                            st.session_state.client,
                            st.session_state.few_shot_examples,
                            df_test,
                            st.session_state.model_name
                        )

                        # Store evaluation results
                        st.session_state.y_true = y_true
                        st.session_state.y_pred = y_pred

                        # Calculate accuracy
                        accuracy = accuracy_score(y_true, y_pred)
                        st.success(f"✅ LLM Evaluation Complete! Test Accuracy: {accuracy:.3f}")

                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")

        # Show evaluation results if available
        if 'y_true' in st.session_state and 'y_pred' in st.session_state:
            st.subheader("📈 Evaluation Results")

            try:
                report = classification_report(
                    st.session_state.y_true,
                    st.session_state.y_pred,
                    output_dict=True
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Classification Metrics")
                    metrics_data = {
                        'Low Quality': {
                            'Precision': report['Low Quality']['precision'],
                            'Recall': report['Low Quality']['recall'],
                            'F1-Score': report['Low Quality']['f1-score']
                        },
                        'High Quality': {
                            'Precision': report['High Quality']['precision'],
                            'Recall': report['High Quality']['recall'],
                            'F1-Score': report['High Quality']['f1-score']
                        }
                    }
                    metrics_df = pd.DataFrame(metrics_data).T
                    st.dataframe(metrics_df.round(3), use_container_width=True)
                    st.metric("Overall Accuracy", f"{report['accuracy']:.3f}")

                with col2:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(st.session_state.y_true, st.session_state.y_pred,
                                          labels=['Low Quality', 'High Quality'])

                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        labels={'x': 'Predicted', 'y': 'Actual'},
                        x=['Low Quality', 'High Quality'],
                        y=['Low Quality', 'High Quality'],
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error generating evaluation metrics: {e}")

    st.markdown("---")

    # Prediction section
    st.header("🎯 Predict Wine Quality")

    if 'few_shot_examples' not in st.session_state:
        st.warning("⚠️ Please prepare few-shot examples first!")
    else:
        st.subheader("Enter Wine Properties")

        col1, col2 = st.columns(2)

        with col1:
            fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0,
                                            value=7.4, step=0.1,
                                            help="Tartaric acid - most acids involved with wine")
            volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0,
                                               value=0.7, step=0.01,
                                               help="Amount of acetic acid - too high can lead to unpleasant vinegar taste")
            citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=2.0, value=0.0,
                                          step=0.01,
                                          help="Adds 'freshness' and flavor to wines")
            residual_sugar = st.number_input("Residual Sugar (g/L)", min_value=0.0, max_value=20.0,
                                             value=1.9, step=0.1,
                                             help="Sugar remaining after fermentation stops")
            chlorides = st.number_input("Chlorides (g/L)", min_value=0.0, max_value=1.0,
                                        value=0.076, step=0.001,
                                        help="Amount of salt in the wine")
            free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide (mg/L)", min_value=0.0,
                                                  max_value=100.0, value=11.0, step=1.0,
                                                  help="Prevents microbial growth and oxidation")

        with col2:
            total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide (mg/L)", min_value=0.0,
                                                   max_value=300.0, value=34.0, step=1.0,
                                                   help="Free + bound forms of SO2")
            density = st.number_input("Density (g/cm³)", min_value=0.9, max_value=1.1, value=0.9978,
                                      step=0.0001, format="%.4f",
                                      help="Density of wine relative to water")
            pH = st.number_input("pH", min_value=2.0, max_value=4.5, value=3.51, step=0.01,
                                 help="Describes acidity/basicity on scale 0-14")
            sulphates = st.number_input("Sulphates (g/L)", min_value=0.0, max_value=2.0, value=0.56,
                                        step=0.01,
                                        help="Wine additive contributing to SO2 levels")
            alcohol = st.number_input("Alcohol (%)", min_value=8.0, max_value=15.0, value=9.4,
                                      step=0.1,
                                      help="Percent alcohol content")

        st.markdown("---")

        if st.button("🔍 Predict Quality", type="primary", use_container_width=True):
            features_dict = {
                'fixed_acidity': fixed_acidity,
                'volatile_acidity': volatile_acidity,
                'citric_acid': citric_acid,
                'residual_sugar': residual_sugar,
                'chlorides': chlorides,
                'free_sulfur_dioxide': free_sulfur_dioxide,
                'total_sulfur_dioxide': total_sulfur_dioxide,
                'density': density,
                'pH': pH,
                'sulphates': sulphates,
                'alcohol': alcohol
            }

            with st.spinner("Classifying wine with LLM..."):
                result = predict_with_llm(
                    st.session_state.client,
                    st.session_state.few_shot_examples,
                    features_dict,
                    st.session_state.model_name
                )

            st.markdown("---")
            st.markdown("## 🎊 Prediction Result")

            # Display prediction
            if result['prediction'] == 'High Quality':
                st.success("### 🍷 High Quality Wine")
                st.markdown("**Quality Score: > 5**")
                st.balloons()
            else:
                st.error("### 📉 Low Quality Wine")
                st.markdown("**Quality Score: ≤ 5**")

            # Show input summary
            with st.expander("📋 View Input Summary"):
                input_df = pd.DataFrame([features_dict])
                st.dataframe(input_df.T, use_container_width=True)


if __name__ == "__main__":
    main()
