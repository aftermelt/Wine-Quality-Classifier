# Red Flag vs Green Flag LLM Classifier

A Streamlit web application for many-shot classification of text as "Red Flag" or "Green Flag" using Large Language Models (LLMs) with few-shot prompting, based on consensus-annotated data from two groups.

## Features

- **LLM-Powered Classification**: Uses OpenAI GPT models (3.5-turbo, GPT-4) with few-shot prompting
- **Few-Shot Learning**: Automatically creates balanced few-shot examples from training data
- **Configurable Prompting**: Adjustable number of examples per class in the prompt
- **Multiple Model Support**: Choose between different OpenAI models
- **Interactive Web Interface**: User-friendly Streamlit app for real-time classification
- **Batch Processing**: Support for multiple upload methods with cost estimation:
  - Text files (one sentence per line)
  - CSV files (with 'sentence' column) 
  - Manual text input
- **LLM Evaluation**: Performance testing on sample test set with API cost tracking
- **Downloadable Results**: Export batch classification results as CSV

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get OpenAI API Key:**
   - Sign up at https://platform.openai.com
   - Create an API key in your dashboard
   - Add credits to your account

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## Data

The app uses consensus data from two annotation groups:
- `data/Consensus items : Group 1 - Red Flag vs Green Flag.json`
- `data/Consensus items: Group 2 - Red Flag vs Green Flag.json`

Each file contains JSON arrays with objects having:
- `item_id`: Unique identifier
- `sentence`: Text content
- `gold_label`: Either "Red Flag" or "Green Flag"

## Usage

1. **API Setup**: Enter your OpenAI API key in the sidebar and select a model
2. **Prepare Few-Shot Examples**: Click "Prepare Few-Shot Examples" to create balanced training examples
3. **Single Classification**: Enter text and click "Classify Text" for individual predictions
4. **Batch Classification**: Upload files or enter multiple sentences (with cost estimates)
5. **LLM Evaluation**: Test performance on a sample of the test set

## Model Details

- **Algorithm**: Few-shot prompting with OpenAI GPT models
- **Prompt Engineering**: Structured prompts with clear task definitions and balanced examples
- **Few-Shot Examples**: Configurable number of examples per class (3-20)
- **Temperature**: Low temperature (0.1) for consistent predictions
- **Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix on test samples

## Cost Considerations

- **Single Classification**: ~$0.001 per text
- **Batch Processing**: Cost scales linearly with number of texts
- **Model Evaluation**: ~$0.05-0.20 for 50-item test sample
- **Model Choice**: GPT-3.5-turbo is most cost-effective, GPT-4 is more accurate but expensive

## Example Usage

### Single Text Classification
```
Input: "The rain had softened to a drizzle."
Output: ✅ Green Flag (Confidence: 0.85)
```

### Few-Shot Prompt Structure
```
You are a text classifier that categorizes sentences as either "Red Flag" or "Green Flag".

Red Flag: Concerning, negative, problematic, or potentially harmful content...
Green Flag: Neutral, positive, educational, or harmless content...

Examples:
Text: "It had been left on Elara's doorstep in a small wooden box with no note."
Classification: Red Flag

Text: "Photosynthesis is crucial for life on Earth..."
Classification: Green Flag

Text: "[Your input text here]"
Classification:
```

### Batch Classification
Upload files with cost estimates shown before processing. Results include confidence scores and are downloadable as CSV.