# Wine Quality Classifier

A Streamlit web application for many-shot classification of text as "Red Flag" or "Green Flag" using Large Language Models (LLMs) with few-shot prompting, based on a Kaggle Dataset


## Data

The app uses consensus data from Kaggle:
https://www.kaggle.com/datasets/taweilo/wine-quality-dataset-balanced-classification

Each file contains JSON arrays with objects having:

fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,quality

## Cost Considerations

- **Single Classification**: ~$0.001 per text
- **Batch Processing**: Cost scales linearly with number of texts
- **Model Evaluation**: ~$0.05-0.20 for 50-item test sample
- **Model Choice**: GPT-3.5-turbo is most cost-effective
