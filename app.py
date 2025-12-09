import streamlit as st
import pandas as pd
import openai


# load data
def load_data():
    try:
        return pd.read_csv('wine_data.csv')
    except:
        return pd.read_csv('data/wine_data.csv')

# make the prompt for gpt
def create_prompt(examples, user_wine):
    prompt = "Classify wine as 'High Quality' (score > 5) or 'Low Quality' (score ≤ 5).\n\nExamples:\n"

    for _, row in examples.iterrows():
        wine_info = f"Acidity:{row['fixed_acidity']}, Alcohol:{row['alcohol']}, pH:{row['pH']}"
        label = 'High Quality' if row['quality'] > 5 else 'Low Quality'
        prompt += f"{wine_info} → {label}\n"

    user_info = f"Acidity:{user_wine['fixed_acidity']}, Alcohol:{user_wine['alcohol']}, pH:{user_wine['pH']}"
    prompt += f"\n{user_info} →"
    return prompt

# get the prediction
def predict(client, df, user_wine):
    # random examples (10 wines)
    examples = df.sample(10)

    # create prompt
    prompt = create_prompt(examples, user_wine)

    # send prompt to chatgpt
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )

    result = response.choices[0].message.content.strip()
    return "High Quality" if "High" in result else "Low Quality"


# name of app (title)
st.title("🍷 Wine Quality Classifier")

# api prompt
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

# open the api key (use it)
client = openai.OpenAI(api_key=api_key)
df = load_data()

if df.empty:
    st.error("Sorry, wine_data.csv not found.")
    st.stop()

st.success(f"Sucessfully loaded {len(df)} wines.")

# Input Form
st.subheader("Enter Wine Properties")

col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, 7.4, 0.1)
    volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.6, 0.7, 0.01)
    citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0, 0.01)
    residual_sugar = st.number_input("Residual Sugar", 0.0, 16.0, 1.9, 0.1)

with col2:
    chlorides = st.number_input("Chlorides", 0.01, 0.62, 0.076, 0.001)
    free_sulfur_dioxide = st.number_input("Free SO2", 1.0, 72.0, 11.0, 1.0)
    total_sulfur_dioxide = st.number_input("Total SO2", 6.0, 290.0, 34.0, 1.0)
    density = st.number_input("Density", 0.99, 1.01, 0.9978, 0.0001, format="%.4f")

with col3:
    pH = st.number_input("pH", 2.7, 4.0, 3.51, 0.01)
    sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.56, 0.01)
    alcohol = st.number_input("Alcohol %", 8.0, 15.0, 9.4, 0.1)

# get the prediction
if st.button("🔍 Predict Quality", type="primary"):
    user_wine = {
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

    with st.spinner("Analyzing..."):
        prediction = predict(client, df, user_wine)

    st.markdown("---")
    if prediction == "High Quality":
        st.success("### High Quality Wine.")
        st.balloons()
    else:
        st.error("### Low Quality Wine.")