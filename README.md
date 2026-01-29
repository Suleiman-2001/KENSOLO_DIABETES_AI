# KENSOLO AI Analytics Dashboard

This is a Streamlit-based AI analytics system that can:

- Detect potential issues in your dataset
- Auto-select targets for prediction
- Provide NLP summaries
- Generate predictions & recommendations
- Visualize graphs
- Provide downloadable JSON and PDF reports

## How to Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click 'New App' → Connect your GitHub repository
3. Select the 'main' branch and 'app.py' as the main file
4. Streamlit Cloud will install dependencies from 'requirements.txt'
5. Your AI dashboard will be live online!

## How to Run Locally

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py

All outputs will be saved in the 'outputs/' folder.