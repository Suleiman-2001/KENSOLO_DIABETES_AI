import os

# Root folder
root = os.getcwd()
print(f"Creating KENSOLO AI project structure in {root}...\n")

# --- Folders ---
folders = ["core", "engines", "utils", "outputs", "outputs/graphs"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ Created folder: {folder}")

# --- requirements.txt ---
req_content = """streamlit==1.25.0
pandas==2.1.1
numpy==1.26.4
matplotlib==3.8.0
reportlab==4.1.3
scikit-learn==1.3.2
"""
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write(req_content)
print("\n✅ requirements.txt created")

# --- README.md ---
readme_content = (
"# KENSOLO AI Analytics Dashboard\n\n"
"This is a Streamlit-based AI analytics system that can:\n\n"
"- Detect potential issues in your dataset\n"
"- Auto-select targets for prediction\n"
"- Provide NLP summaries\n"
"- Generate predictions & recommendations\n"
"- Visualize graphs\n"
"- Provide downloadable JSON and PDF reports\n\n"
"## How to Deploy on Streamlit Cloud\n\n"
"1. Go to https://share.streamlit.io/\n"
"2. Click 'New App' → Connect your GitHub repository\n"
"3. Select the 'main' branch and 'app.py' as the main file\n"
"4. Streamlit Cloud will install dependencies from 'requirements.txt'\n"
"5. Your AI dashboard will be live online!\n\n"
"## How to Run Locally\n\n"
"Install dependencies:\n"
"pip install -r requirements.txt\n\n"
"Run the app:\n"
"streamlit run app.py\n\n"
"All outputs will be saved in the 'outputs/' folder."
)

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)
print("\n✅ README.md created")

print("\n🎉 Project structure ready! You can now add your AI code and app.py.")
