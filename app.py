import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules

# Streamlit App
st.set_page_config(page_title="Frequent Pattern Mining Dashboard", layout="wide")

st.title("🧩 Frequent Pattern Mining Dashboard")
st.write("Mine frequent itemsets using **Apriori** or **FP-Growth** algorithm.")

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader("Upload Transaction Dataset (CSV)", type=["csv"])

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["Apriori", "FP-Growth"]
)

min_support = st.sidebar.slider(
    "Minimum Support (Relative %)",
    min_value=0.01,
    max_value=1.0,
    value=0.2,
    step=0.01
)

fix_k = st.sidebar.checkbox("Fix pattern length (k)?", value=False)
if fix_k:
    k_value = st.sidebar.number_input("Pattern Length (k)", min_value=1, step=1, value=2)
else:
    k_value = None

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### 📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Check if dataset is one-hot encoded or needs encoding
    if df.dtypes.isin(['object']).any():
        st.info("🧮 Detected non-numeric data — performing one-hot encoding.")
        df = pd.get_dummies(df)

    st.write("### ✅ Encoded Data")
    st.dataframe(df.head())

    # Algorithm selection
    if algorithm == "Apriori":
        st.write(f"🔍 Running **Apriori** with min_support = {min_support}")
        freq_items = apriori(df, min_support=min_support, use_colnames=True)
    else:
        st.write(f"⚡ Running **FP-Growth** with min_support = {min_support}")
        freq_items = fpgrowth(df, min_support=min_support, use_colnames=True)

    # Filter by pattern length if specified
    if fix_k and not freq_items.empty:
        freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))
        freq_items = freq_items[freq_items['length'] == k_value]

    # Display Results
    if not freq_items.empty:
        st.write("### 📊 Frequent Itemsets")
        st.dataframe(freq_items)

        # Show association rules
        st.write("### 🔗 Association Rules")
        rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
        st.dataframe(rules)
    else:
        st.warning("No frequent itemsets found with the current settings.")

else:
    st.info("⬆️ Please upload a CSV file to begin.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + mlxtend")
