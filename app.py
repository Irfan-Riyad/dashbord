import streamlit as st
import pandas as pd

# Try importing mlxtend, and show helpful message if it's missing
try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
except ModuleNotFoundError:
    st.error("""
    ❌ The `mlxtend` package is missing.  
    Please make sure your **requirements.txt** file includes:
    ```
    streamlit
    pandas
    scikit-learn
    mlxtend
    ```
    Then redeploy your app.
    """)
    st.stop()

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Frequent Pattern Mining Dashboard", layout="wide")

st.title("🧩 Frequent Pattern Mining Dashboard")
st.write("Mine frequent itemsets using **Apriori** or **FP-Growth** from the `mlxtend` library.")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")

data_option = st.sidebar.radio(
    "Select Dataset Option:",
    ["Upload CSV", "Use Sample Dataset"]
)

uploaded_file = None
if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

algorithm = st.sidebar.selectbox("Select Algorithm", ["Apriori", "FP-Growth"])
min_support = st.sidebar.slider("Minimum Support (Relative %)", 0.01, 1.0, 0.2, 0.01)

fix_k = st.sidebar.checkbox("Fix pattern length (k)?", value=False)
if fix_k:
    k_value = st.sidebar.number_input("Pattern Length (k)", min_value=1, step=1, value=2)
else:
    k_value = None

# --- Load Dataset ---
if data_option == "Use Sample Dataset":
    st.info("Using sample market basket dataset.")
    df = pd.DataFrame({
        'milk': [1, 1, 0, 1, 0],
        'bread': [1, 0, 1, 1, 1],
        'butter': [0, 1, 1, 1, 1],
        'cheese': [0, 0, 1, 0, 1]
    })
else:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = None

# --- Analysis ---
if df is not None:
    try:
        st.write("### 📄 Preview of Data")
        st.dataframe(df.head())

        # One-hot encode if needed
        if df.dtypes.isin(['object']).any():
            st.info("🧮 Detected non-numeric data — performing one-hot encoding.")
            df = pd.get_dummies(df)

        st.write("### ✅ Encoded Data Preview")
        st.dataframe(df.head())

        # Run algorithm
        if algorithm == "Apriori":
            st.write(f"🔍 Running **Apriori** with min_support = {min_support}")
            freq_items = apriori(df, min_support=min_support, use_colnames=True)
        else:
            st.write(f"⚡ Running **FP-Growth** with min_support = {min_support}")
            freq_items = fpgrowth(df, min_support=min_support, use_colnames=True)

        # Filter by pattern length
        if fix_k and not freq_items.empty:
            freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))
            freq_items = freq_items[freq_items['length'] == k_value]

        if not freq_items.empty:
            st.write("### 📊 Frequent Itemsets")
            st.dataframe(freq_items)

            st.write("### 🔗 Association Rules")
            rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
            st.dataframe(rules)

            st.write("### 📈 Top Frequent Itemsets by Support")
            top_items = freq_items.sort_values("support", ascending=False).head(10)
            top_items.index = top_items['itemsets'].apply(lambda x: ', '.join(list(x)))
            st.bar_chart(top_items['support'])
        else:
            st.warning("No frequent itemsets found for the current parameters.")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("⬆️ Please upload a CSV file or use the sample dataset to begin.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + mlxtend")
