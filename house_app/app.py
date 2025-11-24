 

"""import streamlit as st
import pandas as pd
import plotly.express as px
import os
import pickle

# ✅ Set page config
st.set_page_config(page_title="🏠 House Prices Explorer", layout="wide")

# ✅ Title
st.title("🏠 House Prices Explorer")

# ✅ Current working directory
st.write("**Current Working Directory:**", os.getcwd())

# ✅ Load dataset safely
try:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../train.csv"))
    st.write(df.columns)

    
    # 🔄 Rename columns to your preferred names
    df.rename(columns={
        'GrLivArea': 'area',
        'Neighborhood': 'city',
        'SalePrice': 'price',
         'HouseStyle': 'house_style'

    }, inplace=True)


    house_style_list = df["house_style"].unique()

    
    st.success("✅ Dataset loaded successfully!")
    st.write("### Sample Data", df.head())

    # 💰 Price Distribution
    st.subheader("💰 Price Distribution")
    fig = px.histogram(df, x="price", nbins=50, title="Distribution of House Prices")
    st.plotly_chart(fig)

except Exception as e:
    st.error(f"❌ Error loading data: {e}") 


# 🏠 Prediction Section
st.title("🏠 House Price Prediction App")
st.markdown("#### Made with ❤️ using Streamlit and Machine Learning")

# 🧩 Scatter plot
fig = px.scatter(df, x="area", y="price", color="city", title="Price vs Area by City")
st.plotly_chart(fig, use_container_width=True)

# 🧱 Sidebar for input
st.sidebar.header("Enter House Details")

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Inputs
area = st.sidebar.number_input("Area (sq ft)", min_value=500, max_value=10000, step=100)
bedrooms = st.sidebar.slider("Bedrooms", 1, 5, 3)
city = st.sidebar.selectbox("City", df['city'].unique())

# 🧮 Predict
if st.sidebar.button("Predict Price"):
    result = model.predict([[area, bedrooms]])
    st.success(f"Estimated Price: ₹{round(result[0], 2)} Lakhs")





overall_qual = st.number_input("Overall Quality", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sqft)", 300, 6000, 1500)
year_built = st.number_input("Year Built", 1870, 2025, 1990)


if st.button("Predict Price"):
    pred = model.predict([[overall_qual, gr_liv_area, year_built]])
    st.success(f"Predicted Price: ${pred[0]:,.2f}")


#fig = px.imshow(df.corr(), text_auto=True)
#st.plotly_chart(fig)


numeric_df = df.select_dtypes(include=['int64', 'float64'])
fig = px.imshow(numeric_df.corr(), text_auto=True)
st.plotly_chart(fig)


fig = px.scatter(df, x="area", y="price")
st.plotly_chart(fig)



fig = px.box(df, x="city", y="price")
st.plotly_chart(fig)




import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# -----------------------------
# Load Model & Data
# -----------------------------
df = pd.read_csv("cleaned_data.csv")  # apna data file
model = pickle.load(open("model.pkl", "rb"))

st.title("🏡 House Price Prediction App")

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔍 Filters")

# 1. Neighborhood Filter
neighborhood_list = df["Neighborhood"].unique()
selected_neighborhood = st.sidebar.multiselect(
    "Select Neighborhood",
    options=neighborhood_list,
    default=neighborhood_list
)

# 2. House Style Filter
house_style_list = df["HouseStyle"].unique()
selected_house_style = st.sidebar.multiselect(
    "Select House Style",
    options=house_style_list,
    default=house_style_list
)

# 3. Price Range Slider
min_price = int(df["SalePrice"].min())
max_price = int(df["SalePrice"].max())

price_range = st.sidebar.slider(
    "Select Price Range",
    min_price, max_price, (min_price, max_price)
)

# 4. Year Built Range Filter
min_year = int(df["YearBuilt"].min())
max_year = int(df["YearBuilt"].max())

year_range = st.sidebar.slider(
    "Year Built Range",
    min_year, max_year, (min_year, max_year)
)

# -----------------------------
# Filter Data Based on Sidebar
# -----------------------------
filtered_df = df[
    (df["Neighborhood"].isin(selected_neighborhood)) &
    (df["HouseStyle"].isin(selected_house_style)) &
    (df["SalePrice"].between(price_range[0], price_range[1])) &
    (df["YearBuilt"].between(year_range[0], year_range[1]))
]

# -----------------------------
# Show Filtered Table
# -----------------------------
st.subheader("📋 Filtered Dataset")
st.dataframe(filtered_df)

# -----------------------------
# Chart (Auto-Refresh)
# -----------------------------
st.subheader("📊 Price Distribution (Filtered)")
fig = px.histogram(filtered_df, x="SalePrice", nbins=40)
st.plotly_chart(fig)




model = pickle.load(open('model.pkl', 'rb'))


df = pd.read_csv("cleaned_data.csv")


st.write(df.columns)"""




import streamlit as st
import pandas as pd
import plotly.express as px
import os
import pickle
from pathlib import Path

# NOTE: user uploaded file available at /mnt/data/Screenshot 2025-11-23 194819.png
# If you need to reference uploaded files elsewhere, that path is: 
# UPLOADED_FILE_PATH = "/mnt/data/Screenshot 2025-11-23 194819.png"

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="🏠 House Prices Explorer", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_data(csv_path: str):
    """Load CSV safely and return dataframe."""
    return pd.read_csv(csv_path)

@st.cache_data
def load_model(pkl_path: str):
    """Load a pickle model if available, otherwise return None."""
    try:
        with open(pkl_path, "rb") as f:
            m = pickle.load(f)
        return m
    except Exception:
        return None

# -----------------------------
# Data load (single source)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR.joinpath("..", "train.csv")  # keeps same relative path as original
CLEANED_CSV = BASE_DIR.joinpath("cleaned_data.csv")
MODEL_PKL = BASE_DIR.joinpath("model.pkl")

# Prefer train.csv as requested by user
csv_to_use = TRAIN_CSV

try:
    df = load_data(str(csv_to_use))
except FileNotFoundError:
    st.error(f"Could not find {csv_to_use}. Please put your train.csv in the project folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Show columns (for debugging) — comment out in production if you want
st.write("**Loaded columns:**", list(df.columns))

# -----------------------------
# Standardize column names once
# -----------------------------
# We'll keep both original and normalized column names available to avoid KeyErrors.
rename_map = {
    'GrLivArea': 'area',
    'Neighborhood': 'city',
    'SalePrice': 'price',
    'HouseStyle': 'house_style'
}
# Only rename if those original columns exist
existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
if existing_renames:
    df = df.rename(columns=existing_renames)

# For safety, allow using either original or renamed names later by creating aliases
# e.g., ensure the normalized names exist when possible
if 'area' not in df.columns and 'GrLivArea' in df.columns:
    df['area'] = df['GrLivArea']
if 'city' not in df.columns and 'Neighborhood' in df.columns:
    df['city'] = df['Neighborhood']
if 'price' not in df.columns and 'SalePrice' in df.columns:
    df['price'] = df['SalePrice']
if 'house_style' not in df.columns and 'HouseStyle' in df.columns:
    df['house_style'] = df['HouseStyle']

# -----------------------------
# Page header
# -----------------------------
st.title("🏠 House Prices Explorer")
st.markdown("#### Made with ❤️ using Streamlit and Machine Learning")

st.write("**Current Working Directory:**", os.getcwd())

st.success("✅ Dataset loaded successfully!")
st.write("### Sample Data", df.head())

# -----------------------------
# Sidebar: Filters & Inputs
# -----------------------------
st.sidebar.header("🔍 Filters & Prediction Inputs")

# Filters (use fallbacks if columns missing)
city_options = df['city'].dropna().unique() if 'city' in df.columns else df['Neighborhood'].dropna().unique()
house_style_options = df['house_style'].dropna().unique() if 'house_style' in df.columns else df['HouseStyle'].dropna().unique()

selected_cities = st.sidebar.multiselect("Select City/Neighborhood", options=sorted(city_options), default=list(sorted(city_options)))
selected_house_styles = st.sidebar.multiselect("Select House Style", options=sorted(house_style_options), default=list(sorted(house_style_options)))

# Price range
if 'price' in df.columns:
    min_price = int(df['price'].min())
    max_price = int(df['price'].max())
else:
    min_price = int(df['SalePrice'].min()) if 'SalePrice' in df.columns else 0
    max_price = int(df['SalePrice'].max()) if 'SalePrice' in df.columns else 1000000

price_range = st.sidebar.slider("Select Price Range", min_price, max_price, (min_price, max_price))

# Year built range
min_year = int(df['YearBuilt'].min()) if 'YearBuilt' in df.columns else 1900
max_year = int(df['YearBuilt'].max()) if 'YearBuilt' in df.columns else 2025
year_range = st.sidebar.slider("Year Built Range", min_year, max_year, (min_year, max_year))

# Prediction inputs (simple example -- adapt to your model's expected features)
st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Inputs")
area_input = st.sidebar.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1500, step=50)
bedrooms_input = st.sidebar.slider("Bedrooms", 0, 10, 3)
overall_qual_input = st.sidebar.slider("Overall Quality", 1, 10, 5)
gr_liv_area_input = st.sidebar.number_input("Above Ground Living Area (sqft)", min_value=100, max_value=6000, value=1500, step=50)
year_built_input = st.sidebar.number_input("Year Built", min_value=1800, max_value=2100, value=2000)

# -----------------------------
# Apply Filters
# -----------------------------
filtered_df = df.copy()
# city
if 'city' in df.columns:
    filtered_df = filtered_df[filtered_df['city'].isin(selected_cities)]
else:
    filtered_df = filtered_df[filtered_df['Neighborhood'].isin(selected_cities)]
# house_style
if 'house_style' in df.columns:
    filtered_df = filtered_df[filtered_df['house_style'].isin(selected_house_styles)]
else:
    filtered_df = filtered_df[filtered_df['HouseStyle'].isin(selected_house_styles)]
# price
if 'price' in df.columns:
    filtered_df = filtered_df[filtered_df['price'].between(price_range[0], price_range[1])]
else:
    filtered_df = filtered_df[filtered_df['SalePrice'].between(price_range[0], price_range[1])]
# year
filtered_df = filtered_df[filtered_df['YearBuilt'].between(year_range[0], year_range[1])]

# -----------------------------
# Visualizations
# -----------------------------
st.subheader("📊 Price Distribution (Filtered)")
if 'price' in filtered_df.columns:
    fig = px.histogram(filtered_df, x='price', nbins=40, title="Price Distribution")
else:
    fig = px.histogram(filtered_df, x='SalePrice', nbins=40, title="Price Distribution")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 Price vs Area")
if 'area' in filtered_df.columns and 'price' in filtered_df.columns:
    fig2 = px.scatter(filtered_df, x='area', y='price', color='city' if 'city' in filtered_df.columns else None, title="Price vs Area")
else:
    # fallback to available numeric columns
    xcol = 'GrLivArea' if 'GrLivArea' in filtered_df.columns else (filtered_df.select_dtypes(include=['int64','float64']).columns[0] if not filtered_df.empty else None)
    ycol = 'SalePrice' if 'SalePrice' in filtered_df.columns else (filtered_df.select_dtypes(include=['int64','float64']).columns[1] if filtered_df.shape[1] > 1 else None)
    if xcol and ycol:
        fig2 = px.scatter(filtered_df, x=xcol, y=ycol, title="Price vs Area (fallback)")
    else:
        fig2 = None

if fig2 is not None:
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("📦 Correlation Matrix (Numeric)")
numeric_df = filtered_df.select_dtypes(include=['int64', 'float64'])
if not numeric_df.empty:
    fig3 = px.imshow(numeric_df.corr(), text_auto=True)
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.write("No numeric columns to display correlation.")

st.subheader("📋 Filtered Dataset")
st.dataframe(filtered_df)

# -----------------------------
# Model Prediction (best-effort)
# -----------------------------
model = load_model(str(MODEL_PKL))
if model is None:
    st.warning("Model (model.pkl) not found or could not be loaded. Prediction will not work until a compatible model.pkl is placed in the project folder.")
else:
    st.subheader("🏷️ Predict Price (using loaded model)")
    # Try to detect model input size to provide a compatible input ordering
    try:
        # Example: try a few plausible feature vectors
        test_input = [area_input, bedrooms_input, overall_qual_input, gr_liv_area_input, year_built_input]
        pred = model.predict([test_input])
        st.success(f"Model appears to accept {len(test_input)} features. Example prediction: {pred[0]}")

        if st.button("Predict Price from Sidebar Inputs"):
            try:
                pred_val = model.predict([test_input])
                st.success(f"Predicted Price: {pred_val[0]}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    except Exception:
        st.info("Loaded model couldn't be probed with a simple input. Use a model trained to accept the features you supply.")

# -----------------------------
# Footer / Debug
# -----------------------------
st.markdown("---")
st.write("If you want, I can adapt this file to match the exact feature-order your `model.pkl` expects. Reply 'adapt model input' and paste your model training feature list.")

