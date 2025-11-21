 

import streamlit as st
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
    
    # 🔄 Rename columns to your preferred names
    df.rename(columns={
        'GrLivArea': 'area',
        'Neighborhood': 'city',
        'SalePrice': 'price'
    }, inplace=True)
    
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


 