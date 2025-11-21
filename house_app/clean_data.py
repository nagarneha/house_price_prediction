import pandas as pd

df = pd.read_csv("train.csv")

df.rename(columns={
    "BedroomAbvGr": "bedrooms",
    "GrLivArea": "area",
    "SalePrice": "price"
}, inplace=True)

drop_cols = [
    "Street", "Alley", "LotShape", "LandContour", "Utilities",
    "LotConfig", "LandSlope", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
    "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual",
    "ExterCond", "Foundation", "BsmtQual", "BsmtCond",
    "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating",
    "HeatingQC", "CentralAir", "Electrical", "KitchenQual",
    "Functional", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PavedDrive", "PoolQC",
    "Fence", "MiscFeature", "SaleType", "SaleCondition"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])
df = df.fillna(df.median(numeric_only=True))

df.to_csv("cleaned_data.csv", index=False)

print("cleaned_data.csv successfully created!")
 

df = pd.read_csv("cleaned_data.csv")
