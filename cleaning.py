import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_json('used_cars.json')  # Replace with your file
print("Initial dataset shape:", df.shape)

print("=== DUPLICATES ===")
print(f"Total duplicates: {df.duplicated().sum()}")

# Remove exact duplicates
df_clean = df.drop_duplicates()
print(f"After removal: {len(df_clean)} rows")

# Remove duplicates by key columns only
key_cols = ['make', 'model', 'year', 'mileage', 'price']
df_clean = df.drop_duplicates(subset=key_cols)
print(f"Key duplicates removed: {len(df_clean)} rows")

print("=== INVALID DATA ===")

# 1. Negative values (impossible)
df_clean = df_clean[df_clean['mileage'] >= 0]
df_clean = df_clean[df_clean['price'] > 0]
df_clean = df_clean[df_clean['year'] > 1900]

# 2. Impossible combinations
df_clean = df_clean[df_clean['year'] <= 2024]  # Future cars?
df_clean = df_clean[df_clean['mileage'] <= 500000]  # 500k max realistic

# 3. Price too low/high
price_q1, price_q3 = df_clean['price'].quantile([0.01, 0.99])
df_clean = df_clean[(df_clean['price'] >= price_q1) & 
                   (df_clean['price'] <= price_q3)]

print(f"Invalid data removed: {len(df_clean)} rows")

print("=== STRING CLEANING ===")

def clean_text(df):
    for col in df.select_dtypes(include=['object']).columns:
        # Remove extra spaces
        df[col] = df[col].str.strip()
        df[col] = df[col].str.title()  # "toyota" → "Toyota"
        
        # Fix common typos
        df[col] = df[col].replace({
            'Transmision': 'Transmission',
            'Automatiс': 'Automatic',
            'Gasolin': 'Gasoline'
        })
        
        # Standardize
        df[col] = df[col].replace('4x4', 'Four Wheel Drive')
        df[col] = df[col].replace('FWD', 'Front Wheel Drive')
    
    return df

df_clean = clean_text(df_clean)
print("Text standardized!")

print("=== OUTLIERS ===")

def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return outliers, lower, upper

# CAR-SPECIFIC outlier rules
outlier_cols = ['mileage', 'price', 'engine_size']

for col in outlier_cols:
    outliers, lower, upper = detect_outliers_iqr(df_clean, col)
    print(f"{col}: {len(outliers)} outliers [{lower:.0f}, {upper:.0f}]")
    
    # Remove OR cap
    # df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)  # CAP instead

# Z-score method (alternative)
df_clean = df_clean[(np.abs(stats.zscore(df_clean[outlier_cols])) < 3).all(axis=1)]

print("=== CONSISTENCY ===")

# 1. Year vs Age
df_clean['age'] = 2024 - df_clean['year']
df_clean = df_clean[df_clean['age'] >= 0]

# 2. Mileage vs Year (realistic wear)
df_clean['expected_mileage'] = df_clean['age'] * 15000  # 15k/year avg
df_clean = df_clean[df_clean['mileage'] <= df_clean['expected_mileage'] * 3]

# 3. Price vs Mileage (basic check)
df_clean['price_per_mile'] = df_clean['price'] / (df_clean['mileage'] + 1)
df_clean = df_clean[df_clean['price_per_mile'] >= 0.05]  # $0.05/mile min

print("Consistency checks passed!")

print("=== DATA TYPES ===")

# Convert to proper types
df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce')
df_clean['mileage'] = pd.to_numeric(df_clean['mileage'], errors='coerce')
df_clean['price'] = pd.to_numeric(df_clean['price'], errors='coerce')

# Date columns
if 'listed_date' in df_clean.columns:
    df_clean['listed_date'] = pd.to_datetime(df_clean['listed_date'], errors='coerce')

# Categorical
cat_cols = ['make', 'model', 'transmission']
for col in cat_cols:
    df_clean[col] = df_clean[col].astype('category')

print(df_clean.dtypes)