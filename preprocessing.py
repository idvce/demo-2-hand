import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_json('used_cars.json')  # Replace with your file
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Numerical columns - fill with median
num_cols = df.select_dtypes(include=[np.number]).columns
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical columns - fill with mode
cat_cols = df.select_dtypes(include=['object']).columns
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("Missing values after imputation:")
print(df.isnull().sum())

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply to key numerical columns
outlier_cols = ['price', 'mileage', 'year']
for col in outlier_cols:
    df = remove_outliers(df, col)
    print(f"Removed outliers from {col}: {len(df)} rows remaining")

# Alternative: Cap outliers instead of removing
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

# Create new features
df['age'] = 2024 - df['year']  # Car age
df['price_per_mile'] = df['price'] / (df['mileage'] + 1)  # Avoid division by zero
df['is_newer_car'] = (df['age'] <= 5).astype(int)

# Extract features from complex columns
if 'transmission' in df.columns:
    df['is_automatic'] = (df['transmission'].str.lower() == 'automatic').astype(int)

print("New features created:")
print(df[['age', 'price_per_mile', 'is_newer_car']].head())

# Method 1: Label Encoding (for ordinal data)
le = LabelEncoder()
ordinal_cols = ['condition', 'fuel_type']  # Example ordinal columns
for col in ordinal_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Method 2: One-Hot Encoding (for nominal data)
nominal_cols = ['make', 'model', 'location']
df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# Method 3: Frequency Encoding (for high cardinality)
def frequency_encoding(df, column):
    freq_enc = df[column].value_counts().to_dict()
    df[column + '_freq'] = df[column].map(freq_enc)
    return df

if 'model' in df.columns:
    df_encoded = frequency_encoding(df_encoded, 'model')

# If you have date columns (e.g., 'listed_date')
if 'listed_date' in df.columns:
    df['listed_date'] = pd.to_datetime(df['listed_date'], errors='coerce')
    df['days_listed'] = (pd.Timestamp.now() - df['listed_date']).dt.days
    df['month_listed'] = df['listed_date'].dt.month
    df['year_listed'] = df['listed_date'].dt.year

# Separate features and target
X = df_encoded.drop('price', axis=1)  # Assuming 'price' is target
y = df_encoded['price']

# Scale numerical features
scaler = StandardScaler()
num_features = X.select_dtypes(include=[np.number]).columns
X[num_features] = scaler.fit_transform(X[num_features])

print("Features after scaling:")
print(X[num_features].head())

# Check final dataset
X_processed = X
print("Final dataset shape:", X_processed.shape)
print("\nCorrelation with target:")
correlations = pd.concat([X_processed, y], axis=1).corr()['price'].sort_values(ascending=False)
print(correlations.head(10))

# Save processed data
X_processed.to_csv('X_processed.csv', index=False)
y.to_csv('y_processed.csv', index=False)
print("✅ Data preprocessing complete!")