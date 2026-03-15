from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def missing_data_pipeline(df):
    # Define columns
    num_cols = ['mileage', 'engine_size', 'year']
    cat_cols = ['condition', 'transmission']
    
    # Numerical pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    return preprocessor
def validate_imputation(df_original, df_filled):
    """Check if imputation makes sense"""
    
    # 1. No missing values left
    assert df_filled.isnull().sum().sum() == 0
    
    # 2. Distributions similar
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for i, col in enumerate(['mileage', 'engine_size']):
        df_original[col].hist(alpha=0.5, label='Original', ax=axes[0,i])
        df_filled[col].hist(alpha=0.5, label='Filled', ax=axes[0,i])
        axes[0,i].legend()
        axes[0,i].set_title(f'{col} Distribution')
    
    plt.tight_layout()
    plt.show()

validate_imputation(df_original, df_filled)

# Use it
preprocessor = missing_data_pipeline(df)
X_clean = preprocessor.fit_transform(df[num_cols + cat_cols])