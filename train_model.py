import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error

def train_and_save_models(data_path='sales_data.csv', output_dir='models'):
    print("=" * 60)
    print("Memulai Training Model SCM Demand Forecasting (Ridge Regression)")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    print(f"[1/5] Membaca dataset dari: {data_path}")
    df = pd.read_csv(data_path)
    
    target_products = ['P0001', 'P0002', 'P0003']
    df = df[df['Product ID'].isin(target_products)].copy()
    if 'Demand' in df.columns:
        df = df.drop(columns=['Demand'])

    df['Date'] = pd.to_datetime(df['Date'])

    df['Weather Sunny'] = (df['Weather Condition'] == 'Sunny').astype(int)
    df['Weather Rainy'] = (df['Weather Condition'] == 'Rainy').astype(int)
    df['Weather Cloudy'] = (df['Weather Condition'] == 'Cloudy').astype(int)
    df['Weather Snowy'] = (df['Weather Condition'] == 'Snowy').astype(int)

    df['Product_Category_ID'] = df['Product ID'] + '_' + df['Category']

    # 2. Monthly Aggregation 
    print("[2/5] Mengagregasikan data bulanan (Monthly Aggregation)...")
    monthly_df = (
        df.groupby([
            pd.Grouper(key='Date', freq='ME'),
            'Store ID',
            'Product_Category_ID'
        ])
        .agg({
            'Product ID': 'first',
            'Category': 'first',
            'Units Sold': 'sum',
            'Units Ordered': 'sum',
            'Inventory Level': 'mean',
            'Price': 'mean',
            'Discount': 'mean',
            'Promotion': 'mean',
            'Competitor Pricing': 'mean',
            'Epidemic': 'max',
            'Region': 'first',
            'Weather Sunny': 'sum',
            'Weather Rainy': 'sum',
            'Weather Cloudy': 'sum',
            'Weather Snowy': 'sum',
            'Seasonality': 'first'
        })
        .reset_index()
    )

    # 3. Feature Engineering 
    print("[3/5] Melakukan Feature Engineering (Temporal, Harga, Lag, & Rolling)...")
    # Temporal
    monthly_df['Year'] = monthly_df['Date'].dt.year
    monthly_df['Month'] = monthly_df['Date'].dt.month
    monthly_df['Quarter'] = monthly_df['Date'].dt.quarter
    monthly_df['Is_Year_Start'] = monthly_df['Date'].dt.is_year_start.astype(int)
    monthly_df['Is_Year_End'] = monthly_df['Date'].dt.is_year_end.astype(int)
    monthly_df['Is_Quarter_End'] = monthly_df['Date'].dt.is_quarter_end.astype(int)

    # Price & Promotion
    monthly_df['PriceGap'] = monthly_df['Price'] - monthly_df['Competitor Pricing']
    monthly_df['Priceratio'] = monthly_df['Price'] / monthly_df['Competitor Pricing'].replace(0, np.nan)
    monthly_df['DiscountRate'] = monthly_df['Discount'] / 100.0
    monthly_df['PromotionDiscount'] = monthly_df['Promotion'] * monthly_df['DiscountRate']

    # Lag & Rolling
    monthly_df = monthly_df.sort_values(by=['Store ID', 'Product_Category_ID', 'Date'])
    monthly_df['Sales_Lag_1'] = monthly_df.groupby(['Store ID', 'Product_Category_ID'])['Units Sold'].shift(1)
    monthly_df['Sales_Lag_2'] = monthly_df.groupby(['Store ID', 'Product_Category_ID'])['Units Sold'].shift(2)
    monthly_df['Sales_Lag_3'] = monthly_df.groupby(['Store ID', 'Product_Category_ID'])['Units Sold'].shift(3)
    monthly_df['Rolling_Mean_3'] = monthly_df.groupby(['Store ID', 'Product_Category_ID'])['Units Sold'].transform(
        lambda x: x.shift(1).rolling(window=3).mean()
    )
    monthly_df.fillna(0, inplace=True)

    # One-Hot Encoding Region & Seasonality
    df_ml = pd.get_dummies(monthly_df, columns=['Region', 'Seasonality'], drop_first=True)

    all_dummies = ['Region_North', 'Region_South', 'Region_West', 
                   'Seasonality_Spring', 'Seasonality_Summer', 'Seasonality_Winter']
    for col in all_dummies:
        if col not in df_ml.columns:
            df_ml[col] = 0

    exclude_cols = ['Date', 'Store ID', 'Product_Category_ID', 'Product ID', 'Category',
                    'Units Sold', 'Units Ordered', 'Inventory Level']
    features = [c for c in df_ml.columns if c not in exclude_cols]
    print(f"      Total Fitur: {len(features)} kolom: {features}")

    # 4. Training 15 Ridge Regression Models
    print("[4/5] Melatih 15 Model Ridge Regression (alpha=1.0, random_state=42)...")
    models = {}
    history = {}
    profiles = {}
    evaluation_scores = {}

    exog_cols = ['Price', 'Discount', 'Promotion', 'Competitor Pricing', 'Epidemic',
                 'Weather Sunny', 'Weather Rainy', 'Weather Cloudy', 'Weather Snowy']

    for (store, prod), grp in df_ml.groupby(['Store ID', 'Product ID']):
        grp = grp.sort_values('Date')
        X = grp[features]
        y = grp['Units Sold']

        # Split 3 bulan terakhir untuk validasi akurasi
        X_train, X_test = X.iloc[:-3], X.iloc[-3:]
        y_train, y_test = y.iloc[:-3], y.iloc[-3:]

        # Model Ridge
        eval_model = Ridge(alpha=1.0, random_state=42)
        eval_model.fit(X_train, y_train)
        pred_test = eval_model.predict(X_test)
        mape_score = mean_absolute_percentage_error(y_test, pred_test) * 100.0

        # Full training
        final_model = Ridge(alpha=1.0, random_state=42)
        final_model.fit(X, y)

        models[(store, prod)] = final_model
        history[(store, prod)] = grp[['Date', 'Units Sold']].copy()

        cat = grp['Category'].iloc[0]
        region_tuple = (
            int(grp['Region_North'].iloc[0]),
            int(grp['Region_South'].iloc[0]),
            int(grp['Region_West'].iloc[0])
        )
        monthly_exog = grp.groupby('Month')[exog_cols].mean()
        overall_exog = grp[exog_cols].mean()

        profiles[(store, prod)] = {
            'category': cat,
            'region': region_tuple,
            'monthly_exog': monthly_exog,
            'overall_exog': overall_exog
        }

        series_key = f"{store}_{prod}"
        evaluation_scores[series_key] = round(float(mape_score), 2)
        print(f"      - Seri {series_key} ({cat}): MAPE H3 = {mape_score:.2f}%")

    avg_mape = float(np.mean(list(evaluation_scores.values())))
    print(f"      ==> Rata-rata MAPE Keseluruhan: {avg_mape:.2f}%")

    # 5. Simpan Artifact Model ke Disk
    print(f"[5/5] Menyimpan model artifacts ke direktori '{output_dir}'...")

    artifact_file = os.path.join(output_dir, 'ridge_forecast_artifacts.joblib')
    artifacts_data = {
        'model_name': 'Ridge Regression',
        'hyperparameters': {'alpha': 1.0, 'random_state': 42},
        'trained_at': datetime.now().isoformat(),
        'features': features,
        'models': models,
        'profiles': profiles,
        'history': history,
        'historical_dates': sorted(df_ml['Date'].drop_duplicates().tolist()),
        'stores': sorted(df_ml['Store ID'].unique().tolist()),
        'categories': sorted(df_ml['Category'].unique().tolist()),
        'df_ml': df_ml,
        'avg_mape': avg_mape,
        'evaluation_scores': evaluation_scores
    }

    joblib.dump(artifacts_data, artifact_file)
    print(f"      [OK] Model artifact berhasil disimpan: {artifact_file} ({os.path.getsize(artifact_file)} bytes)")

    metadata_file = os.path.join(output_dir, 'model_metadata.json')
    readable_metadata = {
        'model_type': 'Ridge Regression',
        'hyperparameters': {'alpha': 1.0, 'random_state': 42},
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'target_products': target_products,
        'stores_count': len(artifacts_data['stores']),
        'total_models_trained': len(models),
        'features_count': len(features),
        'features_list': features,
        'average_mape_percent': round(avg_mape, 2),
        'series_evaluation_mape': evaluation_scores
    }

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(readable_metadata, f, indent=2)
    print(f"      [OK] Model metadata tersimpan: {metadata_file}")

    print("=" * 60)
    print("Training Selesai! Model siap digunakan oleh forecast_engine.")
    print("=" * 60)
    return artifact_file

if __name__ == '__main__':
    train_and_save_models()
