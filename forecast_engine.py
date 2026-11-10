import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from train_model import train_and_save_models

class SCMForecastEngine:
    def __init__(self, artifact_path='models/ridge_forecast_artifacts.joblib', data_path='sales_data.csv'):
        self.artifact_path = artifact_path
        self.data_path = data_path
        self.models = {}
        self.history = {}
        self.profiles = {}
        self.df_ml = None
        self.features = []
        self.historical_dates = []
        self.stores = []
        self.categories = []
        self.min_date = '2022-01'
        self.max_date = '2024-01'
        self._load_artifacts()

    def _load_artifacts(self):
        if not os.path.exists(self.artifact_path):
            print(f"[Engine] Model artifact tidak ditemukan di '{self.artifact_path}'. Melatih dan menyimpan artifact...")
            out_dir = os.path.dirname(self.artifact_path) or 'models'
            train_and_save_models(self.data_path, output_dir=out_dir)

        print(f"[Engine] Memuat model artifacts dari disk: {self.artifact_path}")
        artifacts = joblib.load(self.artifact_path)

        self.models = artifacts['models']
        self.features = artifacts['features']
        self.profiles = artifacts['profiles']
        self.history = artifacts['history']
        self.historical_dates = artifacts['historical_dates']
        self.stores = artifacts['stores']
        self.categories = artifacts['categories']
        self.df_ml = artifacts.get('df_ml')
        self.min_date = self.historical_dates[0].strftime('%Y-%m')
        self.max_date = self.historical_dates[-1].strftime('%Y-%m')

        print(f"[Engine] Sukses memuat {len(self.models)} model Ridge Regression dan {len(self.features)} fitur.")

    def forecast_range(self, start_month_str, end_month_str, store_filter=None, category_filter=None):
        start_dt = pd.to_datetime(start_month_str + '-01')
        end_dt = pd.to_datetime(end_month_str + '-01') + pd.offsets.MonthEnd(1)

        # Generate monthly intervals
        requested_dates = pd.date_range(start=start_dt, end=end_dt, freq='ME')

        # Map of historical actuals
        hist_records_map = {}
        for (s, p), grp in self.df_ml.groupby(['Store ID', 'Product ID']):
            for _, r in grp.iterrows():
                m_str = r['Date'].strftime('%Y-%m')
                hist_records_map[(s, p, m_str)] = {
                    'actual': float(r['Units Sold']),
                    'row': r
                }

        # Copy history for simulation of future lags
        sim_history = {k: v.copy() for k, v in self.history.items()}

        last_hist_dt = self.historical_dates[-1]

        # Determine all months from min(requested_dates[0], last_hist_dt + 1 month) up to end_dt
        all_months_to_process = []
        cur = requested_dates[0]
        while cur <= end_dt:
            all_months_to_process.append(cur)
            cur = cur + pd.offsets.MonthEnd(1)

        forecast_results = []

        # Autoregressive forward simulation if future months exist
        for dt in all_months_to_process:
            m_str = dt.strftime('%Y-%m')
            is_future = dt > last_hist_dt
            m_val = dt.month
            q_val = dt.quarter
            y_val = dt.year

            # Seasonality one-hot
            if m_val in [12, 1, 2]:
                season = (0, 0, 1)  # Spring=0, Summer=0, Winter=1
            elif m_val in [3, 4, 5]:
                season = (1, 0, 0)  # Spring=1
            elif m_val in [6, 7, 8]:
                season = (0, 1, 0)  # Summer=1
            else:
                season = (0, 0, 0)  # Autumn=0

            for (store, prod), prof in self.profiles.items():
                cat = prof['category']
                model = self.models[(store, prod)]

                if not is_future and (store, prod, m_str) in hist_records_map:
                    hist_data = hist_records_map[(store, prod, m_str)]
                    row = hist_data['row']
                    X_row = pd.DataFrame([row[self.features]])
                    pred_units = float(max(0, model.predict(X_row)[0]))
                    actual_units = hist_data['actual']
                    diff = actual_units - pred_units
                    mape = abs(diff) / actual_units * 100.0 if actual_units != 0 else 0.0
                    status = 'Historical Actual'
                else:
                    # Future month: Autoregressive forecasting using lag simulation
                    hist = sim_history[(store, prod)].sort_values('Date')
                    lags = hist['Units Sold'].iloc[-3:].tolist()
                    lag1 = lags[-1] if len(lags) >= 1 else 0
                    lag2 = lags[-2] if len(lags) >= 2 else lag1
                    lag3 = lags[-3] if len(lags) >= 3 else lag2
                    roll3 = float(np.mean([lag1, lag2, lag3]))

                    # Exogenous variables
                    if m_val in prof['monthly_exog'].index:
                        ex_vals = prof['monthly_exog'].loc[m_val].to_dict()
                    else:
                        ex_vals = prof['overall_exog'].to_dict()

                    p = ex_vals['Price']
                    cp = ex_vals['Competitor Pricing']
                    d = ex_vals['Discount']
                    prom = ex_vals['Promotion']
                    pg = p - cp
                    pr = p / (cp if cp != 0 else np.nan)
                    dr = d / 100.0
                    pd_feat = prom * dr

                    reg_n, reg_s, reg_w = prof['region']
                    sea_sp, sea_su, sea_w = season

                    feat_dict = {
                        'Price': p,
                        'Discount': d,
                        'Promotion': prom,
                        'Competitor Pricing': cp,
                        'Epidemic': ex_vals.get('Epidemic', 0),
                        'Weather Sunny': ex_vals.get('Weather Sunny', 0),
                        'Weather Rainy': ex_vals.get('Weather Rainy', 0),
                        'Weather Cloudy': ex_vals.get('Weather Cloudy', 0),
                        'Weather Snowy': ex_vals.get('Weather Snowy', 0),
                        'Year': y_val,
                        'Month': m_val,
                        'Quarter': q_val,
                        'Is_Year_Start': int(dt.is_year_start),
                        'Is_Year_End': int(dt.is_year_end),
                        'Is_Quarter_End': int(dt.is_quarter_end),
                        'PriceGap': pg,
                        'Priceratio': pr,
                        'DiscountRate': dr,
                        'PromotionDiscount': pd_feat,
                        'Sales_Lag_1': lag1,
                        'Sales_Lag_2': lag2,
                        'Sales_Lag_3': lag3,
                        'Rolling_Mean_3': roll3,
                        'Region_North': reg_n,
                        'Region_South': reg_s,
                        'Region_West': reg_w,
                        'Seasonality_Spring': sea_sp,
                        'Seasonality_Summer': sea_su,
                        'Seasonality_Winter': sea_w,
                    }

                    X_row = pd.DataFrame([feat_dict])[self.features]
                    pred_units = float(max(0, model.predict(X_row)[0]))
                    actual_units = None
                    diff = None
                    mape = None
                    status = 'Future Forecast'

                    # Append to sim_history so next future months can use this prediction as lag
                    sim_history[(store, prod)] = pd.concat([
                        sim_history[(store, prod)],
                        pd.DataFrame([{'Date': dt, 'Units Sold': pred_units}])
                    ], ignore_index=True)

                # Only include in results if it falls within the requested range
                if dt in requested_dates:
                    forecast_results.append({
                        'date': m_str,
                        'store_id': store,
                        'category': cat,
                        'product_id': prod,
                        'product_category_id': f"{prod}_{cat}",
                        'forecast': round(pred_units, 2),
                        'actual': round(actual_units, 2) if actual_units is not None else None,
                        'diff': round(diff, 2) if diff is not None else None,
                        'mape': round(mape, 2) if mape is not None else None,
                        'status': status
                    })

        # Convert to DataFrame for filtering and aggregations
        res_df = pd.DataFrame(forecast_results)

        if store_filter and store_filter != 'ALL':
            res_df = res_df[res_df['store_id'] == store_filter]

        if category_filter and category_filter != 'ALL':
            res_df = res_df[res_df['category'] == category_filter]

        if res_df.empty:
            return {
                'table_data': [],
                'summary': {
                    'total_forecast': 0,
                    'avg_monthly': 0,
                    'top_category': '-',
                    'top_store': '-',
                    'avg_mape': None,
                    'months_count': 0
                },
                'chart_timeline': {'labels': [], 'datasets': []},
                'chart_category': {'labels': [], 'data': []},
                'chart_store': {'labels': [], 'data': []},
                'store_category_matrix': []
            }

        # 1. KPI Summary
        total_forecast = float(res_df['forecast'].sum())
        unique_months = res_df['date'].nunique()
        avg_monthly = total_forecast / unique_months if unique_months > 0 else 0

        # Top Category and Store
        cat_sums = res_df.groupby('category')['forecast'].sum()
        top_cat = cat_sums.idxmax() if not cat_sums.empty else '-'
        store_sums = res_df.groupby('store_id')['forecast'].sum()
        top_store = store_sums.idxmax() if not store_sums.empty else '-'

        # Average MAPE on historical records if present
        actuals_exist = res_df['mape'].dropna()
        avg_mape = round(float(actuals_exist.mean()), 2) if len(actuals_exist) > 0 else None

        # 2. Aggregations: Forecast per Category per Store
        matrix_df = (
            res_df.groupby(['store_id', 'category'])
            .agg({
                'forecast': 'sum',
                'actual': lambda x: round(x.dropna().sum(), 2) if len(x.dropna()) > 0 else None
            })
            .reset_index()
        )
        matrix_df['forecast'] = matrix_df['forecast'].round(2)
        store_category_matrix = matrix_df.to_dict('records')

        # 3. Chart 1: Timeline Trend (Monthly forecast per Category or per Store)
        # Group by Month and Category
        timeline_cats = sorted(res_df['category'].unique().tolist())
        dates_list = sorted(res_df['date'].unique().tolist())
        timeline_datasets = []

        palette = [
            '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
            '#06B6D4', '#EC4899', '#6366F1', '#14B8A6', '#F97316'
        ]

        for i, c in enumerate(timeline_cats):
            c_df = res_df[res_df['category'] == c].groupby('date')['forecast'].sum()
            data_points = [round(float(c_df.get(d, 0)), 1) for d in dates_list]
            timeline_datasets.append({
                'label': c,
                'data': data_points,
                'borderColor': palette[i % len(palette)],
                'backgroundColor': palette[i % len(palette)] + '20',
                'borderWidth': 2,
                'tension': 0.3,
                'fill': False,
                'pointRadius': 4
            })

        chart_timeline = {
            'labels': dates_list,
            'datasets': timeline_datasets
        }

        # 4. Chart 2: Category Breakdown
        cat_labels = list(cat_sums.index)
        cat_data = [round(float(v), 1) for v in cat_sums.values]
        chart_category = {
            'labels': cat_labels,
            'data': cat_data,
            'colors': palette[:len(cat_labels)]
        }

        # 5. Chart 3: Store Breakdown
        store_labels = list(store_sums.index)
        store_data = [round(float(v), 1) for v in store_sums.values]
        chart_store = {
            'labels': store_labels,
            'data': store_data,
            'colors': ['#2563EB', '#7C3AED', '#059669', '#D97706', '#DC2626'][:len(store_labels)]
        }

        return {
            'table_data': res_df.sort_values(by=['date', 'store_id', 'category']).to_dict('records'),
            'summary': {
                'total_forecast': round(total_forecast, 2),
                'avg_monthly': round(avg_monthly, 2),
                'top_category': top_cat,
                'top_store': top_store,
                'avg_mape': avg_mape,
                'months_count': unique_months
            },
            'chart_timeline': chart_timeline,
            'chart_category': chart_category,
            'chart_store': chart_store,
            'store_category_matrix': store_category_matrix
        }
