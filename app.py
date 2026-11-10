import os
import io
import csv
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response
import json
from forecast_engine import SCMForecastEngine
from historical_engine import HistoricalEngine
from inventory_engine import InventoryEngine

app = Flask(__name__)

engine = SCMForecastEngine()
hist_engine = HistoricalEngine('sales_data.csv')
inventory_engine = InventoryEngine(forecast_engine=engine, data_path='sales_data.csv')

@app.route('/')
def index():
    return render_template(
        'index.html',
        min_date='2024-02',
        max_date='2030-12',
        default_start='2024-02',
        default_end='2024-07',
        stores=engine.stores,
        categories=engine.categories
    )

@app.route('/api/historical', methods=['POST', 'GET'])
def api_historical():
    try:
        if request.method == 'POST':
            data = request.get_json(silent=True) or {}
        else:
            data = request.args
        store = data.get('store', 'ALL')
        category = data.get('category', 'ALL')
        start = data.get('start_date') or hist_engine.min_date
        end = data.get('end_date') or hist_engine.max_date
        try:
            if pd.to_datetime(start) > pd.to_datetime(end):
                return jsonify({'status': 'error', 'message': 'Tanggal Mulai tidak boleh lebih besar dari Tanggal Selesai.'}), 400
        except (ValueError, TypeError):
            return jsonify({'status': 'error', 'message': 'Format tanggal tidak valid.'}), 400
        if store != 'ALL' and store not in hist_engine.stores:
            return jsonify({'status': 'error', 'message': f'Store {store} tidak ditemukan.'}), 400
        if category != 'ALL' and category not in hist_engine.categories:
            return jsonify({'status': 'error', 'message': f'Kategori {category} tidak ditemukan.'}), 400
        return jsonify({'status': 'success',
                        'params': {'store': store, 'category': category,
                                   'start_date': start, 'end_date': end},
                        'data': hist_engine.analyze(store, category, start, end)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/info')
def api_info():
    return jsonify({
        'status': 'success',
        'forecast_start_date': '2024-02',
        'stores': engine.stores,
        'categories': engine.categories,
        'model_name': 'Ridge Regression',
        'model_params': {'alpha': 1.0, 'random_state': 42}
    })

@app.route('/api/forecast', methods=['POST'])
def api_forecast():
    try:
        data = request.get_json() or {}
        start_date = data.get('start_date', '2024-02')
        end_date = data.get('end_date', '2024-07')
        store_filter = data.get('store', 'ALL')
        category_filter = data.get('category', 'ALL')

        if start_date > end_date:
            return jsonify({
                'status': 'error',
                'message': 'Tanggal Mulai tidak boleh lebih besar dari Tanggal Selesai.'
            }), 400

        result = engine.forecast_range(
            start_month_str=start_date,
            end_month_str=end_date,
            store_filter=store_filter,
            category_filter=category_filter
        )

        return jsonify({
            'status': 'success',
            'params': {
                'start_date': start_date,
                'end_date': end_date,
                'store': store_filter,
                'category': category_filter,
                'model': 'Ridge Regression (alpha=1.0)'
            },
            'data': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/export-csv')
def export_csv():
    start_date = request.args.get('start_date', '2024-02')
    end_date = request.args.get('end_date', '2024-07')
    store_filter = request.args.get('store', 'ALL')
    category_filter = request.args.get('category', 'ALL')

    result = engine.forecast_range(
        start_month_str=start_date,
        end_month_str=end_date,
        store_filter=store_filter,
        category_filter=category_filter
    )

    table_data = result.get('table_data', [])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'Periode (Bulan)', 'Store ID', 'Kategori', 'Product ID',
        'Forecast Demand (Units Sold)', 'Model Prediksi'
    ])

    for row in table_data:
        writer.writerow([
            row.get('date', ''),
            row.get('store_id', ''),
            row.get('category', ''),
            row.get('product_id', ''),
            row.get('forecast', ''),
            'Ridge Regression (alpha=1.0)'
        ])

    csv_data = output.getvalue()
    filename = f"forecast_scm_{start_date}_to_{end_date}.csv"

    return Response(
        csv_data,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )
@app.route('/api/inventory', methods=['GET'])
def api_inventory():
    try:
        result = inventory_engine.get_analysis()
        response_data = {
            key: json.loads(df.to_json(orient='records', date_format='iso'))
            for key, df in result.items()
        }

        return jsonify({
            'status': 'success',
            'data': response_data
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("Menjalankan SCM Demand Forecasting Dashboard...")
    app.run(debug=True, host='127.0.0.1', port=5000)