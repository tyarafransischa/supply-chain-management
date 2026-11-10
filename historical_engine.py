import numpy as np
import pandas as pd

def _clean(x):
    if x is None:
        return None
    if isinstance(x, (np.floating, float)):
        return None if (np.isnan(x) or np.isinf(x)) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x

def _clean_list(arr):
    return [_clean(v) for v in arr]

class HistoricalEngine:
    WEATHER_ORDER = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
    SEASON_ORDER = ['Spring', 'Summer', 'Autumn', 'Winter']

    def __init__(self, data_path='sales_data.csv'):
        df = pd.read_csv(data_path, parse_dates=['Date'])
        df.columns = [c.strip() for c in df.columns]
        df = df.drop_duplicates().copy()
        df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
        df['PriceGap'] = df['Price'] - df['Competitor Pricing']
        self.df = df
        self.stores = sorted(df['Store ID'].unique().tolist())
        self.categories = sorted(df['Category'].unique().tolist())
        self.min_date = df['Date'].min().strftime('%Y-%m-%d')
        self.max_date = df['Date'].max().strftime('%Y-%m-%d')

    # filter
    def _filter(self, store, category, start, end):
        d = self.df
        if store and store != 'ALL':
            d = d[d['Store ID'] == store]
        if category and category != 'ALL':
            d = d[d['Category'] == category]
        if start:  # format YYYY-MM-DD (atau YYYY-MM)
            d = d[d['Date'] >= pd.to_datetime(start)]
        if end:
            end_dt = pd.to_datetime(end)
            if len(end) == 7:
                end_dt = end_dt + pd.offsets.MonthEnd(1)
            d = d[d['Date'] <= end_dt]
        return d

    # sections
    def _kpi(self, d):
        by_store = d.groupby('Store ID')['Units Sold'].sum()
        by_cat = d.groupby('Category')['Units Sold'].sum()
        monthly = d.groupby('Month')['Units Sold'].sum()
        return {
            'total_units': int(d['Units Sold'].sum()),
            'avg_daily_units': _clean(d.groupby('Date')['Units Sold'].sum().mean()),
            'n_rows': int(len(d)),
            'n_months': int(monthly.shape[0]),
            'top_store': by_store.idxmax(),
            'top_store_units': int(by_store.max()),
            'top_category': by_cat.idxmax(),
            'top_category_units': int(by_cat.max()),
            'peak_month': monthly.idxmax().strftime('%Y-%m'),
            'peak_month_units': int(monthly.max()),
        }

    def _trend(self, d):
        pivot = (d.pivot_table(index='Month', columns='Category',
                               values='Units Sold', aggfunc='sum').sort_index().fillna(0))
        total = d.groupby('Month')['Units Sold'].sum().sort_index()
        return {
            'labels': [m.strftime('%Y-%m') for m in pivot.index],
            'total': [int(v) for v in total.values],
            'series': {c: [int(v) for v in pivot[c].values] for c in pivot.columns},
        }

    def _sold_vs_demand(self, d):
        m = d.groupby('Month')[['Units Sold', 'Demand', 'Units Ordered']].sum().sort_index()
        return {
            'labels': [x.strftime('%Y-%m') for x in m.index],
            'units_sold': [int(v) for v in m['Units Sold']],
            'demand': [int(v) for v in m['Demand']],
            'units_ordered': [int(v) for v in m['Units Ordered']],
        }

    def _breakdown(self, d):
        def agg(col):
            g = d.groupby(col)['Units Sold'].sum().sort_values(ascending=False)
            return {'labels': g.index.tolist(), 'values': [int(v) for v in g.values]}
        return {'store': agg('Store ID'), 'category': agg('Category'), 'region': agg('Region')}

    def _weather_season(self, d):
        w = d.groupby('Weather Condition')['Units Sold'].mean()
        w = w.reindex([x for x in self.WEATHER_ORDER if x in w.index])
        s = d.groupby('Seasonality')['Units Sold'].mean()
        s = s.reindex([x for x in self.SEASON_ORDER if x in s.index])
        return {
            'weather': {'labels': w.index.tolist(), 'mean': _clean_list(w.values)},
            'season': {'labels': s.index.tolist(), 'mean': _clean_list(s.values)},
        }

    def _promo_price(self, d):
        promo = d.groupby('Promotion')['Units Sold'].mean()
        base, with_p = promo.get(0), promo.get(1)
        lift = None
        if base and with_p is not None and not np.isnan(base) and not np.isnan(with_p):
            lift = _clean((with_p - base) / base * 100)

        disc = d.groupby('Discount')['Units Sold'].mean().sort_index()

        bins = [-np.inf, -10, -5, 0, 5, 10, np.inf]
        labels = ['< -10', '-10 s/d -5', '-5 s/d 0', '0 s/d 5', '5 s/d 10', '> 10']
        gap = d.assign(GapBin=pd.cut(d['PriceGap'], bins=bins, labels=labels))
        gap_mean = gap.groupby('GapBin', observed=False)['Units Sold'].mean()

        return {
            'promo': {'tanpa_promosi': _clean(base), 'dengan_promosi': _clean(with_p), 'lift_pct': lift},
            'discount': {'labels': [f'{int(k)}%' for k in disc.index], 'mean': _clean_list(disc.values)},
            'price_gap': {'labels': labels, 'mean': _clean_list(gap_mean.values)},
        }

    def analyze(self, store='ALL', category='ALL', start=None, end=None):
        d = self._filter(store, category, start, end)
        if d.empty:
            raise ValueError('Tidak ada data untuk kombinasi filter tersebut.')
        return {
            'period': {'start': d['Date'].min().strftime('%Y-%m-%d'),
                       'end': d['Date'].max().strftime('%Y-%m-%d')},
            'kpi': self._kpi(d),
            'trend': self._trend(d),
            'sold_vs_demand': self._sold_vs_demand(d),
            'breakdown': self._breakdown(d),
            'weather_season': self._weather_season(d),
            'promo_price': self._promo_price(d),
        }