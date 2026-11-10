from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

class InventoryEngine:
    TARGET_PRODUCTS = ["P0001", "P0002", "P0003"]
    ORDERING_COST = 200.0
    HOLDING_COST_RATE = 0.05
    MARGIN = 0.25
    VENDOR_ORDERING_COST = 800.0
    VENDOR_HOLDING_COST_RATE = 0.05
    SERVICE_LEVEL_Z = 1.8808  # target service level 97%
    SERVICE_LEVEL_SCENARIOS = {
        "90%": 1.2816,
        "95%": 1.6449,
        "97%": 1.8808,
        "99%": 2.3263,
    }
    DAYS_PER_YEAR = 365.0

    LEAD_TIME_MAP = {
        "Groceries": {"lead_time": 5.0, "std_lead_time": 1.0, "max_delay": 2.0},
        "Clothing": {"lead_time": 7.0, "std_lead_time": 1.5, "max_delay": 3.0},
        "Toys": {"lead_time": 7.0, "std_lead_time": 1.5, "max_delay": 3.0},
        "Electronics": {"lead_time": 10.0, "std_lead_time": 2.0, "max_delay": 4.0},
        "Furniture": {"lead_time": 14.0, "std_lead_time": 2.5, "max_delay": 5.0},
    }

    def __init__(self, forecast_engine: Any, data_path: str | Path = "sales_data.csv"):
        self.forecast_engine = forecast_engine
        self.data_path = Path(data_path)
        self.daily_data = self._load_daily_data()
        self.forecast_start, self.forecast_end = self._get_forecast_period()
        self._cache: dict[str, pd.DataFrame] | None = None

    def _load_daily_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset tidak ditemukan: {self.data_path}")

        df = pd.read_csv(self.data_path)
        required = {
            "Date", "Store ID", "Product ID", "Category", "Demand",
            "Units Sold", "Inventory Level", "Price"
        }
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Kolom dataset belum lengkap: {missing}")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).copy()
        df = df[df["Product ID"].isin(self.TARGET_PRODUCTS)].copy()
        if df.empty:
            raise ValueError("Tidak ditemukan data untuk produk P0001, P0002, dan P0003.")

        df["StoreProduct_ID"] = (
            df["Store ID"].astype(str) + "_" +
            df["Product ID"].astype(str) + "_" +
            df["Category"].astype(str)
        )
        for col in ["Demand", "Units Sold", "Inventory Level", "Price"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values(["StoreProduct_ID", "Date"]).reset_index(drop=True)

    def _get_forecast_period(self) -> tuple[str, str]:
        last_month = pd.Timestamp(self.forecast_engine.historical_dates[-1]).to_period("M")
        first_future_month = (last_month + 1).to_timestamp()
        last_future_month = (last_month + 12).to_timestamp()
        return first_future_month.strftime("%Y-%m"), last_future_month.strftime("%Y-%m")

    def _get_12_month_forecast(self) -> pd.DataFrame:
        result = self.forecast_engine.forecast_range(
            start_month_str=self.forecast_start,
            end_month_str=self.forecast_end,
            store_filter="ALL",
            category_filter="ALL",
        )
        rows = result.get("table_data", [])
        if not rows:
            raise ValueError(
                "Forecast engine tidak menghasilkan data. Periksa model artifact dan periode forecast."
            )

        forecast_df = pd.DataFrame(rows)
        required = {"store_id", "product_id", "forecast", "date"}
        missing = sorted(required.difference(forecast_df.columns))
        if missing:
            raise ValueError(f"Kolom output forecast tidak sesuai: {missing}")

        forecast_df["forecast"] = pd.to_numeric(forecast_df["forecast"], errors="coerce").fillna(0)
        annual = (
            forecast_df.groupby(["store_id", "product_id"], as_index=False)
            .agg(
                Forecast_12M_Demand=("forecast", "sum"),
                Forecast_Months=("date", "nunique"),
            )
            .rename(columns={"store_id": "Store ID", "product_id": "Product ID"})
        )
        if not (annual["Forecast_Months"] == 12).all():
            raise ValueError("Forecast yang diterima tidak mencakup tepat 12 bulan.")
        annual["Forecast_12M_Demand"] = annual["Forecast_12M_Demand"].clip(lower=0)
        return annual

    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denominator = denominator.replace(0, np.nan)
        return numerator / denominator

    def _calculate_historical(self) -> pd.DataFrame:
        df = self.daily_data.copy()
        total_observed_days = df["Date"].nunique()
        if total_observed_days == 0:
            raise ValueError("Tidak ada tanggal observasi pada dataset.")
        yearly_factor = self.DAYS_PER_YEAR / total_observed_days

        df["Lost Sales"] = (df["Demand"] - df["Units Sold"]).clip(lower=0)
        df["Is Stockout"] = (df["Demand"] > df["Units Sold"]).astype(int)

        rows = []
        for store_product_id, group in df.groupby("StoreProduct_ID", sort=True):
            avg_price = group["Price"].mean()
            unit_cost = avg_price * (1.0 - self.MARGIN)
            annual_demand = group["Demand"].sum() * yearly_factor
            annual_sold = group["Units Sold"].sum() * yearly_factor
            annual_lost_sales = group["Lost Sales"].sum() * yearly_factor
            avg_daily_demand = group["Demand"].mean()
            std_daily_demand = group["Demand"].std(ddof=1)
            avg_inventory = group["Inventory Level"].mean()
            avg_inventory_cost = avg_inventory * unit_cost
            annual_cogs = annual_sold * unit_cost
            itr = annual_cogs / avg_inventory_cost if avg_inventory_cost > 0 else np.nan
            days_supply = self.DAYS_PER_YEAR / itr if pd.notna(itr) and itr > 0 else np.nan
            total_demand = group["Demand"].sum()
            fill_rate = (group["Units Sold"].sum() / total_demand * 100) if total_demand > 0 else np.nan

            rows.append({
                "StoreProduct_ID": store_product_id,
                "Store ID": group["Store ID"].iloc[0],
                "Product ID": group["Product ID"].iloc[0],
                "Category": group["Category"].iloc[0],
                "Average Price": avg_price,
                "Unit Cost": unit_cost,
                "Annual Historical Demand": annual_demand,
                "Annual Sold": annual_sold,
                "Annual Lost Sales": annual_lost_sales,
                "Average Daily Demand": avg_daily_demand,
                "Standard Deviation of Daily Demand": std_daily_demand,
                "Average Inventory Units": avg_inventory,
                "Average Inventory Cost": avg_inventory_cost,
                "Annual COGS": annual_cogs,
                "Inventory Turnover Rate (ITR)": itr,
                "Inventory Days of Supply": days_supply,
                "Fill Rate (%)": fill_rate,
                "Stockout Frequency (%)": group["Is Stockout"].mean() * 100,
                "Annual Lost Profit": annual_lost_sales * avg_price * self.MARGIN,
            })

        return pd.DataFrame(rows)

    def _calculate_eoq(self, historical: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
        eoq = historical.merge(
            forecast, on=["Store ID", "Product ID"], how="inner", validate="one_to_one"
        )
        if len(eoq) != len(historical):
            missing = len(historical) - len(eoq)
            raise ValueError(f"Ada {missing} kombinasi Store-Product yang tidak memiliki hasil forecast.")

        eoq["Demand"] = eoq["Forecast_12M_Demand"]
        eoq["Cb"] = self.ORDERING_COST
        eoq["Hb"] = eoq["Unit Cost"] * self.HOLDING_COST_RATE
        eoq["EOQ"] = np.sqrt(
            self._safe_divide(2 * eoq["Demand"] * eoq["Cb"], eoq["Hb"])
        ).replace([np.inf, -np.inf], np.nan)
        eoq["Freq_Order"] = self._safe_divide(eoq["Demand"], eoq["EOQ"])
        eoq["Interval_Days"] = self._safe_divide(self.DAYS_PER_YEAR, eoq["Freq_Order"])
        eoq["Avg_Cycle_Inventory"] = eoq["EOQ"] / 2
        eoq["Annual_Cb"] = eoq["Freq_Order"] * eoq["Cb"]
        eoq["Annual_Hb"] = eoq["Avg_Cycle_Inventory"] * eoq["Hb"]
        eoq["Annual_TCb"] = eoq["Annual_Cb"] + eoq["Annual_Hb"]

        lead = eoq["Category"].map(self.LEAD_TIME_MAP)
        if lead.isna().any():
            unknown = sorted(eoq.loc[lead.isna(), "Category"].dropna().unique().tolist())
            raise ValueError(f"Belum ada asumsi lead time untuk kategori: {unknown}")
        eoq["Demand_Perday"] = eoq["Demand"] / self.DAYS_PER_YEAR
        eoq["Std_Demand"] = eoq["Standard Deviation of Daily Demand"].fillna(0)
        eoq["LeadTime_Days"] = lead.map(lambda x: x["lead_time"])
        eoq["Std_LeadTime"] = lead.map(lambda x: x["std_lead_time"])
        eoq["Max_Delay_Days"] = lead.map(lambda x: x["max_delay"])
        eoq["SS_Delay"] = eoq["Demand_Perday"] * eoq["Max_Delay_Days"]
        eoq["SS_Demand"] = self.SERVICE_LEVEL_Z * np.sqrt(
            eoq["LeadTime_Days"] * (eoq["Std_Demand"] ** 2)
            + (eoq["Demand_Perday"] ** 2) * (eoq["Std_LeadTime"] ** 2)
        )
        eoq["SS"] = np.ceil(eoq["SS_Demand"])

        for service_level, z_score in self.SERVICE_LEVEL_SCENARIOS.items():
            ss_scenario = z_score * np.sqrt(
                eoq["LeadTime_Days"] * (eoq["Std_Demand"] ** 2)
                + (eoq["Demand_Perday"] ** 2) * (eoq["Std_LeadTime"] ** 2)
            )
            eoq[f"SS_{service_level}"] = np.ceil(ss_scenario)
            eoq[f"ROP_{service_level}"] = (
                eoq["Demand_Perday"] * eoq["LeadTime_Days"]
                + eoq[f"SS_{service_level}"]
            )

        eoq["Demand_LeadTime"] = eoq["Demand_Perday"] * eoq["LeadTime_Days"]
        eoq["Reorder_Point"] = eoq["Demand_LeadTime"] + eoq["SS"]
        eoq["Max_Inventory_with_SS"] = eoq["EOQ"] + eoq["SS"]
        eoq["Avg_Inventory_with_SS"] = eoq["EOQ"] / 2 + eoq["SS"]
        eoq["TCb_SS"] = eoq["Freq_Order"] * eoq["Cb"] + eoq["Avg_Inventory_with_SS"] * eoq["Hb"]

        eoq["Hb_Sensitivity_AvgPrice"] = eoq["Average Price"] * self.HOLDING_COST_RATE
        eoq["EOQ_Sensitivity_AvgPrice"] = np.sqrt(
            self._safe_divide(2 * eoq["Demand"] * eoq["Cb"], eoq["Hb_Sensitivity_AvgPrice"])
        )
        return eoq

    def _calculate_coordination(self, eoq: pd.DataFrame) -> pd.DataFrame:
        coord = eoq[[
            "StoreProduct_ID", "Store ID", "Product ID", "Category", "Demand",
            "Unit Cost", "Cb", "Hb"
        ]].copy()
        coord["Cs"] = self.VENDOR_ORDERING_COST
        coord["Hs"] = coord["Unit Cost"] * self.VENDOR_HOLDING_COST_RATE

        coord["EOQ_NoCoord"] = np.sqrt(self._safe_divide(
            2 * coord["Demand"] * coord["Cb"], coord["Hb"]
        ))
        coord["TCb_NoCoord"] = (
            self._safe_divide(coord["Demand"], coord["EOQ_NoCoord"]) * coord["Cb"]
            + coord["EOQ_NoCoord"] / 2 * coord["Hb"]
        )
        coord["TCs_NoCoord"] = (
            self._safe_divide(coord["Demand"], coord["EOQ_NoCoord"]) * coord["Cs"]
            + coord["EOQ_NoCoord"] / 2 * coord["Hs"]
        )
        coord["TC_NoCoord"] = coord["TCb_NoCoord"] + coord["TCs_NoCoord"]

        coord["EOQ_Coord"] = np.sqrt(self._safe_divide(
            2 * coord["Demand"] * (coord["Cb"] + coord["Cs"]),
            coord["Hb"] + coord["Hs"],
        ))
        coord["TCb_Coord"] = (
            self._safe_divide(coord["Demand"], coord["EOQ_Coord"]) * coord["Cb"]
            + coord["EOQ_Coord"] / 2 * coord["Hb"]
        )
        coord["TCs_Coord"] = (
            self._safe_divide(coord["Demand"], coord["EOQ_Coord"]) * coord["Cs"]
            + coord["EOQ_Coord"] / 2 * coord["Hs"]
        )
        coord["TC_Coord"] = coord["TCb_Coord"] + coord["TCs_Coord"]
        coord["Savings_Buyer"] = coord["TCb_NoCoord"] - coord["TCb_Coord"]
        coord["Savings_Supplier"] = coord["TCs_NoCoord"] - coord["TCs_Coord"]
        coord["Savings_Total"] = coord["TC_NoCoord"] - coord["TC_Coord"]
        coord["Savings_Pct"] = self._safe_divide(
            coord["Savings_Total"], coord["TC_NoCoord"]
        ) * 100
        return coord

    def _build_eoq_tradeoff(self, eoq: pd.DataFrame, points: int = 120) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for _, item in eoq.sort_values("StoreProduct_ID").iterrows():
            q_star = float(item["EOQ"])
            demand = float(item["Demand"])
            cb = float(item["Cb"])
            hb = float(item["Hb"])
            if not np.isfinite(q_star) or q_star <= 0 or hb <= 0:
                continue
            q_values = np.linspace(max(1.0, q_star * 0.25), q_star * 2.5, points)
            ordering = (demand / q_values) * cb
            holding = (q_values / 2.0) * hb
            total = ordering + holding
            optimal_total = (demand / q_star) * cb + (q_star / 2.0) * hb
            for q, order_cost, hold_cost, total_cost in zip(q_values, ordering, holding, total):
                rows.append({
                    "StoreProduct_ID": item["StoreProduct_ID"], "Q": float(q),
                    "Ordering_Cost": float(order_cost), "Holding_Cost": float(hold_cost),
                    "Total_Cost": float(total_cost), "EOQ": q_star,
                    "Optimal_Total_Cost": float(optimal_total),
                })
        return pd.DataFrame(rows)

    def _build_sawtooth_profiles(self, eoq: pd.DataFrame, cycles: int = 3) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for _, item in eoq.sort_values("StoreProduct_ID").iterrows():
            q_star = float(item["EOQ"])
            safety_stock = float(item["SS"])
            rop = float(item["Reorder_Point"])
            daily_demand = float(item["Demand_Perday"])
            lead_time = float(item["LeadTime_Days"])
            if not np.isfinite(q_star) or q_star <= 0 or daily_demand <= 0:
                continue
            cycle_days = q_star / daily_demand
            max_inventory = q_star + safety_stock
            reorder_day = cycle_days - lead_time
            for cycle in range(cycles):
                start_day, end_day = cycle * cycle_days, (cycle + 1) * cycle_days
                for day, level, point_type in [
                    (start_day, max_inventory, "Cycle Start"),
                    (end_day, safety_stock, "Cycle End"),
                ]:
                    rows.append({
                        "StoreProduct_ID": item["StoreProduct_ID"], "Day": float(day),
                        "Inventory_Level": float(level), "Safety_Stock": safety_stock,
                        "Reorder_Point": rop, "Max_Inventory": max_inventory,
                        "EOQ": q_star, "Lead_Time": lead_time,
                        "Cycle_Days": cycle_days, "Point_Type": point_type,
                    })
                if reorder_day > 0:
                    rows.append({
                        "StoreProduct_ID": item["StoreProduct_ID"],
                        "Day": float(start_day + reorder_day), "Inventory_Level": rop,
                        "Safety_Stock": safety_stock, "Reorder_Point": rop,
                        "Max_Inventory": max_inventory, "EOQ": q_star,
                        "Lead_Time": lead_time, "Cycle_Days": cycle_days,
                        "Point_Type": "Reorder",
                    })
        return pd.DataFrame(rows).sort_values(["StoreProduct_ID", "Day"], kind="stable").reset_index(drop=True)

    @staticmethod
    def _build_summary(historical: pd.DataFrame, eoq: pd.DataFrame,
                       coordination: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
        no_coord = float(coordination["TC_NoCoord"].sum())
        coord = float(coordination["TC_Coord"].sum())
        savings = no_coord - coord
        return pd.DataFrame([{
            "Total Store-Product": int(inventory["StoreProduct_ID"].nunique()),
            "Need Reorder": int((inventory["Inventory Status"] == "Need Reorder").sum()),
            "Stockout": int((inventory["Inventory Status"] == "Stockout").sum()),
            "Overstock": int((inventory["Inventory Status"] == "Overstock").sum()),
            "Average Fill Rate (%)": float(historical["Fill Rate (%)"].mean()),
            "Annual Forecast Demand": float(eoq["Demand"].sum()),
            "Annual Lost Profit": float(historical["Annual Lost Profit"].sum()),
            "Total Cost No Coordination": no_coord,
            "Total Cost Coordination": coord,
            "Total Cost Savings": savings,
            "Cost Savings (%)": float(savings / no_coord * 100) if no_coord else 0.0,
        }])

    def get_analysis(self, refresh: bool = False) -> dict[str, Any]:
        if self._cache is not None and not refresh:
            return {key: value.copy() if isinstance(value, pd.DataFrame) else value
                    for key, value in self._cache.items()}

        historical = self._calculate_historical()
        forecast = self._get_12_month_forecast()
        eoq = self._calculate_eoq(historical, forecast)
        coordination = self._calculate_coordination(eoq)

        latest_stock = (
            self.daily_data.sort_values("Date")
            .groupby("StoreProduct_ID", as_index=False)
            .tail(1)[["StoreProduct_ID", "Date", "Inventory Level"]]
            .rename(columns={"Date": "Latest Date", "Inventory Level": "Current Stock"})
        )
        inventory = eoq.merge(latest_stock, on="StoreProduct_ID", how="left")
        inventory["Max_Inventory_Level"] = inventory["Max_Inventory_with_SS"]
        inventory["Inventory Status"] = np.select(
            [
                inventory["Current Stock"].isna(),
                inventory["Current Stock"] <= 0,
                inventory["Current Stock"] <= inventory["Reorder_Point"],
                inventory["Current Stock"] > inventory["Max_Inventory_Level"],
            ],
            ["Data Tidak Tersedia", "Stockout", "Need Reorder", "Overstock"],
            default="Safe",
        )
        inventory["Suggested Order Qty"] = np.where(
            inventory["Inventory Status"].isin(["Stockout", "Need Reorder"]),
            np.ceil(inventory["EOQ"].fillna(0)),
            0,
        )

        eoq_tradeoff = self._build_eoq_tradeoff(eoq)
        sawtooth = self._build_sawtooth_profiles(eoq)
        summary = self._build_summary(historical, eoq, coordination, inventory)

        self._cache = {
            "summary": summary,
            "historical": historical,
            "forecast": forecast,
            "eoq": eoq,
            "eoq_tradeoff": eoq_tradeoff,
            "sawtooth": sawtooth,
            "coordination": coordination,
            "inventory": inventory,
        }
        return {key: value.copy() for key, value in self._cache.items()}

    def get_inventory_data(self, refresh: bool = False) -> pd.DataFrame:
        return self.get_analysis(refresh=refresh)["inventory"]
