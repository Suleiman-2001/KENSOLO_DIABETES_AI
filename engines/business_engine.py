# engines/business_engine.py
import pandas as pd
import numpy as np

def _safe_to_datetime(series):
    """Try convert a column to datetime safely."""
    dt = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    if dt.isnull().all():
        return None
    return dt

def detect_business_columns(df):
    """Auto-detect common business columns."""
    cols = {c.lower(): c for c in df.columns}

    def find_col(keywords):
        for k in keywords:
            for col_lower, original in cols.items():
                if k in col_lower:
                    return original
        return None

    detected = {
        "date_col": find_col(["date", "order_date", "timestamp", "datetime", "time"]),
        "sales_col": find_col(["sales", "revenue", "amount", "total", "total_sales"]),
        "profit_col": find_col(["profit", "margin"]),
        "cost_col": find_col(["cost", "expense"]),
        "product_col": find_col(["product", "item", "sku"]),
        "category_col": find_col(["category", "cat", "department"]),
        "location_col": find_col(["location", "region", "country", "city", "branch", "store"]),
        "customer_col": find_col(["customer", "client", "buyer", "user"]),
        "order_id_col": find_col(["order_id", "invoice", "transaction", "receipt"]),
        "quantity_col": find_col(["qty", "quantity", "units"])
    }
    
    # Add warnings if key columns are missing
    detected["warnings"] = []
    if not detected["sales_col"]:
        detected["warnings"].append("Sales column not detected")
    if not detected["date_col"]:
        detected["warnings"].append("Date column not detected")
    if not detected["profit_col"]:
        detected["warnings"].append("Profit column not detected")
    
    return detected

def sales_trends(df, date_col, sales_col):
    if not date_col or not sales_col:
        return {"error": "date_col or sales_col not found"}

    temp = df.copy()
    temp[date_col] = _safe_to_datetime(temp[date_col])
    temp = temp.dropna(subset=[date_col, sales_col])
    if temp.empty:
        return {"error": "No valid datetime rows or sales data"}

    temp["day"] = temp[date_col].dt.date
    temp["week"] = temp[date_col].dt.to_period("W").astype(str)
    temp["month"] = temp[date_col].dt.to_period("M").astype(str)
    temp["hour"] = temp[date_col].dt.hour
    temp["weekday"] = temp[date_col].dt.day_name()

    # Return trends with safeguards
    return {
        "daily_trend": temp.groupby("day")[sales_col].sum().tail(30).to_dict(),
        "weekly_trend": temp.groupby("week")[sales_col].sum().tail(12).to_dict(),
        "monthly_trend": temp.groupby("month")[sales_col].sum().tail(12).to_dict(),
        "top_hours_by_revenue": temp.groupby("hour")[sales_col].sum().sort_values(ascending=False).head(5).to_dict(),
        "top_weekdays_by_revenue": temp.groupby("weekday")[sales_col].sum().sort_values(ascending=False).head(5).to_dict(),
    }

def top_products_categories(df, sales_col, product_col=None, category_col=None):
    if not sales_col:
        return {"error": "sales_col not found"}

    result = {}
    if product_col and product_col in df.columns:
        top_products = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(10)
        result["top_products"] = top_products.to_dict()
    if category_col and category_col in df.columns:
        top_categories = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False).head(10)
        result["top_categories"] = top_categories.to_dict()
    if not result:
        result["warning"] = "No product/category column found or insufficient data"
    return result

def top_locations(df, sales_col, location_col=None):
    if not sales_col or not location_col:
        return {"error": "sales_col or location_col not found"}

    top_locations = df.groupby(location_col)[sales_col].sum().sort_values(ascending=False).head(10)
    return {"top_locations": top_locations.to_dict()}

def customer_analytics(df, date_col, sales_col, customer_col):
    if not customer_col or not sales_col:
        return {"error": "customer_col or sales_col not found"}

    result = {}
    # Best customers
    top_customers = df.groupby(customer_col)[sales_col].sum().sort_values(ascending=False).head(10)
    result["best_customers_high_value"] = top_customers.to_dict()

    # Frequent buyers
    freq_customers = df.groupby(customer_col).size().sort_values(ascending=False).head(10)
    result["most_frequent_customers"] = freq_customers.to_dict()

    # Inactive customers
    if date_col and date_col in df.columns:
        temp = df.copy()
        temp[date_col] = _safe_to_datetime(temp[date_col])
        temp = temp.dropna(subset=[date_col])
        if not temp.empty:
            last_date = temp[date_col].max()
            cutoff = last_date - pd.Timedelta(days=30)
            last_purchase = temp.groupby(customer_col)[date_col].max()
            inactive = last_purchase[last_purchase < cutoff].sort_values().head(20)
            result["inactive_customers_30days"] = {str(k): str(v) for k, v in inactive.items()}
        else:
            result["inactive_customers_30days"] = {"warning": "Not enough date data to detect inactivity"}
    else:
        result["inactive_customers_30days"] = {"warning": "date_col missing -> cannot detect inactivity"}

    return result

def profit_analytics(df, sales_col=None, profit_col=None, cost_col=None, product_col=None, category_col=None):
    result = {}

    if profit_col in df.columns and product_col in df.columns:
        top_profit_products = df.groupby(product_col)[profit_col].sum().sort_values(ascending=False).head(10)
        result["top_products_by_profit"] = top_profit_products.to_dict()

    if sales_col and profit_col and product_col and all(c in df.columns for c in [sales_col, profit_col, product_col]):
        agg = df.groupby(product_col).agg(
            total_sales=(sales_col, "sum"),
            total_profit=(profit_col, "sum")
        )
        agg["profit_margin"] = np.where(agg["total_sales"] != 0, agg["total_profit"] / agg["total_sales"], 0)
        low_profit_high_sales = agg.sort_values(["total_sales", "profit_margin"], ascending=[False, True]).head(10)
        result["high_sales_low_profit_products"] = low_profit_high_sales.reset_index().to_dict(orient="records")

    if category_col and sales_col and profit_col and all(c in df.columns for c in [category_col, sales_col, profit_col]):
        cat = df.groupby(category_col).agg(
            total_sales=(sales_col, "sum"),
            total_profit=(profit_col, "sum")
        )
        cat["margin"] = np.where(cat["total_sales"] != 0, cat["total_profit"] / cat["total_sales"], 0)
        best_margins = cat.sort_values("margin", ascending=False).head(10)
        result["best_category_margins"] = best_margins.reset_index().to_dict(orient="records")

    if cost_col and profit_col and all(c in df.columns for c in [cost_col, profit_col]):
        corr = df[[cost_col, profit_col]].corr().iloc[0, 1]
        result["cost_profit_correlation"] = float(corr)

    if not result:
        result["warning"] = "Not enough columns detected for profit analysis"

    return result

def run_business_intelligence(df):
    detected = detect_business_columns(df)

    date_col = detected["date_col"]
    sales_col = detected["sales_col"]
    profit_col = detected["profit_col"]
    cost_col = detected["cost_col"]
    product_col = detected["product_col"]
    category_col = detected["category_col"]
    location_col = detected["location_col"]
    customer_col = detected["customer_col"]

    return {
        "detected_columns": detected,
        "sales_trends": sales_trends(df, date_col, sales_col),
        "top_products_categories": top_products_categories(df, sales_col, product_col, category_col),
        "top_locations": top_locations(df, sales_col, location_col),
        "customer_analytics": customer_analytics(df, date_col, sales_col, customer_col),
        "profit_analytics": profit_analytics(df, sales_col, profit_col, cost_col, product_col, category_col),
    }
