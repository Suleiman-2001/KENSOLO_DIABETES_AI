# engines/business_graph_engine.py
import os
import matplotlib
matplotlib.use("Agg")  # ensures saving works without GUI
import matplotlib.pyplot as plt
import pandas as pd


def _save_bar_chart(data_dict, title, xlabel, ylabel, save_path):
    """
    Saves a bar chart from a dictionary.
    """
    if not data_dict or not isinstance(data_dict, dict):
        return None

    labels = list(data_dict.keys())
    values = list(data_dict.values())

    if len(labels) == 0:
        return None

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def _save_line_chart(data_dict, title, xlabel, ylabel, save_path):
    """
    Saves a line chart from a dictionary.
    """
    if not data_dict or not isinstance(data_dict, dict):
        return None

    x = list(data_dict.keys())
    y = list(data_dict.values())

    if len(x) == 0:
        return None

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def generate_business_graphs(df, business_insights, folder="outputs/graphs"):
    """
    Generates business charts based on business_insights results:
    - daily/weekly/monthly sales trend
    - top products
    - top categories
    - top locations
    - best customers
    """
    os.makedirs(folder, exist_ok=True)
    saved = []

    if not business_insights or "error" in business_insights:
        return saved

    # ----------------------------
    # SALES TRENDS
    # ----------------------------
    sales_trends = business_insights.get("sales_trends", {})

    daily = sales_trends.get("daily_trend", {})
    weekly = sales_trends.get("weekly_trend", {})
    monthly = sales_trends.get("monthly_trend", {})

    p1 = _save_line_chart(
        daily,
        title="Daily Sales Trend (Last 30 days)",
        xlabel="Day",
        ylabel="Sales",
        save_path=os.path.join(folder, "sales_trend_daily.png")
    )
    if p1: saved.append(p1)

    p2 = _save_line_chart(
        weekly,
        title="Weekly Sales Trend (Last 12 weeks)",
        xlabel="Week",
        ylabel="Sales",
        save_path=os.path.join(folder, "sales_trend_weekly.png")
    )
    if p2: saved.append(p2)

    p3 = _save_line_chart(
        monthly,
        title="Monthly Sales Trend (Last 12 months)",
        xlabel="Month",
        ylabel="Sales",
        save_path=os.path.join(folder, "sales_trend_monthly.png")
    )
    if p3: saved.append(p3)

    # ----------------------------
    # TOP PRODUCTS + CATEGORIES
    # ----------------------------
    top_pc = business_insights.get("top_products_categories", {})
    top_products = top_pc.get("top_products", {})
    top_categories = top_pc.get("top_categories", {})

    p4 = _save_bar_chart(
        top_products,
        title="Top Products by Sales",
        xlabel="Product",
        ylabel="Sales",
        save_path=os.path.join(folder, "top_products_sales.png")
    )
    if p4: saved.append(p4)

    p5 = _save_bar_chart(
        top_categories,
        title="Top Categories by Sales",
        xlabel="Category",
        ylabel="Sales",
        save_path=os.path.join(folder, "top_categories_sales.png")
    )
    if p5: saved.append(p5)

    # ----------------------------
    # TOP LOCATIONS
    # ----------------------------
    top_loc = business_insights.get("top_locations", {})
    top_locations = top_loc.get("top_locations", {})

    p6 = _save_bar_chart(
        top_locations,
        title="Top Locations by Revenue",
        xlabel="Location",
        ylabel="Revenue",
        save_path=os.path.join(folder, "top_locations_revenue.png")
    )
    if p6: saved.append(p6)

    # ----------------------------
    # CUSTOMER ANALYTICS
    # ----------------------------
    cust = business_insights.get("customer_analytics", {})
    best_customers = cust.get("best_customers_high_value", {})
    frequent_customers = cust.get("most_frequent_customers", {})

    p7 = _save_bar_chart(
        best_customers,
        title="Best Customers (High Value)",
        xlabel="Customer",
        ylabel="Total Spend",
        save_path=os.path.join(folder, "best_customers_value.png")
    )
    if p7: saved.append(p7)

    p8 = _save_bar_chart(
        frequent_customers,
        title="Most Frequent Customers",
        xlabel="Customer",
        ylabel="Transactions Count",
        save_path=os.path.join(folder, "frequent_customers.png")
    )
    if p8: saved.append(p8)

    print(f"✅ Business graphs saved: {len(saved)} files in {folder}")
    return saved
