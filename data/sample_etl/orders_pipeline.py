"""
Orders Pipeline - processes order data and joins with customer dimension.
Uses cust_data as an alias for the customer table in legacy queries.
"""
import pandas as pd
import sqlalchemy as sa

engine = sa.create_engine("postgresql://user:pass@localhost/dwh")


def extract_recent_orders():
    query = """
        SELECT
            o.order_id,
            o.cust_id,
            o.order_date,
            o.total_amount,
            o.status,
            c.cust_name,
            c.region
        FROM orders o
        INNER JOIN cust_data c ON o.cust_id = c.cust_id
        WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
    """
    return pd.read_sql(query, engine)


def calculate_customer_ltv():
    """Compute lifetime value aggregated by customer."""
    query = """
        WITH customer_orders AS (
            SELECT
                cd.cust_id,
                cd.cust_name,
                SUM(o.total_amount) AS total_spend
            FROM cust_data cd
            LEFT JOIN orders o ON cd.cust_id = o.cust_id
            WHERE o.status = 'COMPLETED'
            GROUP BY cd.cust_id, cd.cust_name
        )
        SELECT * FROM customer_orders
        ORDER BY total_spend DESC
    """
    return pd.read_sql(query, engine)


def write_ltv_report(df):
    df.to_sql("customer_ltv_report", engine, if_exists="replace", index=False)
    print(f"Written {len(df)} rows to customer_ltv_report")


if __name__ == "__main__":
    orders_df = extract_recent_orders()
    ltv_df = calculate_customer_ltv()
    write_ltv_report(ltv_df)
