"""
Customer Data Pipeline - PySpark ETL
Reads raw customer data, applies transformations, and writes to data warehouse.
"""
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CustomerPipeline").getOrCreate()


def load_raw_customers():
    """Load raw customer records from source system via cust_tab."""
    df = spark.sql("""
        SELECT
            c.cust_id,
            c.cust_name,
            c.cust_email,
            c.signup_date,
            a.region
        FROM cust_tab c
        LEFT JOIN address_dim a ON c.cust_id = a.cust_id
        WHERE c.is_active = 1
    """)
    return df


def enrich_with_orders(cust_df):
    """Join customer data with order history from ord_tab."""
    orders_df = spark.sql("""
        SELECT
            o.cust_id,
            COUNT(o.order_id)   AS total_orders,
            SUM(o.order_amount) AS lifetime_value
        FROM ord_tab o
        WHERE o.status = 'COMPLETED'
        GROUP BY o.cust_id
    """)
    return cust_df.join(orders_df, on="cust_id", how="left")


def write_to_warehouse(df):
    """Write enriched customer data to the customer_360 mart table."""
    df.write.mode("overwrite").saveAsTable("customer_360")


def run_pipeline():
    raw = load_raw_customers()
    enriched = enrich_with_orders(raw)
    enriched = enriched.withColumn("segment",
        F.when(F.col("lifetime_value") > 10000, "premium")
         .when(F.col("lifetime_value") > 1000, "regular")
         .otherwise("basic")
    )
    write_to_warehouse(enriched)
    print(f"Pipeline complete. Rows written: {enriched.count()}")


if __name__ == "__main__":
    run_pipeline()
