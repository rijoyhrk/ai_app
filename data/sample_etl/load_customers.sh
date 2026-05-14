#!/usr/bin/env bash
# ==============================================================
# load_customers.sh
# Orchestrates the customer data load from S3 → Redshift
# ==============================================================
set -euo pipefail

S3_BUCKET="s3://datalake/raw/customers/"
REDSHIFT_CONN="host=redshift.company.com dbname=dwh user=etl_user"
LOG_FILE="/var/log/etl/customer_load_$(date +%Y%m%d).log"

echo "[INFO] Starting customer load pipeline at $(date)" | tee -a "$LOG_FILE"

# Step 1: Copy raw data from S3 into Redshift staging
psql "$REDSHIFT_CONN" -c "
COPY customer_staging (cust_id, cust_name, cust_email, signup_date, region)
FROM '$S3_BUCKET'
IAM_ROLE 'arn:aws:iam::123456789:role/RedshiftS3Role'
FORMAT AS CSV IGNOREHEADER 1;
" | tee -a "$LOG_FILE"

echo "[INFO] S3 copy to customer_staging complete" | tee -a "$LOG_FILE"

# Step 2: Run SQL transformation
psql "$REDSHIFT_CONN" -f /opt/etl/sql/customer_transformation.sql 2>&1 | tee -a "$LOG_FILE"

# Step 3: Validate row counts on cust_tab and customer
CUST_TAB_COUNT=$(psql "$REDSHIFT_CONN" -t -c "SELECT COUNT(*) FROM cust_tab;")
CUSTOMER_COUNT=$(psql "$REDSHIFT_CONN" -t -c "SELECT COUNT(*) FROM customer;")

echo "[INFO] cust_tab rows:  $CUST_TAB_COUNT" | tee -a "$LOG_FILE"
echo "[INFO] customer rows:  $CUSTOMER_COUNT" | tee -a "$LOG_FILE"

# Step 4: Trigger downstream refresh
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    /opt/etl/python/customer_data_pipeline.py \
    2>&1 | tee -a "$LOG_FILE"

echo "[INFO] Customer load pipeline completed at $(date)" | tee -a "$LOG_FILE"
