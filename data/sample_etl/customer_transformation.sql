-- ============================================================
-- customer_transformation.sql
-- Transforms and deduplicates customer records.
-- Source: CUST (legacy alias), customer_staging
-- Target: customer (production table)
-- ============================================================

-- Step 1: Clean and deduplicate from staging
CREATE OR REPLACE TABLE customer_clean AS
SELECT
    cust_id,
    UPPER(TRIM(cust_name))  AS cust_name,
    LOWER(TRIM(cust_email)) AS cust_email,
    signup_date,
    region,
    ROW_NUMBER() OVER (PARTITION BY cust_email ORDER BY signup_date ASC) AS row_num
FROM customer_staging
WHERE cust_email IS NOT NULL;

-- Step 2: Merge into production customer table
MERGE INTO customer AS tgt
USING (SELECT * FROM customer_clean WHERE row_num = 1) AS src
    ON tgt.cust_id = src.cust_id
WHEN MATCHED THEN
    UPDATE SET
        tgt.cust_name  = src.cust_name,
        tgt.cust_email = src.cust_email,
        tgt.region     = src.region,
        tgt.updated_at = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN
    INSERT (cust_id, cust_name, cust_email, signup_date, region, created_at)
    VALUES (src.cust_id, src.cust_name, src.cust_email, src.signup_date, src.region, CURRENT_TIMESTAMP());

-- Step 3: Pull metrics from legacy CUST table for validation
SELECT
    'legacy_cust_count'    AS metric,
    COUNT(*)               AS val
FROM CUST

UNION ALL

SELECT
    'new_customer_count'   AS metric,
    COUNT(*)               AS val
FROM customer;
