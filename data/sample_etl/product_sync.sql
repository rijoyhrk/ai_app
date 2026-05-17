-- ============================================================
-- product_sync.sql
-- Syncs product catalog from source → DWH prod_tbl
-- ============================================================

-- Incremental load from product_staging into prod_tbl (alias for product table)
INSERT INTO prod_tbl (product_id, product_name, category, price, updated_at)
SELECT
    ps.product_id,
    ps.product_name,
    ps.category,
    ps.price,
    CURRENT_TIMESTAMP()
FROM product_staging ps
WHERE ps.updated_at > (SELECT MAX(updated_at) FROM prod_tbl);

-- Cross-reference customer orders with product catalog
SELECT
    c.cust_name,
    p.product_name,
    COUNT(oi.order_item_id) AS purchase_count
FROM CUSTOMER c
JOIN orders o ON c.cust_id = o.cust_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN prod_tbl p ON oi.product_id = p.product_id
GROUP BY c.cust_name, p.product_name
ORDER BY purchase_count DESC
LIMIT 100;
