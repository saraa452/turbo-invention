WITH sales_base AS (
    SELECT
        transaction_id,
        transaction_date,
        customer_id,
        product_id,
        store_id,
        quantity,
        unit_price,
        discount_pct,
        quantity * unit_price AS gross_revenue,
        quantity * unit_price * (1 - discount_pct) AS net_revenue
    FROM sales_transactions
)
SELECT
    COUNT(*) AS total_transactions,
    COUNT(DISTINCT customer_id) AS active_customers,
    SUM(net_revenue) AS total_net_revenue,
    AVG(net_revenue) AS avg_ticket,
    SUM(gross_revenue - net_revenue) AS total_discount_value
FROM sales_base;

WITH sales_by_category AS (
    SELECT
        p.category,
        SUM(st.quantity * st.unit_price * (1 - st.discount_pct)) AS net_revenue
    FROM sales_transactions st
    JOIN products p ON p.product_id = st.product_id
    GROUP BY p.category
)
SELECT
    category,
    net_revenue,
    RANK() OVER (ORDER BY net_revenue DESC) AS category_rank
FROM sales_by_category
ORDER BY net_revenue DESC;
