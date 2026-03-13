SELECT
    DATE_TRUNC('month', st.transaction_date) AS month,
    s.region,
    p.category,
    SUM(st.quantity * st.unit_price * (1 - st.discount_pct)) AS net_revenue,
    SUM(st.quantity) AS units_sold
FROM sales_transactions st
JOIN stores s ON s.store_id = st.store_id
JOIN products p ON p.product_id = st.product_id
GROUP BY
    DATE_TRUNC('month', st.transaction_date),
    s.region,
    p.category
ORDER BY month, net_revenue DESC;
