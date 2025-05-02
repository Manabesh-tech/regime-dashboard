# Even more simplified volatility query that avoids complex SQL features
volatility_query = f"""
WITH time_buckets AS (
    -- Generate 10-minute time buckets for the past 24 hours
    SELECT 
        generate_series(
            date_trunc('hour', '{start_time_utc}'::timestamp) + 
            INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 10),
            '{end_time_utc}'::timestamp,
            INTERVAL '10 minutes'
        ) AS bucket_start
),
pair_list AS (
    -- List of pairs to process
    SELECT unnest(ARRAY['{pairs_str}']) AS pair_name
),
pair_buckets AS (
    -- Create all combinations of time buckets and pairs
    SELECT 
        tb.bucket_start,
        tb.bucket_start + INTERVAL '10 minutes' AS bucket_end,
        pl.pair_name
    FROM 
        time_buckets tb
    CROSS JOIN
        pair_list pl
),
-- Get all trades within each bucket for each pair
bucket_trades AS (
    SELECT
        pb.bucket_start,
        pb.pair_name,
        tf.deal_price
    FROM
        pair_buckets pb
    JOIN
        public.trade_fill_fresh tf ON
        tf.created_at >= pb.bucket_start AND
        tf.created_at < pb.bucket_end AND
        tf.pair_name = pb.pair_name
),
-- Calculate aggregates for each bucket
price_stats AS (
    SELECT
        bucket_start,
        pair_name,
        MIN(deal_price) AS min_price,
        MAX(deal_price) AS max_price,
        COUNT(*) AS trade_count,
        AVG(deal_price) AS avg_price
    FROM
        bucket_trades
    GROUP BY
        bucket_start, pair_name
)

SELECT
    ps.bucket_start AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp_sg,
    ps.pair_name,
    ps.min_price,
    ps.max_price,
    ps.avg_price,
    ps.trade_count,
    -- Simple volatility estimate based on high-low range
    CASE 
        WHEN ps.min_price > 0 AND ps.min_price IS NOT NULL AND ps.max_price IS NOT NULL AND ps.trade_count >= 3
        THEN (ps.max_price - ps.min_price) / ps.min_price 
        ELSE NULL 
    END AS price_range_pct
FROM
    price_stats ps
WHERE
    ps.trade_count >= 3  -- Only include intervals with at least 3 trades
ORDER BY
    ps.pair_name, ps.bucket_start DESC
"""