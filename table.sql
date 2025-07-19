//query surf all trade  pairs
select *from trade_pool_pairs where status in (1,2);

//query surf price data
select *from oracle_price_log_partition_20250526 where pair_name ='ETH/USDT' and source_type =0;

//query rollbit price data
select *from oracle_price_log_partition_20250526 where pair_name ='ETH/USDT' and source_type =1;


//query rollbit pair config 
select *from rollbit_pair_config rpc ;

//query surf all_bid,all_ask 
select all_bid,all_ask from oracle_order_book_level_price_data_partition_v5_20250526 where pair_name='LPT';

//query surf exchange all_bid,all_ask 
select all_bid,all_ask  from oracle_exchange_price_partition_v1_20250526 where pair_name='LPT';


//query exchange_fee
select * from oracle_exchange_fee;



uat_db_params = {
    'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
    'port': 5432,
    'database': 'report_dev',  # Different database
    'user': 'public_rw',     # Different user
    'password': 'aTJ92^kl04hllk'  # Different password
}

//query rollbit_pair_config
SELECT bust_buffer, created_at
FROM rollbit_pair_config
WHERE pair_name='1000PUMP/USDT' and created_at >= '2025-07-18 19:34:00'
  AND created_at <= '2025-07-18 19:36:00'
ORDER BY created_at DESC;
//query surf buffer_rate config
SELECT buffer_rate,created_at
FROM trade_pair_risk_history
WHERE pair_name='PUMP/USDT' and created_at >= '2025-07-18 19:34:00'
  AND created_at <= '2025-07-18 19:36:00'
ORDER BY created_at DESC;