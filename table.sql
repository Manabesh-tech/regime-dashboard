
uat_db_params = {
    'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
    'port': 5432,
    'database': 'report_dev',  # Different database
    'user': 'public_rw',     # Different user
    'password': 'aTJ92^kl04hllk'  # Different password
}


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
//query  surf buffer rate
select *from trade_pair_risk_history tprh where pair_name='PUMP/USDT' and created_at >='2025-07-24 5:15:00.000 +0800' and created_at <='2025-07-24 14:02:29.011 +0800' ORDER BY created_at DESC;
//query surf volatility history
select *from  trade_pair_volatility_log where pair_name='PUMP/USDT' and created_at >='2025-07-24 5:16:00.000 +0800' and created_at <='2025-07-24 14:02:29.011 +0800' ORDER BY created_at DESC;
//query rollbit buffer rate
select *from rollbit_pair_config where pair_name='1000PUMP/USDT' and created_at >='2025-08-15 01:18:00.000 +0800' and created_at <='2025-07-24 01:20:00.011 +0800' ORDER BY created_at DESC;

