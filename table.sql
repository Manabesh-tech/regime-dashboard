//query surf all trade  pairs
select *from trade_pool_pairs where status in (1,2);

//query surf price data
select *from oracle_price_log_partition_20250526 where pair_name ='ETH/USDT' and source_type =0;

//query rollbit price data
select *from oracle_price_log_partition_20250526 where pair_name ='ETH/USDT' and source_type =1;


//query rollbit pair config 
select *from rollbit_pair_config rpc ;

//query surf all_bid,all_ask 
select all_bid,all_ask from oracle_order_book_level_price_data_partition_v5_20250526;

//query surf exchange all_bid,all_ask 
select all_bid,all_ask  from oracle_exchange_price_partition_v1_20250526;


//query exchange_fee
select * from oracle_exchange_fee;
