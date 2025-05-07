import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import psycopg2
import pytz
from sklearn.cluster import KMeans

st.set_page_config(page_title="Crypto Breakout Analysis", page_icon="📈", layout="wide")

# --- UI Setup ---
st.title("Crypto Range Breakout Analysis")
st.subheader("Range breakouts by 3-hour blocks, averaged over 7 days")

# DB connection
db_params = {
    'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
    'port': 5432,
    'database': 'replication_report',
    'user': 'public_replication',
    'password': '866^FKC4hllk'
}

conn = psycopg2.connect(**db_params)

# Get available tokens
def fetch_trading_pairs():
    query = """
    SELECT pair_name 
    FROM trade_pool_pairs 
    WHERE status = 1
    ORDER BY pair_name
    """
    
    df = pd.read_sql_query(query, conn)
    return df['pair_name'].tolist()

# Get all tokens
all_tokens = fetch_trading_pairs()

# UI Controls
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # Select tokens
    selected_tokens = st.multiselect(
        "Select up to 5 tokens for analysis", 
        all_tokens,
        default=[]
    )
    
    if len(selected_tokens) > 5:
        st.warning("Please select up to 5 tokens for best performance")
        selected_tokens = selected_tokens[:5]
    
    if not selected_tokens:
        st.info("Please select at least one token to begin analysis")
        # Don't set any default tokens, leave it blank

with col2:
    # Breakout detection method
    breakout_method = st.selectbox(
        "Breakout Detection Method",
        ["Bollinger Bands", "ATR Multiple", "Adaptive Threshold"],
        index=0
    )

with col3:
    # Refresh button
    if st.button("Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()

# Singapore time
sg_tz = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(sg_tz)
st.write(f"Current time (Singapore): {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Get partition tables
def get_partition_tables(start_date, end_date):
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
        
    # Remove timezone
    start_date = start_date.replace(tzinfo=None)
    end_date = end_date.replace(tzinfo=None)
    
    # Generate all dates
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    
    # Table names
    table_names = [f"oracle_price_log_partition_{date}" for date in dates]
    
    # Check which tables exist
    cursor = conn.cursor()
    if table_names:
        table_list_str = "', '".join(table_names)
        cursor.execute(f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('{table_list_str}')
        """)
        
        existing_tables = [row[0] for row in cursor.fetchall()]
    cursor.close()
    
    return existing_tables

# Build query for partition tables
def build_query(tables, token, start_time, end_time):
    if not tables:
        return ""
    
    union_parts = []
    for table in tables:
        # IMPORTANT: Add 8 hours to convert to Singapore time
        query = f"""
        SELECT 
            pair_name,
            created_at + INTERVAL '8 hour' AS timestamp,
            final_price
        FROM 
            public.{table}
        WHERE 
            created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
            AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
            AND source_type = 0
            AND pair_name = '{token}'
        """
        union_parts.append(query)
    
    return " UNION ".join(union_parts) + " ORDER BY timestamp"

# Get price data for the last 7 days
@st.cache_data(ttl=300)  # 5-minute cache
def get_price_data(token, days=7):
    # Time range
    now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
    start_time_sg = now_sg - timedelta(days=days)
    
    # Get relevant partition tables
    start_date = start_time_sg.replace(tzinfo=None)
    end_date = now_sg.replace(tzinfo=None)
    partition_tables = get_partition_tables(start_date, end_date)
    
    if not partition_tables:
        st.error(f"No data tables found for {start_date} to {end_date}")
        return None
    
    # Convert to strings for query
    start_time_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    
    # Build and execute query
    query = build_query(partition_tables, token, start_time_str, end_time_str)
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        st.error(f"No data found for {token}")
        return None
    
    # Process timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # Create 5-minute OHLC data
    ohlc = df['final_price'].resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    return ohlc

# Define breakout detection functions
def detect_breakouts_atr(df, window=14, multiplier=2.0):
    """Detect breakouts using ATR (Average True Range)"""
    # Calculate ATR
    df = df.copy()
    
    # True Range
    df['tr1'] = abs(df['high'] - df['low'])
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Average True Range
    df['atr'] = df['tr'].rolling(window=window).mean()
    
    # Define range as ATR * multiplier
    df['upper_band'] = df['close'].shift().rolling(window=window).mean() + (df['atr'] * multiplier)
    df['lower_band'] = df['close'].shift().rolling(window=window).mean() - (df['atr'] * multiplier)
    
    # Detect breakouts
    df['breakout_up'] = (df['close'] > df['upper_band']) & (df['close'].shift() <= df['upper_band'].shift())
    df['breakout_down'] = (df['close'] < df['lower_band']) & (df['close'].shift() >= df['lower_band'].shift())
    df['breakout'] = df['breakout_up'] | df['breakout_down']
    
    return df

def detect_breakouts_bollinger(df, window=20, std_dev=2.0):
    """Detect breakouts using Bollinger Bands"""
    df = df.copy()
    
    # Calculate Bollinger Bands
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    df['upper_band'] = df['ma'] + (df['std'] * std_dev)
    df['lower_band'] = df['ma'] - (df['std'] * std_dev)
    
    # Detect breakouts
    df['breakout_up'] = (df['close'] > df['upper_band']) & (df['close'].shift() <= df['upper_band'].shift())
    df['breakout_down'] = (df['close'] < df['lower_band']) & (df['close'].shift() >= df['lower_band'].shift())
    df['breakout'] = df['breakout_up'] | df['breakout_down']
    
    return df

def detect_breakouts_adaptive(df, short_window=12, long_window=48):
    """Detect breakouts using adaptive thresholds based on recent volatility patterns"""
    df = df.copy()
    
    # Calculate percentage changes
    df['pct_change'] = df['close'].pct_change()
    
    # Calculate short-term and long-term volatility
    df['short_vol'] = df['pct_change'].rolling(window=short_window).std()
    df['long_vol'] = df['pct_change'].rolling(window=long_window).std()
    
    # Calculate the ratio of short-term to long-term volatility
    df['vol_ratio'] = df['short_vol'] / df['long_vol']
    
    # Use recent data points to determine if we're in a range or trend
    df['vol_ratio_smooth'] = df['vol_ratio'].rolling(window=6).mean()
    
    # Adaptive thresholds based on recent market conditions
    # For ranging markets (low vol_ratio), use tighter bands
    # For trending markets (high vol_ratio), use wider bands
    
    # Base threshold is the 20-period standard deviation
    df['base_threshold'] = df['pct_change'].rolling(window=20).std()
    
    # Adjust threshold based on vol_ratio
    # If vol_ratio < 0.8, we're likely in a range, use tight bands (1.5x std)
    # If vol_ratio > 1.2, we're likely in a trend, use wider bands (2.5x std)
    df['threshold_multiplier'] = 1.5 + (df['vol_ratio_smooth'] - 0.8) * (2.5 - 1.5) / (1.2 - 0.8)
    df['threshold_multiplier'] = df['threshold_multiplier'].clip(1.5, 2.5)
    
    df['adaptive_threshold'] = df['base_threshold'] * df['threshold_multiplier']
    
    # Calculate the adaptive bands
    df['ma'] = df['close'].rolling(window=20).mean()
    df['upper_band'] = df['ma'] + (df['adaptive_threshold'] * df['ma'])
    df['lower_band'] = df['ma'] - (df['adaptive_threshold'] * df['ma'])
    
    # Detect breakouts
    df['breakout_up'] = (df['close'] > df['upper_band']) & (df['close'].shift() <= df['upper_band'].shift())
    df['breakout_down'] = (df['close'] < df['lower_band']) & (df['close'].shift() >= df['lower_band'].shift())
    df['breakout'] = df['breakout_up'] | df['breakout_down']
    
    # Try to identify if the breakout is significant by checking follow-through
    # A real breakout should continue in the direction of the breakout
    df['significant_breakout'] = False
    
    # For each breakout, check if the next few candles continue in the same direction
    for i in range(len(df)):
        if df.iloc[i]['breakout']:
            # Check if this is an upward breakout
            if df.iloc[i]['breakout_up']:
                # Check the next 3 candles if available
                follow_through = 0
                for j in range(1, min(4, len(df) - i)):
                    if df.iloc[i+j]['close'] > df.iloc[i]['close']:
                        follow_through += 1
                
                # If at least 2 of the next 3 candles continue higher, it's a significant breakout
                if follow_through >= 2:
                    df.iloc[i, df.columns.get_loc('significant_breakout')] = True
            
            # Check if this is a downward breakout
            elif df.iloc[i]['breakout_down']:
                # Check the next 3 candles if available
                follow_through = 0
                for j in range(1, min(4, len(df) - i)):
                    if df.iloc[i+j]['close'] < df.iloc[i]['close']:
                        follow_through += 1
                
                # If at least 2 of the next 3 candles continue lower, it's a significant breakout
                if follow_through >= 2:
                    df.iloc[i, df.columns.get_loc('significant_breakout')] = True
    
    return df

# Process data and detect breakouts
def process_token_data(token):
    # Get price data
    ohlc_data = get_price_data(token)
    
    if ohlc_data is None or ohlc_data.empty:
        return None
    
    # Detect breakouts based on selected method
    if breakout_method == "ATR Multiple":
        breakout_df = detect_breakouts_atr(ohlc_data)
        breakout_col = 'breakout'
    elif breakout_method == "Bollinger Bands":
        breakout_df = detect_breakouts_bollinger(ohlc_data)
        breakout_col = 'breakout'
    else:  # Adaptive Threshold
        breakout_df = detect_breakouts_adaptive(ohlc_data)
        breakout_col = 'significant_breakout'  # Use significant breakouts for adaptive method
    
    # Create 3-hour blocks
    breakout_df['hour'] = breakout_df.index.hour
    breakout_df['block'] = (breakout_df['hour'] // 3) * 3  # 0, 3, 6, 9, 12, 15, 18, 21
    breakout_df['day'] = breakout_df.index.date
    
    # Count breakouts by block and day
    breakout_counts = breakout_df.groupby(['day', 'block'])[breakout_col].sum().reset_index()
    
    # Convert counts to a pivot table: days as rows, blocks as columns
    pivot_counts = breakout_counts.pivot(index='day', columns='block', values=breakout_col).fillna(0)
    
    # Get the average by block over the last 7 days
    avg_by_block = pivot_counts.mean()
    
    return {
        'token': token,
        'breakout_df': breakout_df,
        'avg_by_block': avg_by_block,
        'breakout_col': breakout_col
    }

# Process data for all selected tokens
results = {}
all_blocks_avg = pd.DataFrame()

with st.spinner("Processing data for selected tokens..."):
    for token in selected_tokens:
        result = process_token_data(token)
        if result:
            results[token] = result
            
            # Add to all blocks average dataframe
            block_avg = result['avg_by_block']
            if not block_avg.empty:
                all_blocks_avg[token] = block_avg

# Display results
if results and not all_blocks_avg.empty:
    # Make sure all blocks are represented (0, 3, 6, 9, 12, 15, 18, 21)
    for block in [0, 3, 6, 9, 12, 15, 18, 21]:
        if block not in all_blocks_avg.index:
            all_blocks_avg.loc[block] = 0
    
    all_blocks_avg = all_blocks_avg.sort_index()
    
    # Create block labels
    block_labels = {
        0: "00:00-03:00",
        3: "03:00-06:00",
        6: "06:00-09:00",
        9: "09:00-12:00",
        12: "12:00-15:00",
        15: "15:00-18:00",
        18: "18:00-21:00",
        21: "21:00-00:00"
    }
    
    # Display block averages
    st.subheader(f"All Pairs: Average Number of Breakouts per 3-Hour Block (Past 7 Days)")
    
    # Create a transposed view with better labels
    display_df = all_blocks_avg.copy()
    display_df.index = [block_labels[idx] for idx in display_df.index]
    
    # Calculate total breakouts and identify peak times
    display_df['Total'] = display_df.sum(axis=1)
    display_df = display_df.sort_values('Total', ascending=False)
    
    # Display the table
    st.dataframe(display_df)
    
    # Create visualization
    fig = px.bar(
        all_blocks_avg, 
        x=all_blocks_avg.index.map(lambda x: block_labels[x]),
        y=all_blocks_avg.columns,
        title="All Pairs: Average Breakouts by 3-Hour Block (Past 7 Days)",
        labels={'index': 'Time Block (Singapore)', 'value': 'Avg. Number of Breakouts'}
    )
    
    fig.update_layout(
        height=500,
        xaxis_title="Time Block (Singapore)",
        yaxis_title="Average Number of Breakouts",
        legend_title="Token",
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show highest breakout periods overall
    highest_overall = all_blocks_avg.mean(axis=1).sort_values(ascending=False)
    highest_block = highest_overall.index[0]
    lowest_block = highest_overall.index[-1]
    
    st.subheader("Range vs. Breakout Trading Patterns")
    
    st.markdown(f"""
    ### Key Findings:
    
    - **Peak Breakout Period:** {block_labels[highest_block]} (Avg: {highest_overall.iloc[0]:.2f} breakouts)
    - **Lowest Breakout Period:** {block_labels[lowest_block]} (Avg: {highest_overall.iloc[-1]:.2f} breakouts)
    """)
    
    # Trader type analysis
    st.markdown("""
    ### Trader Types by Time Block:
    """)
    
    blocks_sorted = highest_overall.sort_values(ascending=False)
    
    for block, value in blocks_sorted.items():
        if value > highest_overall.mean():
            st.markdown(f"- **{block_labels[block]}**: More breakout traders (higher than average breakouts)")
        else:
            st.markdown(f"- **{block_labels[block]}**: More range traders (lower than average breakouts)")
    
    # Show detailed breakout charts for individual tokens
    st.subheader("Individual Token Analysis")
    
    for token in selected_tokens:
        if token in results:
            result = results[token]
            breakout_df = result['breakout_df']
            avg_by_block = result['avg_by_block']
            breakout_col = result['breakout_col']
            
            st.markdown(f"### {token}")
            
            # Show recent breakouts
            recent_breakouts = breakout_df[breakout_df[breakout_col]].tail(10)
            if not recent_breakouts.empty:
                st.markdown("#### Recent Breakouts")
                recent_display = recent_breakouts[['open', 'high', 'low', 'close']].copy()
                recent_display.index = recent_display.index.strftime('%Y-%m-%d %H:%M')
                st.dataframe(recent_display)
            
            # Create a candlestick chart with breakouts highlighted
            recent_data = breakout_df.tail(288)  # Show last day of data (288 5-min candles)
            
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=recent_data.index,
                    open=recent_data['open'],
                    high=recent_data['high'],
                    low=recent_data['low'],
                    close=recent_data['close'],
                    name="Price"
                )
            )
            
            # Add upper and lower bands
            fig.add_trace(
                go.Scatter(
                    x=recent_data.index,
                    y=recent_data['upper_band'],
                    name="Upper Band",
                    line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dot')
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=recent_data.index,
                    y=recent_data['lower_band'],
                    name="Lower Band",
                    line=dict(color='rgba(0, 0, 255, 0.5)', width=1, dash='dot')
                )
            )
            
            # Highlight breakout points
            breakout_points = recent_data[recent_data[breakout_col]]
            
            # Upward breakouts
            up_breakouts = breakout_points[breakout_points.get('breakout_up', False)]
            if not up_breakouts.empty and 'breakout_up' in breakout_points.columns:
                fig.add_trace(
                    go.Scatter(
                        x=up_breakouts.index,
                        y=up_breakouts['high'],
                        mode='markers',
                        marker=dict(color='green', size=10, symbol='triangle-up'),
                        name="Upward Breakout"
                    )
                )
            
            # Downward breakouts
            down_breakouts = breakout_points[breakout_points.get('breakout_down', False)]
            if not down_breakouts.empty and 'breakout_down' in breakout_points.columns:
                fig.add_trace(
                    go.Scatter(
                        x=down_breakouts.index,
                        y=down_breakouts['low'],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='triangle-down'),
                        name="Downward Breakout"
                    )
                )
            
            fig.update_layout(
                title=f"{token} Recent Price Action with Breakouts",
                xaxis_title="Time",
                yaxis_title="Price",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display breakout distribution by block
            avg_by_block_display = avg_by_block.copy()
            avg_by_block_display.index = [block_labels[idx] for idx in avg_by_block_display.index]
            
            block_fig = px.bar(
                x=avg_by_block_display.index,
                y=avg_by_block_display.values,
                title=f"{token} - Avg. Breakouts by 3-Hour Block",
                labels={'x': 'Time Block (Singapore)', 'y': 'Avg. Number of Breakouts'}
            )
            
            st.plotly_chart(block_fig, use_container_width=True)
            
            st.markdown("---")
    
    # Additional analysis - clustering time blocks
    st.subheader("Time Block Clustering Analysis")
    
    # Prepare data for clustering
    cluster_data = all_blocks_avg.T
    
    if len(cluster_data) >= 3:  # Need at least 3 tokens for meaningful clustering
        # Normalize data for clustering
        normalized_data = (cluster_data - cluster_data.mean()) / cluster_data.std()
        
        # K-means clustering to find similar time blocks
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster_data['cluster'] = kmeans.fit_predict(normalized_data)
        
        # Display clusters
        st.markdown("### Time Block Clusters")
        st.markdown("Grouping time blocks into clusters based on breakout patterns:")
        
        # Get cluster centers and interpret
        centers = kmeans.cluster_centers_
        
        # Determine which cluster has higher average breakouts
        cluster_0_avg = centers[0].mean()
        cluster_1_avg = centers[1].mean()
        
        if cluster_0_avg > cluster_1_avg:
            breakout_cluster = 0
            range_cluster = 1
        else:
            breakout_cluster = 1
            range_cluster = 0
        
        # Display results
        breakout_times = [block_labels[block] for block, cluster in 
                         zip(all_blocks_avg.index, kmeans.labels_) if cluster == breakout_cluster]
        
        range_times = [block_labels[block] for block, cluster in 
                      zip(all_blocks_avg.index, kmeans.labels_) if cluster == range_cluster]
        
        st.markdown(f"""
        #### Breakout Trading Times:
        Times with significantly more breakouts, suggesting more breakout traders active:
        {", ".join(breakout_times)}
        
        #### Range Trading Times:
        Times with fewer breakouts, suggesting more range traders active:
        {", ".join(range_times)}
        """)
        
        # Visualize the clusters
        cluster_viz = pd.DataFrame({
            'Time Block': [block_labels[i] for i in all_blocks_avg.index],
            'Cluster': kmeans.labels_,
            'Average Breakouts': all_blocks_avg.mean(axis=1).values
        })
        
        cluster_viz['Trading Style'] = cluster_viz['Cluster'].apply(
            lambda x: 'Breakout Trading' if x == breakout_cluster else 'Range Trading'
        )
        
        cluster_fig = px.scatter(
            cluster_viz,
            x='Time Block',
            y='Average Breakouts',
            color='Trading Style',
            size='Average Breakouts',
            title="Time Blocks Clustered by Trading Style",
            labels={'Average Breakouts': 'Avg. Number of Breakouts'}
        )
        
        cluster_fig.update_layout(height=500)
        st.plotly_chart(cluster_fig, use_container_width=True)
        
        # Trading recommendation
        st.subheader("Exchange Strategy Recommendations")
        
        st.markdown("""
        Based on the analysis of breakout patterns across different time blocks, here are some strategic recommendations:
        
        1. **Liquidity Management:**
           - Increase liquidity during peak breakout times to handle higher volumes
           - Optimize spread during range-bound periods
           
        2. **Risk Management:**
           - Adjust risk parameters by time of day
           - Monitor liquidation risks more closely during breakout periods
           
        3. **User Experience:**
           - Provide different trading tools based on time of day
           - Show relevant indicators (breakout vs. range) based on current market phase
           
        4. **Marketing and User Acquisition:**
           - Target different trader types in campaigns based on their active hours
           - Create educational content specific to both trading styles
        """)
else:
    st.error("No valid data available for the selected tokens. Please try different tokens or adjust the date range.")