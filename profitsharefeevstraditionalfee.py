@st.cache_data(ttl=300)
def fetch_streamlit_data():
    """Fetch profit share parameters from database"""
    try:
        # Get database connection details from secrets
        metabase_config = st.secrets.get("metabase", {})
        
        if not metabase_config:
            st.warning("Metabase configuration not found in secrets. Using mock data.")
            return get_mock_profit_share_data()
        
        # Direct database connection
        if "host" in metabase_config:
            import psycopg2
            import sqlalchemy
            
            # Create connection string
            connection_string = f"postgresql://{metabase_config['user']}:{metabase_config['password']}@{metabase_config['host']}:{metabase_config['port']}/{metabase_config['database']}"
            
            engine = sqlalchemy.create_engine(connection_string)
            
            # First get the traditional spreads pair names to match against
            spreads_query = """
            SELECT DISTINCT pair_name
            FROM oracle_exchange_spread 
            WHERE source = 'binanceFuture'
                AND time_group = (SELECT MAX(time_group) FROM oracle_exchange_spread)
            ORDER BY pair_name
            """
            spreads_pairs = pd.read_sql(spreads_query, engine)['pair_name'].tolist()
            
            # Now get profit share parameters ONLY for pairs that exist in traditional spreads
            query = """
            SELECT 
                pair_name as "Pair",
                pnl_base_rate as "Base Rate",
                rate_multiplier as "Rate Multiplier", 
                rate_exponent as "Rate Exponent",
                position_multiplier as "Position Multiplier",
                1 as "Bet Multiplier",
                funding_fee as "Buffer Rate"
            FROM trade_pool_pairs
            WHERE created_at = (SELECT MAX(created_at) FROM trade_pool_pairs)
                AND pair_name IN ({})
            ORDER BY pair_name;
            """.format(','.join([f"'{pair}'" for pair in spreads_pairs]))
            
            df = pd.read_sql(query, engine)
            
            # Convert columns to proper data types
            df["Base Rate"] = df["Base Rate"] / 100.0  # Convert to decimal (e.g., 0.1 -> 0.001)
            df["Rate Multiplier"] = df["Rate Multiplier"].astype(int)
            df["Rate Exponent"] = df["Rate Exponent"].astype(int) 
            df["Position Multiplier"] = df["Position Multiplier"].astype(int)
            df["Bet Multiplier"] = df["Bet Multiplier"].astype(int)
            df["Buffer Rate"] = df["Buffer Rate"].astype(float)
            
            st.success(f"✅ Successfully loaded {len(df)} matching pairs from database")
            if len(df) > 0:
                st.info(f"📊 Matched pairs: {', '.join(df['Pair'].tolist())}")
            else:
                st.warning("⚠️ No matching pairs found between traditional spreads and profit-share parameters")
            
            return df
        
    except Exception as e:
        st.error(f"❌ Error fetching profit share data from database: {str(e)}")
        st.info("🔄 Falling back to mock data")
        return get_mock_profit_share_data()
    
    return get_mock_profit_share_data()import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Profit-Sharing vs Traditional Spread Comparison", 
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Profit-Sharing vs Traditional Spread Comparison")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Price moves to test
price_moves = [0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0]
volume_levels = ['1k', '5k', '10k', '20k']
volume_amounts = {'1k': 1000, '5k': 5000, '10k': 10000, '20k': 20000}

# Functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_metabase_data():
    """Fetch traditional spreads from Metabase"""
    try:
        # Get database connection details from secrets
        metabase_config = st.secrets.get("metabase", {})
        
        if not metabase_config:
            st.warning("Metabase configuration not found in secrets. Using mock data.")
            return get_mock_traditional_data()
        
        # Direct database connection
        if "host" in metabase_config:
            import psycopg2
            import sqlalchemy
            
            # Create connection string
            connection_string = f"postgresql://{metabase_config['user']}:{metabase_config['password']}@{metabase_config['host']}:{metabase_config['port']}/{metabase_config['database']}"
            
            engine = sqlalchemy.create_engine(connection_string)
            
            # Exact Metabase query for small volume fees
            query = """
            -- Spreads table with columns: Pair name, volume 1k, volume 5k, volume 10k, volume 20k
            SELECT 
                pair_name,
                MAX(CASE WHEN amount::numeric = 1000 THEN fee::numeric * 10000 END) AS "1k",
                MAX(CASE WHEN amount::numeric = 5000 THEN fee::numeric * 10000 END) AS "5k",
                MAX(CASE WHEN amount::numeric = 10000 THEN fee::numeric * 10000 END) AS "10k",
                MAX(CASE WHEN amount::numeric = 20000 THEN fee::numeric * 10000 END) AS "20k"
            FROM oracle_exchange_spread 
            WHERE source = 'binanceFuture'
                AND time_group = (SELECT MAX(time_group) FROM oracle_exchange_spread)
                AND amount::numeric IN (1000, 5000, 10000, 20000)
            GROUP BY pair_name
            ORDER BY pair_name;
            """
            
            df = pd.read_sql(query, engine)
            st.success(f"✅ Successfully loaded {len(df)} pairs from database (latest time_group)")
            return df
        
        # Fallback to API if database connection fails
        if "api_url" in metabase_config:
            headers = {
                "Authorization": f"Bearer {metabase_config.get('api_key', '')}",
                "Content-Type": "application/json"
            }
            response = requests.get(metabase_config["api_url"], headers=headers)
            if response.status_code == 200:
                return pd.DataFrame(response.json())
        
    except Exception as e:
        st.error(f"❌ Error fetching Metabase data: {str(e)}")
        st.info("🔄 Falling back to mock data")
        return get_mock_traditional_data()
    
    return get_mock_traditional_data()

@st.cache_data(ttl=300)
def fetch_streamlit_data():
    """Fetch profit share parameters from database"""
    try:
        # Get database connection details from secrets
        metabase_config = st.secrets.get("metabase", {})
        
        if not metabase_config:
            st.warning("Metabase configuration not found in secrets. Using mock data.")
            return get_mock_profit_share_data()
        
        # Direct database connection
        if "host" in metabase_config:
            import psycopg2
            import sqlalchemy
            
            # Create connection string
            connection_string = f"postgresql://{metabase_config['user']}:{metabase_config['password']}@{metabase_config['host']}:{metabase_config['port']}/{metabase_config['database']}"
            
            engine = sqlalchemy.create_engine(connection_string)
            
            # First get the traditional spreads pair names to match against
            spreads_query = """
            SELECT DISTINCT pair_name
            FROM oracle_exchange_spread 
            WHERE source = 'binanceFuture'
                AND time_group = (SELECT MAX(time_group) FROM oracle_exchange_spread)
            """
            spreads_pairs = pd.read_sql(spreads_query, engine)['pair_name'].tolist()
            
            # Now get profit share parameters with matching pair names
            # Join trade_pool_pairs with the pairs table to get pair names
            query = """
            SELECT 
                p.pair_name as "Pair",
                tpp.pnl_base_rate / 100.0 as "Base Rate",
                tpp.rate_multiplier as "Rate Multiplier", 
                tpp.rate_exponent as "Rate Exponent",
                tpp.position_multiplier as "Position Multiplier",
                1 as "Bet Multiplier",
                tpp.funding_fee as "Buffer Rate"
            FROM trade_pool_pairs tpp
            JOIN pairs p ON tpp.pair_id = p.pair_id
            WHERE tpp.created_at = (SELECT MAX(created_at) FROM trade_pool_pairs)
                AND p.pair_name IN ({})
            ORDER BY p.pair_name;
            """.format(','.join([f"'{pair}'" for pair in spreads_pairs]))
            
            df = pd.read_sql(query, engine)
            
            # Convert columns to proper data types
            df["Rate Multiplier"] = df["Rate Multiplier"].astype(int)
            df["Rate Exponent"] = df["Rate Exponent"].astype(int) 
            df["Position Multiplier"] = df["Position Multiplier"].astype(int)
            df["Bet Multiplier"] = df["Bet Multiplier"].astype(int)
            
            st.success(f"✅ Successfully loaded {len(df)} matching pairs from database")
            st.info(f"📊 Matched pairs: {', '.join(df['Pair'].tolist())}")
            
            return df
        
    except Exception as e:
        st.error(f"❌ Error fetching profit share data from database: {str(e)}")
        st.info("🔄 Falling back to mock data")
        return get_mock_profit_share_data()
    
    return get_mock_profit_share_data()

def get_mock_traditional_data():
    """Mock traditional spread data based on screenshots"""
    return pd.DataFrame([
        {'pair_name': '1000BONK/USDT', '1k': 1.01, '5k': 2.31, '10k': 3.11, '20k': 4.53},
        {'pair_name': 'AAVE/USDT', '1k': 0.34, '5k': 0.57, '10k': 0.91, '20k': 1.48},
        {'pair_name': 'ADA/USDT', '1k': 1.69, '5k': 1.69, '10k': 1.89, '20k': 1.97},
        {'pair_name': 'AI16Z/USDT', '1k': 6.52, '5k': 7.83, '10k': 10.13, '20k': 14.24},
        {'pair_name': 'ARB/USDT', '1k': 2.96, '5k': 3.08, '10k': 3.7, '20k': 4.75},
        {'pair_name': 'AVAX/USDT', '1k': 0.63, '5k': 0.75, '10k': 1.01, '20k': 1.47},
        {'pair_name': 'BNB/USDT', '1k': 0.15, '5k': 0.15, '10k': 0.17, '20k': 0.23},
        {'pair_name': 'BTC/USDT', '1k': 0.0092, '5k': 0.0092, '10k': 0.0092, '20k': 0.0092},
        {'pair_name': 'DOGE/USDT', '1k': 0.58, '5k': 0.65, '10k': 0.66, '20k': 0.8},
        {'pair_name': 'ENA/USDT', '1k': 3.78, '5k': 3.78, '10k': 4.13, '20k': 4.7},
        {'pair_name': 'ETH/USDT', '1k': 0.038, '5k': 0.045, '10k': 0.038, '20k': 0.038},
        {'pair_name': 'FARTCOIN/USDT', '1k': 1.04, '5k': 1.93, '10k': 2.8, '20k': 4.31},
        {'pair_name': 'HYPE/USDT', '1k': 0.35, '5k': 1.36, '10k': 2.11, '20k': 3.45}
    ])

def get_mock_profit_share_data():
    """Mock profit share parameters based on screenshots"""
    return pd.DataFrame([
        {'Pair': '1000BONK/USDT', 'Position Multiplier': 2, 'Buffer Rate': 0.049, 'Rate Multiplier': 1000, 'Rate Exponent': 1, 'Base Rate': 0.1, 'Bet Multiplier': 1},
        {'Pair': 'AAVE/USDT', 'Position Multiplier': 10, 'Buffer Rate': 0.040, 'Rate Multiplier': 1000, 'Rate Exponent': 1, 'Base Rate': 0.1, 'Bet Multiplier': 1},
        {'Pair': 'ADA/USDT', 'Position Multiplier': 80, 'Buffer Rate': 0.040, 'Rate Multiplier': 1000, 'Rate Exponent': 1, 'Base Rate': 0.1, 'Bet Multiplier': 1},
        {'Pair': 'AI16Z/USDT', 'Position Multiplier': 1, 'Buffer Rate': 0.056, 'Rate Multiplier': 1000, 'Rate Exponent': 1, 'Base Rate': 0.1, 'Bet Multiplier': 1},
        {'Pair': 'ARB/USDT', 'Position Multiplier': 5, 'Buffer Rate': 0.040, 'Rate Multiplier': 1000, 'Rate Exponent': 1, 'Base Rate': 0.1, 'Bet Multiplier': 1},
        {'Pair': 'AVAX/USDT', 'Position Multiplier': 5, 'Buffer Rate': 0.038, 'Rate Multiplier': 1000, 'Rate Exponent': 1, 'Base Rate': 0.1, 'Bet Multiplier': 1},
        {'Pair': 'BNB/USDT', 'Position Multiplier': 80, 'Buffer Rate': 0.036, 'Rate Multiplier': 1000, 'Rate Exponent': 1, 'Base Rate': 0.1, 'Bet Multiplier': 1},
        {'Pair': 'BTC/USDT', 'Position Multiplier': 500, 'Buffer Rate': 0.035, 'Rate Multiplier': 3000, 'Rate Exponent': 1, 'Base Rate': 0.1, 'Bet Multiplier': 1},
        {'Pair': 'DOGE/USDT', 'Position Multiplier': 80, 'Buffer Rate': 0.037, 'Rate Multiplier': 1000, 'Rate Exponent': 1, 'Base Rate': 0.1, 'Bet Multiplier': 1},
        {'Pair': 'ENA/USDT', 'Position Multiplier': 5, 'Buffer Rate': 0.050, 'Rate Multiplier': 1000, 'Rate Exponent': 1, 'Base Rate': 0.1, 'Bet Multiplier': 1},
        {'Pair': 'ETH/USDT', 'Position Multiplier': 100, 'Buffer Rate': 0.035, 'Rate Multiplier': 3000, 'Rate Exponent': 1, 'Base Rate': 0.1, 'Bet Multiplier': 1},
        {'Pair': 'FARTCOIN/USDT', 'Position Multiplier': 1, 'Buffer Rate': 0.040, 'Rate Multiplier': 1000, 'Rate Exponent': 1, 'Base Rate': 0.1, 'Bet Multiplier': 1},
        {'Pair': 'HYPE/USDT', 'Position Multiplier': 1, 'Buffer Rate': 0.040, 'Rate Multiplier': 1000, 'Rate Exponent': 1, 'Base Rate': 0.1, 'Bet Multiplier': 1}
    ])

def calculate_traditional_fee(basis_points, position_size):
    """Calculate traditional spread fee in dollars"""
    return (basis_points / 10000) * position_size

def calculate_profit_sharing_fee(open_price, close_price, bet_amount, params):
    """Calculate profit sharing fee using the exact formula"""
    base_rate = params['base_rate']
    rate_multiplier = params['rate_multiplier']
    rate_exponent = params['rate_exponent']
    position_multiplier = params['position_multiplier']
    bet_multiplier = params['bet_multiplier']
    
    P_t = open_price
    P_T = close_price
    
    price_move_pct = (P_T / P_t) - 1
    abs_price_move_pct = abs(price_move_pct)
    
    if abs_price_move_pct == 0:
        return 0
    
    # Calculate exponential term
    exponential_term = 1 / (abs_price_move_pct * rate_multiplier) ** rate_exponent
    
    # Calculate position term
    position_term = (bet_amount * bet_multiplier) / (10**6 * abs_price_move_pct * position_multiplier)
    
    # Calculate profit sharing fraction
    profit_share_fraction = (1 - base_rate) / (1 + exponential_term + position_term)
    
    # Calculate P_close
    P_close = P_t + profit_share_fraction * (P_T - P_t)
    
    # Calculate fee
    fee_charged_points = (P_T - P_close)
    fee_charged_dollars = (fee_charged_points * bet_amount * bet_multiplier) / 100
    
    return fee_charged_dollars

def create_comparison_table(traditional_data, profit_share_data):
    """Create comprehensive comparison table"""
    # Merge datasets
    merged_data = traditional_data.merge(
        profit_share_data, 
        left_on='pair_name', 
        right_on='Pair', 
        how='inner'
    )
    
    if merged_data.empty:
        st.error("No matching pairs found between traditional and profit share data!")
        return pd.DataFrame()
    
    results = []
    
    for _, row in merged_data.iterrows():
        pair_name = row['pair_name']
        
        # Profit share parameters
        ps_params = {
            'base_rate': row.get('Base Rate', 0.1),
            'rate_multiplier': row.get('Rate Multiplier', 1000),
            'rate_exponent': row.get('Rate Exponent', 1),
            'position_multiplier': row.get('Position Multiplier', 200),
            'bet_multiplier': row.get('Bet Multiplier', 1)
        }
        
        for volume in volume_levels:
            position_size = volume_amounts[volume]
            traditional_spread = row[volume]
            traditional_fee = calculate_traditional_fee(traditional_spread, position_size)
            bet_amount = position_size / ps_params['bet_multiplier']
            
            for price_move in price_moves:
                close_price = 100 * (1 + price_move / 100)
                profit_share_fee = calculate_profit_sharing_fee(100, close_price, bet_amount, ps_params)
                
                difference = profit_share_fee - traditional_fee
                is_ps_more_expensive = difference > 0
                
                results.append({
                    'Pair': pair_name,
                    'Volume': volume,
                    'Position Size': position_size,
                    'Price Move (%)': price_move,
                    'Traditional Fee ($)': traditional_fee,
                    'Profit Share Fee ($)': profit_share_fee,
                    'Difference ($)': difference,
                    'PS More Expensive': is_ps_more_expensive,
                    'Traditional Spread (bp)': traditional_spread
                })
    
    return pd.DataFrame(results)

# Main app
def main():
    # Sidebar options
    refresh_data = st.sidebar.button("🔄 Refresh Data")
    show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)
    
    # Load data
    if refresh_data or 'traditional_data' not in st.session_state:
        with st.spinner("Loading data..."):
            st.session_state.traditional_data = fetch_metabase_data()
            st.session_state.profit_share_data = fetch_streamlit_data()
    
    traditional_data = st.session_state.traditional_data
    profit_share_data = st.session_state.profit_share_data
    
    # Data status
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Traditional Spreads", f"{len(traditional_data)} pairs")
    with col2:
        st.metric("Profit Share Parameters", f"{len(profit_share_data)} pairs")
    
    # Show raw data if requested
    if show_raw_data:
        st.subheader("Raw Data")
        tab1, tab2 = st.tabs(["Traditional Spreads", "Profit Share Parameters"])
        
        with tab1:
            st.dataframe(traditional_data)
        
        with tab2:
            st.dataframe(profit_share_data)
    
    # Create comparison
    st.subheader("📊 Comparison Analysis")
    
    with st.spinner("Calculating comparisons..."):
        comparison_df = create_comparison_table(traditional_data, profit_share_data)
    
    if comparison_df.empty:
        st.stop()
    
    # Filter options
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_pairs = st.multiselect(
            "Select Pairs", 
            options=comparison_df['Pair'].unique(),
            default=comparison_df['Pair'].unique()[:5]  # Default to first 5
        )
    
    with col2:
        selected_volumes = st.multiselect(
            "Select Volumes",
            options=volume_levels,
            default=volume_levels
        )
    
    with col3:
        selected_moves = st.multiselect(
            "Select Price Moves (%)",
            options=price_moves,
            default=price_moves
        )
    
    # Filter data
    filtered_df = comparison_df[
        (comparison_df['Pair'].isin(selected_pairs)) &
        (comparison_df['Volume'].isin(selected_volumes)) &
        (comparison_df['Price Move (%)'].isin(selected_moves))
    ]
    
    # Summary metrics
    st.subheader("📈 Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_trad_fee = filtered_df['Traditional Fee ($)'].mean()
        st.metric("Avg Traditional Fee", f"${avg_trad_fee:.3f}")
    
    with col2:
        avg_ps_fee = filtered_df['Profit Share Fee ($)'].mean()
        st.metric("Avg Profit Share Fee", f"${avg_ps_fee:.3f}")
    
    with col3:
        ps_more_expensive_pct = (filtered_df['PS More Expensive'].sum() / len(filtered_df)) * 100
        st.metric("PS More Expensive", f"{ps_more_expensive_pct:.1f}%")
    
    with col4:
        avg_diff = filtered_df['Difference ($)'].mean()
        st.metric("Avg Difference", f"${avg_diff:.3f}")
    
    # Detailed comparison table
    st.subheader("📋 Detailed Comparison")
    
    # Format the dataframe for display
    display_df = filtered_df.copy()
    display_df['Traditional Fee ($)'] = display_df['Traditional Fee ($)'].apply(lambda x: f"${x:.3f}")
    display_df['Profit Share Fee ($)'] = display_df['Profit Share Fee ($)'].apply(lambda x: f"${x:.3f}")
    display_df['Difference ($)'] = display_df['Difference ($)'].apply(lambda x: f"${x:.3f}")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Visualizations
    st.subheader("📊 Visualizations")
    
    # Chart 1: Fee comparison by price move with traditional fee overlay
    fig1 = px.line(
        filtered_df,
        x='Price Move (%)',
        y='Profit Share Fee ($)',
        color='Pair',
        facet_col='Volume',
        title="Profit Share Fees vs Traditional Fees by Price Move"
    )
    
    # Add traditional fee as horizontal lines for each pair and volume
    for pair in selected_pairs:
        for i, volume in enumerate(['1k', '5k', '10k', '20k']):
            if volume in selected_volumes:
                # Get traditional fee for this pair and volume
                pair_data = filtered_df[(filtered_df['Pair'] == pair) & (filtered_df['Volume'] == volume)]
                if not pair_data.empty:
                    traditional_fee = pair_data['Traditional Fee ($)'].iloc[0]
                    
                    # Add horizontal line for traditional fee
                    fig1.add_hline(
                        y=traditional_fee,
                        line_dash="solid",
                        line_color="black",
                        line_width=2,
                        opacity=0.8,
                        col=i+1,  # Column index for facet
                        annotation_text=f"{pair} Traditional: ${traditional_fee:.3f}",
                        annotation_position="top right",
                        annotation_font_size=8
                    )
    
    # Update layout for better visibility
    fig1.update_layout(
        showlegend=True,
        height=600,
        title_text="Profit Share Fees vs Traditional Fees by Price Move<br><sub>Black lines show traditional fees (constant regardless of price move)</sub>"
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Chart 2: Difference heatmap
    if len(selected_pairs) <= 10:  # Only show heatmap for reasonable number of pairs
        pivot_df = filtered_df.pivot_table(
            index='Pair',
            columns='Price Move (%)',
            values='Difference ($)',
            aggfunc='mean'
        )
        
        fig2 = px.imshow(
            pivot_df,
            aspect="auto",
            title="Fee Difference Heatmap (Profit Share - Traditional)",
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Export options
    st.subheader("💾 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"comparison_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
    
    with col2:
        # Summary statistics
        summary_stats = filtered_df.groupby(['Pair', 'Volume']).agg({
            'Traditional Fee ($)': 'mean',
            'Profit Share Fee ($)': 'mean',
            'Difference ($)': 'mean',
            'PS More Expensive': 'mean'
        }).reset_index()
        
        summary_csv = summary_stats.to_csv(index=False)
        st.download_button(
            label="Download Summary CSV",
            data=summary_csv,
            file_name=f"summary_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )

if __name__ == "__main__":
    main()