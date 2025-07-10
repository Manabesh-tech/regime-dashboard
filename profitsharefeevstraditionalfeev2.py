import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine

# Database connection parameters
DB_PARAMS = {
    'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
    'port': 5432,
    'database': 'replication_report',
    'user': 'public_replication',
    'password': '866^FKC4hllk'
}

# Global engine
ENGINE = create_engine(
    f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['database']}",
    isolation_level="AUTOCOMMIT",
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
    pool_use_lifo=True,
    echo=False
)

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

@st.cache_data(ttl=300)
def fetch_metabase_data():
    """Fetch traditional spreads from Metabase"""
    try:
        query = """
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
        df = pd.read_sql(query, ENGINE)
        st.success(f"✅ Successfully loaded {len(df)} pairs from database (latest time_group)")
        return df
    except Exception as e:
        st.error(f"❌ Error fetching Metabase data: {str(e)}")
        st.info("🔄 Falling back to mock data")
        return get_mock_traditional_data()

@st.cache_data(ttl=300)
def fetch_streamlit_data():
    """Fetch profit share parameters from database"""
    try:
        spreads_query = """
        SELECT DISTINCT pair_name
        FROM oracle_exchange_spread 
        WHERE source = 'binanceFuture'
            AND time_group = (SELECT MAX(time_group) FROM oracle_exchange_spread)
        ORDER BY pair_name
        """
        spreads_pairs_df = pd.read_sql(spreads_query, ENGINE)
        spreads_pairs = spreads_pairs_df['pair_name'].tolist()
        
        st.info(f"🔍 Traditional pairs found: {len(spreads_pairs)}")
        st.info(f"📋 First 10 traditional pairs: {spreads_pairs[:10]}")
        
        profit_pairs_query = """
        SELECT DISTINCT pair_name 
        FROM trade_pool_pairs where status in (1,2)
        ORDER BY pair_name
        """
        profit_pairs_df = pd.read_sql(profit_pairs_query, ENGINE)
        profit_pairs = profit_pairs_df['pair_name'].tolist()
        
        st.info(f"🔍 Profit-share pairs found: {len(profit_pairs)}")
        st.info(f"📋 First 10 profit-share pairs: {profit_pairs[:10]}")
        
        matching_pairs = list(set(spreads_pairs).intersection(set(profit_pairs)))
        st.info(f"🎯 Matching pairs: {len(matching_pairs)}")
        st.info(f"📋 Matching pairs: {matching_pairs[:10] if len(matching_pairs) > 10 else matching_pairs}")
        
        if len(matching_pairs) == 0:
            st.error("❌ No matching pairs found - this shouldn't happen!")
            return get_mock_profit_share_data()
        
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
        WHERE pair_name IN ({})
        ORDER BY pair_name;
        """.format(','.join([f"'{pair}'" for pair in matching_pairs]))
        
        df = pd.read_sql(query, ENGINE)
        
        df["Base Rate"] = df["Base Rate"]
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
    ])

def calculate_traditional_fee(basis_points, position_size):
    """Calculate traditional spread fee in dollars"""
    return (basis_points / 10000) * position_size

def calculate_profit_sharing_fee(open_price, close_price, bet_amount, params):
    """Calculate profit sharing fee using the exact formula"""
    try:
        base_rate = params['base_rate']
        rate_multiplier = params['rate_multiplier']
        rate_exponent = params['rate_exponent']
        position_multiplier = params['position_multiplier']
        bet_multiplier = params['bet_multiplier']
        
        P_t = float(open_price)
        P_T = float(close_price)
        
        if P_t == 0:
            return 0
        
        price_move_pct = (P_T / P_t) - 1
        abs_price_move_pct = abs(price_move_pct)
        
        if abs_price_move_pct <= 1e-10:
            return 0
            
        if rate_multiplier <= 0 or position_multiplier <= 0 or bet_amount <= 0:
            return 0
        
        try:
            exponential_term = 1 / (abs_price_move_pct * rate_multiplier) ** rate_exponent
        except (ZeroDivisionError, OverflowError):
            exponential_term = 0
        
        try:
            position_term = (bet_amount * bet_multiplier) / (10**6 * abs_price_move_pct * position_multiplier)
        except (ZeroDivisionError, OverflowError):
            position_term = 0
        
        denominator = 1 + exponential_term + position_term
        if denominator == 0:
            return 0
            
        profit_share_fraction = (1 - base_rate) / denominator
        
        P_close = P_t + profit_share_fraction * (P_T - P_t)
        
        fee_charged_points = (P_T - P_close)
        fee_charged_dollars = (fee_charged_points * bet_amount * bet_multiplier) / 100
        
        return max(0, fee_charged_dollars)
        
    except Exception as e:
        st.error(f"Error in profit sharing calculation: {str(e)}")
        return 0

def create_comparison_table(traditional_data, profit_share_data):
    """Create comprehensive comparison table"""
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
        
        ps_params = {
            'base_rate': row.get('Base Rate', 0.1),
            'rate_multiplier': row.get('Rate Multiplier', 1000),
            'rate_exponent': row.get('Rate Exponent', 1),
            'position_multiplier': row.get('Position Multiplier', 200),
            'bet_multiplier': row.get('Bet Multiplier', 1)
        }
        
        for volume in volume_levels:
            position_size = volume_amounts[volume]
            traditional_spread = row[volume] + 10
            traditional_fee = calculate_traditional_fee(traditional_spread, position_size) * 2
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

def show_overview_analysis(comparison_df):
    """Show the existing overview analysis"""
    if comparison_df.empty:
        st.error("No data available for analysis")
        return
        
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        select_all_pairs = st.checkbox("Select All Pairs", key="select_all_pairs_overview")
        
        if select_all_pairs:
            selected_pairs = list(comparison_df['Pair'].unique())
        else:
            selected_pairs = st.multiselect(
                "Select Pairs", 
                options=comparison_df['Pair'].unique(),
                default=list(comparison_df['Pair'].unique())[:5],
                key="overview_pairs"
            )
    
    with col2:
        select_all_volumes = st.checkbox("Select All Volumes", key="select_all_volumes_overview")
        
        if select_all_volumes:
            selected_volumes = volume_levels
        else:
            selected_volumes = st.multiselect(
                "Select Volumes",
                options=volume_levels,
                default=volume_levels,
                key="overview_volumes"
            )
    
    with col3:
        select_all_moves = st.checkbox("Select All Price Moves", key="select_all_moves_overview")
        
        if select_all_moves:
            selected_moves = price_moves
        else:
            selected_moves = st.multiselect(
                "Select Price Moves (%)",
                options=price_moves,
                default=price_moves,
                key="overview_moves"
            )
    
    filtered_df = comparison_df[
        (comparison_df['Pair'].isin(selected_pairs)) &
        (comparison_df['Volume'].isin(selected_volumes)) &
        (comparison_df['Price Move (%)'].isin(selected_moves))
    ]
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters")
        return
    
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
    
    st.subheader("📋 Detailed Comparison")
    display_df = filtered_df.copy()
    display_df['Traditional Fee ($)'] = display_df['Traditional Fee ($)'].apply(lambda x: f"${x:.3f}")
    display_df['Profit Share Fee ($)'] = display_df['Profit Share Fee ($)'].apply(lambda x: f"${x:.3f}")
    display_df['Difference ($)'] = display_df['Difference ($)'].apply(lambda x: f"${x:.3f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.subheader("📊 Visualizations")
    fig1 = px.line(
        filtered_df,
        x='Price Move (%)',
        y='Profit Share Fee ($)',
        color='Pair',
        facet_col='Volume',
        title="Profit Share Fees vs Traditional Fees by Price Move"
    )
    
    for pair in selected_pairs:
        for i, volume in enumerate(['1k', '5k', '10k', '20k']):
            if volume in selected_volumes:
                pair_data = filtered_df[(filtered_df['Pair'] == pair) & (filtered_df['Volume'] == volume)]
                if not pair_data.empty:
                    traditional_fee = pair_data['Traditional Fee ($)'].iloc[0]
                    
                    fig1.add_hline(
                        y=traditional_fee,
                        line_dash="solid",
                        line_color="black",
                        line_width=2,
                        opacity=0.8,
                        col=i+1,
                        annotation_text=f"{pair} Traditional: ${traditional_fee:.3f}",
                        annotation_position="top right",
                        annotation_font_size=8
                    )
    
    fig1.update_layout(
        showlegend=True,
        height=600,
        title_text="Profit Share Fees vs Traditional Fees by Price Move<br><sub>Black lines show traditional fees (constant regardless of price move)</sub>"
    )
    
    st.plotly_chart(fig1, use_container_width=True)

def show_pair_specific_analysis(comparison_df):
    """Show detailed analysis for a specific pair"""
    if comparison_df.empty:
        st.error("No data available for pair-specific analysis")
        return
    
    st.subheader("🎯 Select Pair for Detailed Analysis")
    selected_pair = st.selectbox(
        "Choose a trading pair:",
        options=comparison_df['Pair'].unique(),
        key="pair_specific_selector"
    )
    
    pair_data = comparison_df[comparison_df['Pair'] == selected_pair]
    
    if pair_data.empty:
        st.warning(f"No data available for {selected_pair}")
        return
    
    st.subheader(f"📊 Detailed Analysis: {selected_pair}")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['1k Volume', '5k Volume', '10k Volume', '20k Volume'],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    colors = ['blue', 'red', 'green', 'orange']
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for i, volume in enumerate(volume_levels):
        volume_data = pair_data[pair_data['Volume'] == volume].sort_values('Price Move (%)')
        
        if not volume_data.empty:
            row, col = positions[i]
            
            fig.add_trace(
                go.Scatter(
                    x=volume_data['Price Move (%)'],
                    y=volume_data['Profit Share Fee ($)'],
                    mode='lines+markers',
                    name=f'Profit Share ({volume})',
                    line=dict(color=colors[i], width=3),
                    marker=dict(size=6)
                ),
                row=row, col=col
            )
            
            traditional_fee = volume_data['Traditional Fee ($)'].iloc[0]
            fig.add_hline(
                y=traditional_fee,
                line_dash="solid",
                line_color="black",
                line_width=3,
                row=row, col=col,
                annotation_text=f"Traditional: ${traditional_fee:.3f}",
                annotation_position="top right"
            )
    
    fig.update_layout(
        height=800,
        title_text=f"Fee Comparison for {selected_pair}<br><sub>Black lines = Traditional fees, Colored lines = Profit-sharing fees</sub>",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Price Move (%)")
    fig.update_yaxes(title_text="Fee ($)")
    
    st.plotly_chart(fig, use_container_width=True)

def show_breakeven_analysis(traditional_data, profit_share_data):
    """Show breakeven parameter analysis"""
    if traditional_data.empty or profit_share_data.empty:
        st.error("No data available for breakeven analysis")
        return
    
    merged_data = traditional_data.merge(
        profit_share_data, 
        left_on='pair_name', 
        right_on='Pair', 
        how='inner'
    )
    
    if merged_data.empty:
        st.error("No matching pairs for breakeven analysis")
        return
    
    st.subheader("⚖️ Breakeven Parameter Analysis")
    selected_pair = st.selectbox(
        "Choose a trading pair for breakeven analysis:",
        options=merged_data['pair_name'].unique(),
        key="breakeven_pair_selector"
    )
    
    pair_row = merged_data[merged_data['pair_name'] == selected_pair].iloc[0]
    
    current_params = {
        'base_rate': pair_row.get('Base Rate', 0.1),
        'rate_multiplier': pair_row.get('Rate Multiplier', 1000),
        'rate_exponent': pair_row.get('Rate Exponent', 1),
        'position_multiplier': pair_row.get('Position Multiplier', 200),
        'bet_multiplier': pair_row.get('Bet Multiplier', 1)
    }
    
    st.subheader(f"🎯 Current Parameters for {selected_pair}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Position Multiplier", current_params['position_multiplier'])
        st.metric("Rate Multiplier", current_params['rate_multiplier'])
    with col2:
        st.metric("Rate Exponent", current_params['rate_exponent'])
        st.metric("Base Rate", f"{current_params['base_rate']:.3f}")
    with col3:
        st.metric("Bet Multiplier", current_params['bet_multiplier'])
    
    st.subheader("🔍 Breakeven Parameter Optimization")
    
    st.markdown("### 📊 Position Multiplier Optimization")
    optimize_position_multiplier(pair_row, current_params)
    
    st.markdown("---")
    
    st.markdown("### 📊 Rate Multiplier Optimization") 
    optimize_rate_multiplier(pair_row, current_params)
    
    st.markdown("---")
    
    st.markdown("### 📊 Rate Exponent Optimization")
    optimize_rate_exponent(pair_row, current_params)

def parse_dollar_value(value_str):
    """Parse dollar value from string format"""
    return float(str(value_str).replace('$', '').replace(',', ''))

def optimize_position_multiplier(pair_row, current_params):
    """Find breakeven position multiplier"""
    try:
        multiplier_range = range(1, 1001, 10)
        results = []
        
        for volume in volume_levels:
            traditional_spread = pair_row[volume] + 10
            traditional_fee = calculate_traditional_fee(traditional_spread, volume_amounts[volume]) * 2
            
            bet_amount = volume_amounts[volume] / current_params['bet_multiplier']
            current_ps_fee = calculate_profit_sharing_fee(100, 101, bet_amount, current_params)
            current_difference = traditional_fee - current_ps_fee
            
            best_multiplier = None
            min_abs_diff = float('inf')
            best_ps_fee = 0
            
            for pos_mult in multiplier_range:
                test_params = current_params.copy()
                test_params['position_multiplier'] = pos_mult
                
                ps_fee = calculate_profit_sharing_fee(100, 101, bet_amount, test_params)
                abs_diff = abs(ps_fee - traditional_fee)
                
                if abs_diff < min_abs_diff:
                    min_abs_diff = abs_diff
                    best_multiplier = pos_mult
                    best_ps_fee = ps_fee
            
            optimized_difference = traditional_fee - best_ps_fee
            improvement = abs(current_difference) - abs(optimized_difference)
            
            results.append({
                'Volume': volume,
                'Traditional Fee ($)': f"${traditional_fee:.3f}",
                'Current Position Multiplier': current_params['position_multiplier'],
                'Current PS Fee ($)': f"${current_ps_fee:.3f}",
                'Current Difference ($)': f"${current_difference:.3f}",
                'Optimal Position Multiplier': best_multiplier,
                'Optimal PS Fee ($)': f"${best_ps_fee:.3f}",
                'Optimal Difference ($)': f"${optimized_difference:.3f}",
                'Improvement ($)': f"${improvement:.3f}"
            })
        
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, height=300)
        
        improvement_values = [parse_dollar_value(x) for x in df['Improvement ($)']]
        avg_improvement = sum(improvement_values) / len(improvement_values)
        
        if avg_improvement > 0:
            st.success(f"💡 Average improvement: ${avg_improvement:.3f} - Position multiplier optimization recommended!")
        else:
            st.info("📊 Current position multiplier is already well-optimized")
        
    except Exception as e:
        st.error(f"Error in position multiplier optimization: {str(e)}")

def optimize_rate_multiplier(pair_row, current_params):
    """Find breakeven rate multiplier"""
    try:
        multiplier_range = range(100, 5001, 100)
        results = []
        
        for volume in volume_levels:
            traditional_spread = pair_row[volume] + 10
            traditional_fee = calculate_traditional_fee(traditional_spread, volume_amounts[volume]) * 2
            
            bet_amount = volume_amounts[volume] / current_params['bet_multiplier']
            current_ps_fee = calculate_profit_sharing_fee(100, 101, bet_amount, current_params)
            current_difference = traditional_fee - current_ps_fee
            
            best_multiplier = None
            min_abs_diff = float('inf')
            best_ps_fee = 0
            
            for rate_mult in multiplier_range:
                test_params = current_params.copy()
                test_params['rate_multiplier'] = rate_mult
                
                ps_fee = calculate_profit_sharing_fee(100, 101, bet_amount, test_params)
                abs_diff = abs(ps_fee - traditional_fee)
                
                if abs_diff < min_abs_diff:
                    min_abs_diff = abs_diff
                    best_multiplier = rate_mult
                    best_ps_fee = ps_fee
            
            optimized_difference = traditional_fee - best_ps_fee
            improvement = abs(current_difference) - abs(optimized_difference)
            
            results.append({
                'Volume': volume,
                'Traditional Fee ($)': f"${traditional_fee:.3f}",
                'Current Rate Multiplier': current_params['rate_multiplier'],
                'Current PS Fee ($)': f"${current_ps_fee:.3f}",
                'Current Difference ($)': f"${current_difference:.3f}",
                'Optimal Rate Multiplier': best_multiplier,
                'Optimal PS Fee ($)': f"${best_ps_fee:.3f}",
                'Optimal Difference ($)': f"${optimized_difference:.3f}",
                'Improvement ($)': f"${improvement:.3f}"
            })
        
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, height=300)
        
        improvement_values = [parse_dollar_value(x) for x in df['Improvement ($)']]
        avg_improvement = sum(improvement_values) / len(improvement_values)
        
        if avg_improvement > 0:
            st.success(f"💡 Average improvement: ${avg_improvement:.3f} - Rate multiplier optimization recommended!")
        else:
            st.info("📊 Current rate multiplier is already well-optimized")
        
    except Exception as e:
        st.error(f"Error in rate multiplier optimization: {str(e)}")

def optimize_rate_exponent(pair_row, current_params):
    """Find breakeven rate exponent"""
    try:
        exponent_range = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3]
        results = []
        
        for volume in volume_levels:
            traditional_spread = pair_row[volume] + 10
            traditional_fee = calculate_traditional_fee(traditional_spread, volume_amounts[volume]) * 2
            
            bet_amount = volume_amounts[volume] / current_params['bet_multiplier']
            current_ps_fee = calculate_profit_sharing_fee(100, 101, bet_amount, current_params)
            current_difference = traditional_fee - current_ps_fee
            
            best_exponent = None
            min_abs_diff = float('inf')
            best_ps_fee = 0
            
            for rate_exp in exponent_range:
                test_params = current_params.copy()
                test_params['rate_exponent'] = rate_exp
                
                ps_fee = calculate_profit_sharing_fee(100, 101, bet_amount, test_params)
                abs_diff = abs(ps_fee - traditional_fee)
                
                if abs_diff < min_abs_diff:
                    min_abs_diff = abs_diff
                    best_exponent = rate_exp
                    best_ps_fee = ps_fee
            
            optimized_difference = traditional_fee - best_ps_fee
            improvement = abs(current_difference) - abs(optimized_difference)
            
            results.append({
                'Volume': volume,
                'Traditional Fee ($)': f"${traditional_fee:.3f}",
                'Current Rate Exponent': current_params['rate_exponent'],
                'Current PS Fee ($)': f"${current_ps_fee:.3f}",
                'Current Difference ($)': f"${current_difference:.3f}",
                'Optimal Rate Exponent': best_exponent,
                'Optimal PS Fee ($)': f"${best_ps_fee:.3f}",
                'Optimal Difference ($)': f"${optimized_difference:.3f}",
                'Improvement ($)': f"${improvement:.3f}"
            })
        
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, height=300)
        
        improvement_values = [parse_dollar_value(x) for x in df['Improvement ($)']]
        avg_improvement = sum(improvement_values) / len(improvement_values)
        
        if avg_improvement > 0:
            st.success(f"💡 Average improvement: ${avg_improvement:.3f} - Rate exponent optimization recommended!")
        else:
            st.info("📊 Current rate exponent is already well-optimized")
        
    except Exception as e:
        st.error(f"Error in rate exponent optimization: {str(e)}")

def main():
    refresh_data = st.sidebar.button("🔄 Refresh Data")
    show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)
    
    if refresh_data or 'traditional_data' not in st.session_state:
        with st.spinner("Loading data..."):
            st.session_state.traditional_data = fetch_metabase_data()
            st.session_state.profit_share_data = fetch_streamlit_data()
    
    traditional_data = st.session_state.traditional_data
    profit_share_data = st.session_state.profit_share_data
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Traditional Spreads", f"{len(traditional_data)} pairs")
    with col2:
        st.metric("Profit Share Parameters", f"{len(profit_share_data)} pairs")
    
    if show_raw_data:
        st.subheader("Raw Data")
        tab1, tab2 = st.tabs(["Traditional Spreads", "Profit Share Parameters"])
        
        with tab1:
            st.dataframe(traditional_data)
        
        with tab2:
            st.dataframe(profit_share_data)
    
    st.subheader("📊 Comparison Analysis")
    
    with st.spinner("Calculating comparisons..."):
        comparison_df = create_comparison_table(traditional_data, profit_share_data)
    
    if comparison_df.empty:
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["📊 Overview Analysis", "🔍 Pair-Specific Analysis", "⚖️ Breakeven Analysis"])
    
    with tab1:
        show_overview_analysis(comparison_df)
    
    with tab2:
        show_pair_specific_analysis(comparison_df)
    
    with tab3:
        show_breakeven_analysis(traditional_data, profit_share_data)

if __name__ == "__main__":
    main()