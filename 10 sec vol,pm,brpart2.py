# Add this function to fetch Rollbit price data
@st.cache_data(ttl=30)
def get_rollbit_price_data(token, hours=3):
    """Fetch Rollbit price data for a given token"""
    now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
    start_time_sg = now_sg - timedelta(hours=hours)
    
    start_time_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    
    # Handle the PROD suffix in token names
    clean_token = token.replace('PROD', '')
    
    query = f"""
    SELECT 
        created_at + INTERVAL '8 hour' AS timestamp,
        final_price
    FROM oracle_price_log_partition_{now_sg.strftime("%Y%m%d")}
    WHERE created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
    AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
    AND source_type = 1
    AND pair_name = '{clean_token}'
    ORDER BY timestamp
    """
    
    try:
        df = pd.read_sql_query(query, engine)
        
        # If we don't have enough data, try yesterday's partition too
        if df.empty or len(df) < 10:
            yesterday_str = (now_sg - timedelta(days=1)).strftime("%Y%m%d")
            query_yesterday = f"""
            SELECT 
                created_at + INTERVAL '8 hour' AS timestamp,
                final_price
            FROM oracle_price_log_partition_{yesterday_str}
            WHERE created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
            AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
            AND source_type = 1
            AND pair_name = '{clean_token}'
            ORDER BY timestamp
            """
            try:
                df_yesterday = pd.read_sql_query(query_yesterday, engine)
                df = pd.concat([df_yesterday, df]).drop_duplicates().sort_values('timestamp')
            except Exception as e:
                st.warning(f"Could not fetch yesterday's Rollbit price data: {e}")
        
        if df.empty:
            st.warning(f"No Rollbit price data found for {token} with source_type=1")
            return None
            
        # Process timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        return df
    except Exception as e:
        st.error(f"Error fetching Rollbit price data for {token}: {e}")
        return None

# Add this function to calculate volatility for Rollbit price data
def calculate_rollbit_volatility(price_data):
    """Calculate 10-second volatility from price data"""
    if price_data is None or len(price_data) < 2:
        return None
    
    # Resample to 500ms
    price_resampled = price_data['final_price'].resample('500ms').ffill().dropna()
    
    if len(price_resampled) < 2:
        return None
    
    # Create 10-second windows
    result = []
    start_date = price_resampled.index.min().floor('10s')
    end_date = price_resampled.index.max().ceil('10s')
    ten_sec_periods = pd.date_range(start=start_date, end=end_date, freq='10s')
    
    for i in range(len(ten_sec_periods)-1):
        start_window = ten_sec_periods[i]
        end_window = ten_sec_periods[i+1]
        
        window_data = price_resampled[(price_resampled.index >= start_window) & (price_resampled.index < end_window)]
        
        if len(window_data) >= 2:
            # Calculate volatility for 10-second window
            log_returns = np.diff(np.log(window_data.values))
            if len(log_returns) > 0:
                annualization_factor = np.sqrt(3153600)  # For 10-second windows
                volatility = np.std(log_returns) * annualization_factor
                
                result.append({
                    'timestamp': start_window,
                    'rollbit_vol': volatility
                })
    
    if not result:
        return None
    
    result_df = pd.DataFrame(result).set_index('timestamp')
    return result_df

# Now modify the main chart section to include Rollbit volatility
# Find where you load data in the "with st.spinner" section and add:
with st.spinner(f"Loading data for {selected_token}..."):
    vol_data, percentiles = get_volatility_data_10sec(selected_token)
    rollbit_params = fetch_rollbit_parameters_10sec(selected_token)
    uat_buffer = fetch_uat_buffer_rates_10sec(selected_token)
    
    # Add this new line to fetch Rollbit price data
    rollbit_price_data = get_rollbit_price_data(selected_token)
    rollbit_vol_data = None
    if rollbit_price_data is not None and not rollbit_price_data.empty:
        rollbit_vol_data = calculate_rollbit_volatility(rollbit_price_data)

# Then modify the part where you build combined_data and the volatility chart
if vol_data is not None and not vol_data.empty:
    # Convert to percentage
    vol_data_pct = vol_data.copy()
    vol_data_pct['realized_vol'] = vol_data_pct['realized_vol'] * 100

    # Key metrics
    current_vol = vol_data_pct['realized_vol'].iloc[-1]
    avg_vol = vol_data_pct['realized_vol'].mean()
    max_vol = vol_data_pct['realized_vol'].max()
    min_vol = vol_data_pct['realized_vol'].min()
    
    # Calculate current percentile
    all_vols = vol_data_pct['realized_vol'].values
    current_percentile = (all_vols < current_vol).mean() * 100

    # Always create 3 rows (no position multiplier)
    num_rows = 3
    row_heights = [0.4, 0.3, 0.3]

    # Create subplots without titles
    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights
    )

    # Process all data into combined dataframe
    combined_data = vol_data_pct.copy()
    
    # Add Rollbit volatility data if available
    if rollbit_vol_data is not None and not rollbit_vol_data.empty:
        rollbit_vol_data['rollbit_vol_pct'] = rollbit_vol_data['rollbit_vol'] * 100
        combined_data = pd.merge(
            combined_data,
            rollbit_vol_data[['rollbit_vol_pct']],
            left_index=True,
            right_index=True,
            how='outer'
        )
        # Make sure we have values for both volatilities
        combined_data = combined_data.sort_index()
        
        # Calculate metrics for Rollbit volatility
        rollbit_current_vol = combined_data['rollbit_vol_pct'].dropna().iloc[-1] if not combined_data['rollbit_vol_pct'].dropna().empty else None
        rollbit_avg_vol = combined_data['rollbit_vol_pct'].mean() if not combined_data['rollbit_vol_pct'].dropna().empty else None
    else:
        combined_data['rollbit_vol_pct'] = np.nan
        rollbit_current_vol = None
        rollbit_avg_vol = None
    
    # Rest of your code for adding Rollbit params and UAT buffer...
    # [keep your existing code here]
    
    # Update custom hover data to include Rollbit volatility
    hover_template = (
        "<b>Time: %{x}</b><br>" +
        "Volatility: %{customdata[0]:.1f}%<br>" +
        "Rollbit Volatility: %{customdata[3]:.1f}%<br>" +  # Add this line
        "Rollbit Buffer: %{customdata[1]:.3f}%<br>" +
        "UAT Buffer: %{customdata[2]:.3f}%<br>" +
        "<extra></extra>"
    )
    
    customdata = np.column_stack((
        combined_data['realized_vol'].fillna(0),
        combined_data['buffer_rate_pct'].fillna(0),
        combined_data['uat_buffer_rate_pct'].fillna(0),
        combined_data['rollbit_vol_pct'].fillna(0)  # Add this line
    ))
    
    # Panel 1: Volatility - Add both volatility lines
    fig.add_trace(
        go.Scatter(
            x=combined_data.index,
            y=combined_data['realized_vol'],
            mode='lines',
            line=dict(color='blue', width=2),
            name="Oracle Volatility (%)",  # Changed name to clarify
            customdata=customdata,
            hovertemplate=hover_template,
        ),
        row=1, col=1
    )
    
    # Add Rollbit volatility line if available
    if rollbit_vol_data is not None and not rollbit_vol_data.empty:
        fig.add_trace(
            go.Scatter(
                x=combined_data.index,
                y=combined_data['rollbit_vol_pct'],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),  # Different color and style
                name="Rollbit Volatility (%)",
                customdata=customdata,
                hovertemplate=hover_template,
            ),
            row=1, col=1
        )
    
    # Update title to include Rollbit volatility info
    title_parts = [f"{selected_token} Analysis Dashboard (10-second windows)<br>"]
    subtitle_parts = [f"Oracle Vol: {current_vol:.1f}% ({current_percentile:.0f}th percentile)"]
    
    if rollbit_current_vol is not None:
        subtitle_parts.append(f"Rollbit Vol: {rollbit_current_vol:.1f}%")
    
    if latest_rollbit_buffer is not None:
        subtitle_parts.append(f"Rollbit Buffer: {latest_rollbit_buffer:.3f}%")
    if latest_uat_buffer is not None:
        subtitle_parts.append(f"UAT Buffer: {latest_uat_buffer:.3f}%")
    
    title_text = title_parts[0] + f"<sub>{' | '.join(subtitle_parts)}</sub>"
    
    # Update layout
    fig.update_layout(
        title=title_text,
        height=800,
        # Now we want to show the legend to distinguish between volatility lines
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        # Rest of your layout code...
    )
    
    # Update metrics display to include Rollbit volatility
    st.markdown("### Key Metrics")
    
    # Add a row for Oracle and Rollbit volatility comparison
    st.markdown("#### Oracle Volatility")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current", f"{current_vol:.1f}%", f"{current_percentile:.0f}th %ile")
    with col2:
        st.metric("Average", f"{avg_vol:.1f}%")
    with col3:
        st.metric("Max", f"{max_vol:.1f}%")
    with col4:
        st.metric("Min", f"{min_vol:.1f}%")
    
    # Add Rollbit volatility metrics if available
    if rollbit_vol_data is not None and not rollbit_vol_data.empty:
        st.markdown("#### Rollbit Volatility")
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            if rollbit_current_vol is not None:
                st.metric("Current", f"{rollbit_current_vol:.1f}%")
            else:
                st.metric("Current", "N/A")
        with rcol2:
            if rollbit_avg_vol is not None:
                st.metric("Average", f"{rollbit_avg_vol:.1f}%")
            else:
                st.metric("Average", "N/A")
    
    # Rest of your code remains the same...