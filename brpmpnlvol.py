# Function to create volatility plot
def create_volatility_plot(pair_name):
    """Create a plot of volatility data."""
    if pair_name not in st.session_state.pair_data:
        return None
    
    # Get volatility data
    vol_df, current_vol, daily_avg = calculate_volatility(pair_name)
    
    if vol_df is None:
        return None
    
    # Convert to percentage for display
    vol_df_pct = vol_df.copy()
    vol_df_pct['realized_vol'] = vol_df_pct['realized_vol'] * 100  # Convert to percentage
    
    # Create figure
    fig = go.Figure()
    
    # Add volatility line
    fig.add_trace(
        go.Scatter(
            x=vol_df_pct.index,
            y=vol_df_pct['realized_vol'],
            mode='lines',
            line=dict(color='blue', width=2),
            name="Realized Volatility"
        )
    )
    
    # Add daily average line
    if daily_avg is not None:
        daily_avg_pct = daily_avg * 100
        fig.add_trace(
            go.Scatter(
                x=[vol_df_pct.index.min(), vol_df_pct.index.max()],
                y=[daily_avg_pct, daily_avg_pct],
                mode='lines',
                line=dict(color='green', width=2, dash='dash'),
                name=f"Daily Average: {daily_avg_pct:.2f}%"
            )
        )
        
        # Add threshold lines
        threshold1_pct = daily_avg_pct * (1 + st.session_state.vol_threshold_1/100)
        threshold2_pct = daily_avg_pct * (1 + st.session_state.vol_threshold_2/100)
        
        fig.add_trace(
            go.Scatter(
                x=[vol_df_pct.index.min(), vol_df_pct.index.max()],
                y=[threshold1_pct, threshold1_pct],
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                name=f"Threshold 1 (+{st.session_state.vol_threshold_1}%): {threshold1_pct:.2f}%"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[vol_df_pct.index.min(), vol_df_pct.index.max()],
                y=[threshold2_pct, threshold2_pct],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name=f"Threshold 2 (+{st.session_state.vol_threshold_2}%): {threshold2_pct:.2f}%"
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"{pair_name} Realized Volatility (5-minute)",
        xaxis_title="Time",
        yaxis_title="Annualized Volatility (%)",
        height=500,
        hovermode="x unified"
    )
    
    return fig

# Function to create PnL plot
def create_pnl_plot(pair_name):
    """Create a plot of PnL data."""
    if pair_name not in st.session_state.pair_data:
        return None
    
    # Get PnL data
    pnl_df, _ = calculate_pnl(pair_name)
    
    if pnl_df is None or pnl_df.empty:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add PnL line
    fig.add_trace(
        go.Scatter(
            x=pnl_df['timestamp'],
            y=pnl_df['cumulative_pnl'],
            mode='lines',
            line=dict(color='blue', width=2),
            name="Cumulative PnL"
        )
    )
    
    # Add threshold lines
    is_major = st.session_state.is_major_pairs.get(pair_name, False)
    threshold1 = st.session_state.pnl_threshold_major_1 if is_major else st.session_state.pnl_threshold_alt_1
    threshold2 = st.session_state.pnl_threshold_major_2 if is_major else st.session_state.pnl_threshold_alt_2
    
    fig.add_trace(
        go.Scatter(
            x=[pnl_df['timestamp'].min(), pnl_df['timestamp'].max()],
            y=[threshold1, threshold1],
            mode='lines',
            line=dict(color='orange', width=2, dash='dash'),
            name=f"Threshold 1: {threshold1}"
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[pnl_df['timestamp'].min(), pnl_df['timestamp'].max()],
            y=[threshold2, threshold2],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f"Threshold 2: {threshold2}"
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{pair_name} Cumulative PnL",
        xaxis_title="Time",
        yaxis_title="PnL Value",
        height=500,
        hovermode="x unified"
    )
    
    return fig

# Function to create parameter history plot
def create_parameter_history_plot(pair_name):
    """Create a plot of parameter adjustment history."""
    if pair_name not in st.session_state.pair_data or not st.session_state.pair_data[pair_name]['parameter_history']:
        return None, None
    
    # Extract data
    history = st.session_state.pair_data[pair_name]['parameter_history']
    timestamps = [entry[0] for entry in history]
    buffer_rates = [entry[1] for entry in history]
    position_multipliers = [entry[2] for entry in history]
    
    # Create buffer rate figure
    br_fig = go.Figure()
    br_fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=buffer_rates,
            mode='lines+markers',
            name="Buffer Rate"
        )
    )
    
    br_fig.update_layout(
        title=f"{pair_name} Buffer Rate History",
        xaxis_title="Time",
        yaxis_title="Buffer Rate",
        height=400
    )
    
    # Create position multiplier figure
    pm_fig = go.Figure()
    pm_fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=position_multipliers,
            mode='lines+markers',
            name="Position Multiplier"
        )
    )
    
    pm_fig.update_layout(
        title=f"{pair_name} Position Multiplier History",
        xaxis_title="Time",
        yaxis_title="Position Multiplier",
        height=400
    )
    
    return br_fig, pm_fig

# Calculate time until next auto-update
def time_until_next_update():
    """Calculate time until next auto-update."""
    if not st.session_state.auto_update_enabled:
        return None
    
    now = get_sg_time()
    last_update = st.session_state.last_auto_update
    
    # Make sure last_update is timezone-aware
    if last_update.tzinfo is None:
        last_update = pytz.utc.localize(last_update).astimezone(SG_TZ)
    
    elapsed_seconds = (now - last_update).total_seconds()
    
    if elapsed_seconds >= 300:  # 5 minutes in seconds
        return 0
    
    return 300 - elapsed_seconds  # Time remaining in seconds

# Update all pairs function
def update_all_pairs():
    """Update data for all monitored pairs."""
    pairs_updated = 0
    for pair_name in st.session_state.pair_data.keys():
        update_pair_data(pair_name)
        pairs_updated += 1
    
    # Reset auto-update timer
    st.session_state.last_auto_update = get_sg_time()
    
    return pairs_updated

# Auto-update status and timer implementation with HTML-based countdown
def render_countdown_timer():
    """Render an HTML countdown timer that actually works."""
    remaining_seconds = time_until_next_update()
    if remaining_seconds is not None:
        minutes, seconds = divmod(int(remaining_seconds), 60)
        
        # Display current Singapore time and next update time
        now_sg = get_sg_time()
        next_update_time = now_sg + timedelta(seconds=remaining_seconds)
        
        # Create HTML for the timer
        timer_html = f"""
        <div class="update-timer">
            <div id="current-time">Current time (SGT): {now_sg.strftime('%H:%M:%S')}</div>
            <div id="timer">Next auto-update in: <span id="minutes">{minutes:02d}</span>:<span id="seconds">{seconds:02d}</span></div>
            <div>Next update at: {next_update_time.strftime('%H:%M:%S')}</div>
        </div>
        
        <script>
            // Only create one timer
            if (!window.timerInterval) {{
                // Set the countdown time
                var totalSeconds = {int(remaining_seconds)};
                
                // Update the timer every second
                window.timerInterval = setInterval(function() {{
                    totalSeconds--;
                    
                    if (totalSeconds <= 0) {{
                        clearInterval(window.timerInterval);
                        window.timerInterval = null;
                        // Refresh the page when timer reaches zero
                        window.location.reload();
                        return;
                    }}
                    
                    // Calculate minutes and seconds
                    var minutes = Math.floor(totalSeconds / 60);
                    var seconds = totalSeconds % 60;
                    
                    // Format with leading zeros
                    var displayMinutes = (minutes < 10 ? "0" : "") + minutes;
                    var displaySeconds = (seconds < 10 ? "0" : "") + seconds;
                    
                    // Update the display
                    document.getElementById("minutes").innerHTML = displayMinutes;
                    document.getElementById("seconds").innerHTML = displaySeconds;
                }}, 1000);
            }}
        </script>
        """
        
        st.markdown(timer_html, unsafe_allow_html=True)
        
        # Add additional auto-refresh meta tag for fallback
        if remaining_seconds <= 5 and remaining_seconds > 0:
            # This will refresh the page after the specified seconds
            refresh_in = max(1, int(remaining_seconds))
            st.markdown(f'<meta http-equiv="refresh" content="{refresh_in}">', unsafe_allow_html=True)

# Function to render main dashboard
def render_dashboard(pair_name):
    """Render the main dashboard for a specific pair."""
    st.markdown(f"## Parameter Dashboard: {pair_name}")
    
    # Use the countdown timer implementation
    render_countdown_timer()
    
    # Data refreshing controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Manual Update", key=f"refresh_{pair_name}", type="primary"):
            update_pair_data(pair_name)
            st.session_state.last_auto_update = get_sg_time()  # Reset auto-update timer
            st.success("Data updated successfully!")
            st.rerun()
    
    with col2:
        if st.button("Reset PnL", key=f"reset_pnl_{pair_name}"):
            reset_pnl(pair_name)
            st.success("PnL reset successfully!")
            st.rerun()
    
    with col3:
        auto_update = st.checkbox(
            "Auto-update (5 min)", 
            value=st.session_state.auto_update_enabled,
            key=f"auto_update_{pair_name}"
        )
        st.session_state.auto_update_enabled = auto_update
    
    # Display pair type (Major or Alt)
    is_major = st.session_state.is_major_pairs.get(pair_name, False)
    pair_type = "Major" if is_major else "Alt"
    st.markdown(f"**Pair Type**: {pair_type} (PnL Thresholds: {st.session_state.pnl_threshold_major_1}/{st.session_state.pnl_threshold_major_2 if is_major else st.session_state.pnl_threshold_alt_1}/{st.session_state.pnl_threshold_alt_2})")
    
    # Current metrics section
    st.markdown("### Current Metrics")
    
    # Create a grid of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_vol = st.session_state.pair_data[pair_name].get('current_volatility')
        daily_avg = st.session_state.pair_data[pair_name].get('daily_avg_volatility')
        
        if current_vol is not None and daily_avg is not None:
            vol_pct = current_vol * 100
            avg_pct = daily_avg * 100
            vol_increase = ((current_vol - daily_avg) / daily_avg) * 100 if daily_avg > 0 else 0
            
            vol_class = "indicator-green"
            if vol_increase >= st.session_state.vol_threshold_2:
                vol_class = "indicator-red"
            elif vol_increase >= st.session_state.vol_threshold_1:
                vol_class = "indicator-yellow"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Current Volatility</h4>
                <p class="{vol_class}">{vol_pct:.2f}%</p>
                <small>Daily Avg: {avg_pct:.2f}% ({vol_increase:+.2f}%)</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h4>Current Volatility</h4>
                <p>N/A</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        cumulative_pnl = st.session_state.pair_data[pair_name].get('pnl_cumulative', 0)
        is_major = st.session_state.is_major_pairs.get(pair_name, False)
        threshold1 = st.session_state.pnl_threshold_major_1 if is_major else st.session_state.pnl_threshold_alt_1
        threshold2 = st.session_state.pnl_threshold_major_2 if is_major else st.session_state.pnl_threshold_alt_2
        
        pnl_class = "indicator-green"
        if cumulative_pnl <= threshold2:
            pnl_class = "indicator-red"
        elif cumulative_pnl <= threshold1:
            pnl_class = "indicator-yellow"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Cumulative PnL</h4>
            <p class="{pnl_class}">{cumulative_pnl:.2f}</p>
            <small>Thresholds: {threshold1} / {threshold2}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Get current buffer rate and base buffer rate
        current_br = st.session_state.pair_data[pair_name].get('buffer_rate')
        base_br = st.session_state.pair_data[pair_name].get('base_buffer_rate')
        
        if current_br is not None and base_br is not None:
            br_pct_change = ((current_br - base_br) / base_br) * 100
            
            br_class = "indicator-green"
            if br_pct_change > 0:
                br_class = "indicator-red" if br_pct_change >= 30 else "indicator-yellow"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Buffer Rate</h4>
                <p class="{br_class}">{current_br:.6f}</p>
                <small>Base: {base_br:.6f} ({br_pct_change:+.2f}%)</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h4>Buffer Rate</h4>
                <p>N/A</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        # Get current position multiplier and base position multiplier
        current_pm = st.session_state.pair_data[pair_name].get('position_multiplier')
        base_pm = st.session_state.pair_data[pair_name].get('base_position_multiplier')
        
        if current_pm is not None and base_pm is not None:
            pm_pct_change = ((current_pm - base_pm) / base_pm) * 100
            
            pm_class = "indicator-green"
            if pm_pct_change < 0:
                pm_class = "indicator-red" if pm_pct_change <= -30 else "indicator-yellow"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Position Multiplier</h4>
                <p class="{pm_class}">{current_pm:.1f}</p>
                <small>Base: {base_pm:.1f} ({pm_pct_change:+.2f}%)</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h4>Position Multiplier</h4>
                <p>N/A</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display adjustment tier information
    vol_tier = st.session_state.pair_data[pair_name].get('vol_adjustment_tier', 0)
    pnl_tier = st.session_state.pair_data[pair_name].get('pnl_adjustment_tier', 0)
    overall_tier = max(vol_tier, pnl_tier)
    
    tier_descriptions = {
        0: "Normal - Base parameters applied",
        1: f"Tier 1 - Parameters {st.session_state.parameter_adjustment_pct}% worse than base",
        2: f"Tier 2 - Parameters {st.session_state.parameter_adjustment_pct * 2}% worse than base"
    }
    
    tier_classes = {
        0: "tier-0",
        1: "tier-1",
        2: "tier-2"
    }
    
    st.markdown(f"""
    <div class="metric-card {tier_classes[overall_tier]}">
        <h4>Current Adjustment Tier</h4>
        <p>Tier {overall_tier}</p>
        <small>{tier_descriptions[overall_tier]}</small>
        <p>Volatility Tier: {vol_tier} | PnL Tier: {pnl_tier}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display charts in tabs
    tabs = st.tabs(["Volatility", "PnL", "Parameter History"])
    
    with tabs[0]:
        vol_fig = create_volatility_plot(pair_name)
        if vol_fig:
            st.plotly_chart(vol_fig, use_container_width=True)
        else:
            st.info("No volatility data available.")
    
    with tabs[1]:
        pnl_fig = create_pnl_plot(pair_name)
        if pnl_fig:
            st.plotly_chart(pnl_fig, use_container_width=True)
        else:
            st.info("No PnL data available.")
    
    with tabs[2]:
        br_fig, pm_fig = create_parameter_history_plot(pair_name)
        if br_fig and pm_fig:
            st.plotly_chart(br_fig, use_container_width=True)
            st.plotly_chart(pm_fig, use_container_width=True)
            
            # Also show history as a table
            history = st.session_state.pair_data[pair_name]['parameter_history']
            if history:
                st.markdown("### Parameter Adjustment History")
                history_df = pd.DataFrame(history, columns=['Timestamp', 'Buffer Rate', 'Position Multiplier', 'Reason'])
                st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No parameter adjustment history available.")

# Main application
def main():
    # Initialize session state
    init_session_state()
    
    # Check if daily reset is needed
    if check_daily_reset():
        st.success("Daily PnL reset completed.")
    
    # Check if auto-update is needed
    if check_auto_update():
        # Now update ALL monitored pairs, not just the current one
        update_all_pairs()
        st.rerun()
    
    # Set page title
    st.title("Volatility & PnL Parameter Adjustment System")
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # Pair selection
    available_pairs = fetch_pairs()
    
    if not available_pairs:
        st.error("No trading pairs found. Please check database connection.")
        return
    
    # Default to BTC/USDT if available
    default_index = 0
    if "BTC/USDT" in available_pairs:
        default_index = available_pairs.index("BTC/USDT")
    
    selected_pair = st.sidebar.selectbox(
        "Select Trading Pair",
        available_pairs,
        index=default_index
    )
    
    # Set current pair
    st.session_state.current_pair = selected_pair
    
    # Initialize pair if needed
    init_pair_state(selected_pair)
    
    # Pair type selection (Major or Alt)
    is_major = st.sidebar.checkbox(
        "This is a major pair",
        value=st.session_state.is_major_pairs.get(selected_pair, False),
        key=f"is_major_{selected_pair}"
    )
    st.session_state.is_major_pairs[selected_pair] = is_major
    
    # Add an "Update All Pairs" button
    if st.sidebar.button("Update All Pairs Data", type="primary"):
        with st.spinner("Updating data for all pairs..."):
            pairs_updated = update_all_pairs()
        st.sidebar.success(f"Updated data for {pairs_updated} pairs!")
        st.rerun()
    
    # Adjustment settings
    st.sidebar.markdown("### Adjustment Settings")
    
    # Volatility threshold settings
    vol_threshold_1 = st.sidebar.slider(
        "Volatility Threshold 1 (%)",
        min_value=10,
        max_value=200,
        value=st.session_state.vol_threshold_1,
        step=5,
        help="Percentage increase over daily average to trigger Tier 1 adjustment"
    )
    
    vol_threshold_2 = st.sidebar.slider(
        "Volatility Threshold 2 (%)",
        min_value=vol_threshold_1 + 10,
        max_value=300,
        value=max(st.session_state.vol_threshold_2, vol_threshold_1 + 10),
        step=5,
        help="Percentage increase over daily average to trigger Tier 2 adjustment"
    )
    
    # PnL threshold settings
    st.sidebar.markdown("#### PnL Thresholds for Major Pairs")
    pnl_threshold_major_1 = st.sidebar.number_input(
        "PnL Threshold 1 (Major)",
        value=st.session_state.pnl_threshold_major_1,
        step=50,
        help="PnL threshold to trigger Tier 1 adjustment for major pairs"
    )
    
    pnl_threshold_major_2 = st.sidebar.number_input(
        "PnL Threshold 2 (Major)",
        value=min(st.session_state.pnl_threshold_major_2, pnl_threshold_major_1 - 50),
        step=50,
        help="PnL threshold to trigger Tier 2 adjustment for major pairs"
    )
    
    st.sidebar.markdown("#### PnL Thresholds for Alt Pairs")
    pnl_threshold_alt_1 = st.sidebar.number_input(
        "PnL Threshold 1 (Alts)",
        value=st.session_state.pnl_threshold_alt_1,
        step=20,
        help="PnL threshold to trigger Tier 1 adjustment for altcoin pairs"
    )
    
    pnl_threshold_alt_2 = st.sidebar.number_input(
        "PnL Threshold 2 (Alts)",
        value=min(st.session_state.pnl_threshold_alt_2, pnl_threshold_alt_1 - 20),
        step=20,
        help="PnL threshold to trigger Tier 2 adjustment for altcoin pairs"
    )
    
    # Parameter adjustment percentage
    parameter_adjustment_pct = st.sidebar.slider(
        "Parameter Adjustment (%)",
        min_value=5,
        max_value=50,
        value=st.session_state.parameter_adjustment_pct,
        step=5,
        help="Percentage to adjust parameters at each tier"
    )
    
    # Update session state with new settings
    st.session_state.vol_threshold_1 = vol_threshold_1
    st.session_state.vol_threshold_2 = vol_threshold_2
    st.session_state.pnl_threshold_major_1 = pnl_threshold_major_1
    st.session_state.pnl_threshold_major_2 = pnl_threshold_major_2
    st.session_state.pnl_threshold_alt_1 = pnl_threshold_alt_1
    st.session_state.pnl_threshold_alt_2 = pnl_threshold_alt_2
    st.session_state.parameter_adjustment_pct = parameter_adjustment_pct
    
    # Apply settings button
    if st.sidebar.button("Apply Settings", key="apply_settings"):
        # Reapply parameter adjustments
        adjust_parameters(selected_pair)
        st.sidebar.success("Settings applied!")
        st.rerun()
    
    # Reset All Pairs PnL button
    if st.sidebar.button("Reset All Pairs PnL", key="reset_all_pnl"):
        reset_count = 0
        for pair in st.session_state.pair_data:
            if reset_pnl(pair):
                reset_count += 1
        st.sidebar.success(f"Reset PnL for {reset_count} pairs!")
        st.rerun()
    
    # Update data button
    if st.sidebar.button("Update Data Now", key="update_data"):
        with st.spinner("Updating data..."):
            update_pair_data(selected_pair)
        st.session_state.last_auto_update = get_sg_time()  # Reset auto-update timer
        st.sidebar.success("Data updated successfully!")
        st.rerun()
    
    # Render the dashboard for the selected pair
    render_dashboard(selected_pair)

if __name__ == "__main__":
    main()