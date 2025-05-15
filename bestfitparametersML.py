import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Buffer Fitting", page_icon="🤖", layout="wide")

# --- UI Setup ---
st.title("Machine Learning Buffer Fitting Tool")

# DB connection
db_params = {
    'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
    'port': 5432,
    'database': 'replication_report',
    'user': 'public_replication',
    'password': '866^FKC4hllk'
}

engine = create_engine(
    f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
)

# Cache token list
@st.cache_data(ttl=3600)
def fetch_trading_pairs():
    query = """
    SELECT pair_name 
    FROM trade_pool_pairs 
    WHERE status = 1
    ORDER BY pair_name
    """
    df = pd.read_sql_query(query, engine)
    return df['pair_name'].tolist()

# Get volatility and rollbit data
@st.cache_data(ttl=60)
def get_combined_data(token):
    hours_to_fetch = 24
    now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
    start_time_sg = now_sg - timedelta(hours=hours_to_fetch)
    
    # Get volatility data
    today_str = now_sg.strftime("%Y%m%d")
    yesterday_str = (now_sg - timedelta(days=1)).strftime("%Y%m%d")
    start_time_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    
    # Volatility query
    vol_query = f"""
    SELECT 
        created_at + INTERVAL '8 hour' AS timestamp,
        final_price
    FROM public.oracle_price_log_partition_{today_str}
    WHERE created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
    AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
    AND source_type = 0
    AND pair_name = '{token}'
    ORDER BY timestamp
    """
    
    try:
        price_df = pd.read_sql_query(vol_query, engine)
        
        # If no data, try yesterday's partition
        if price_df.empty:
            vol_query_yesterday = f"""
            SELECT 
                created_at + INTERVAL '8 hour' AS timestamp,
                final_price
            FROM public.oracle_price_log_partition_{yesterday_str}
            WHERE created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
            AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
            AND source_type = 0
            AND pair_name = '{token}'
            ORDER BY timestamp
            """
            price_df = pd.read_sql_query(vol_query_yesterday, engine)
        
        if price_df.empty:
            return None
        
        # Process price data
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        price_df = price_df.set_index('timestamp').sort_index()
        
        # Resample to 500ms
        price_data = price_df['final_price'].resample('500ms').ffill().dropna()
        
        # Calculate 10-second volatility
        result = []
        start_date = price_data.index.min().floor('10s')
        end_date = price_data.index.max().ceil('10s')
        ten_sec_periods = pd.date_range(start=start_date, end=end_date, freq='10s')
        
        for i in range(len(ten_sec_periods)-1):
            start_window = ten_sec_periods[i]
            end_window = ten_sec_periods[i+1]
            
            window_data = price_data[(price_data.index >= start_window) & (price_data.index < end_window)]
            
            if len(window_data) >= 2:
                log_returns = np.diff(np.log(window_data.values))
                if len(log_returns) > 0:
                    annualization_factor = np.sqrt(3153600)
                    volatility = np.std(log_returns) * annualization_factor * 100  # Convert to percentage
                    
                    result.append({
                        'timestamp': start_window,
                        'volatility': volatility
                    })
        
        vol_df = pd.DataFrame(result).set_index('timestamp')
        
    except Exception as e:
        st.error(f"Error getting volatility: {e}")
        return None
    
    # Get Rollbit data
    rollbit_query = f"""
    SELECT 
        pair_name,
        bust_buffer * 100 AS buffer_rate,  -- Convert to percentage
        position_multiplier,
        created_at + INTERVAL '8 hour' AS timestamp
    FROM rollbit_pair_config 
    WHERE pair_name = '{token}'
    AND created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
    AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
    ORDER BY created_at
    """
    
    try:
        rollbit_df = pd.read_sql_query(rollbit_query, engine)
        if rollbit_df.empty:
            return None
            
        rollbit_df['timestamp'] = pd.to_datetime(rollbit_df['timestamp'])
        rollbit_df = rollbit_df.set_index('timestamp').sort_index()
        rollbit_df = rollbit_df.resample('10s').ffill()
        
    except Exception as e:
        st.error(f"Error getting Rollbit data: {e}")
        return None
    
    # Merge data
    combined = pd.merge(vol_df, rollbit_df, left_index=True, right_index=True, how='inner')
    
    return combined

def create_features(volatility_data):
    """Create various features from volatility data"""
    features = pd.DataFrame()
    
    # Basic features
    features['vol'] = volatility_data
    features['vol_squared'] = volatility_data ** 2
    features['vol_cubed'] = volatility_data ** 3
    features['vol_sqrt'] = np.sqrt(volatility_data)
    features['vol_log'] = np.log(volatility_data + 1)
    
    # Reciprocal features
    features['vol_reciprocal'] = 1 / (volatility_data + 0.01)
    
    # Exponential features
    features['vol_exp'] = np.exp(volatility_data / 100)  # Scale down to avoid overflow
    
    # Trigonometric features (normalized volatility)
    vol_norm = volatility_data / 100
    features['vol_sin'] = np.sin(vol_norm)
    features['vol_cos'] = np.cos(vol_norm)
    
    # Rolling features
    vol_series = pd.Series(volatility_data)
    features['vol_rolling_mean_5'] = vol_series.rolling(5, min_periods=1).mean().values
    features['vol_rolling_std_5'] = vol_series.rolling(5, min_periods=1).std().fillna(0).values
    features['vol_rolling_mean_10'] = vol_series.rolling(10, min_periods=1).mean().values
    features['vol_rolling_std_10'] = vol_series.rolling(10, min_periods=1).std().fillna(0).values
    
    # Lagged features
    features['vol_lag_1'] = vol_series.shift(1).fillna(method='bfill').values
    features['vol_lag_5'] = vol_series.shift(5).fillna(method='bfill').values
    
    # Difference features
    features['vol_diff_1'] = vol_series.diff().fillna(0).values
    features['vol_diff_5'] = vol_series.diff(5).fillna(0).values
    
    # Percentile features (calculate on the fly)
    percentiles = [25, 50, 75, 95]
    for p in percentiles:
        threshold = np.percentile(volatility_data, p)
        features[f'above_p{p}'] = (volatility_data > threshold).astype(int)
    
    return features

def train_multiple_models(X_train, y_train, X_test, y_test):
    """Train multiple ML models and return results"""
    results = {}
    
    models = {
        'Linear Regression': LinearRegression(),
        'Polynomial (deg=2)': make_pipeline(PolynomialFeatures(2), LinearRegression()),
        'Polynomial (deg=3)': make_pipeline(PolynomialFeatures(3), LinearRegression()),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.001),
        'Huber Regression': HuberRegressor(),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
        'SVR (RBF)': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'SVR (Linear)': SVR(kernel='linear', C=1.0),
        'Neural Network': MLPRegressor(hidden_layers_sizes=(100, 50), max_iter=500, random_state=42),
        'Gaussian Process': GaussianProcessRegressor(kernel=RBF(length_scale=1.0), random_state=42)
    }
    
    # Scale features for certain models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in models.items():
        try:
            # Use scaled features for certain models
            if name in ['SVR (RBF)', 'SVR (Linear)', 'Neural Network', 'Gaussian Process']:
                model.fit(X_train_scaled, y_train)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'mae': test_mae,
                'rmse': test_rmse,
                'predictions': y_pred_test,
                'scaler': scaler if name in ['SVR (RBF)', 'SVR (Linear)', 'Neural Network', 'Gaussian Process'] else None
            }
            
        except Exception as e:
            st.warning(f"Failed to train {name}: {str(e)}")
    
    return results

def make_pipeline(preprocessor, model):
    """Create a simple pipeline"""
    class SimplePipeline:
        def __init__(self, preprocessor, model):
            self.preprocessor = preprocessor
            self.model = model
        
        def fit(self, X, y):
            X_transformed = self.preprocessor.fit_transform(X)
            self.model.fit(X_transformed, y)
            return self
        
        def predict(self, X):
            X_transformed = self.preprocessor.transform(X)
            return self.model.predict(X_transformed)
    
    return SimplePipeline(preprocessor, model)

# Main UI
all_tokens = fetch_trading_pairs()

col1, col2 = st.columns([3, 1])

with col1:
    selected_token = st.selectbox(
        "Select Token",
        all_tokens,
        index=all_tokens.index("BTC/USDT") if "BTC/USDT" in all_tokens else 0
    )

with col2:
    if st.button("Analyze All Tokens", type="primary"):
        st.session_state.analyze_all = True

# Single token analysis
with st.spinner(f"Loading data for {selected_token}..."):
    data = get_combined_data(selected_token)

if data is not None and len(data) > 0:
    # Use last 1000 points
    recent_data = data.tail(1000)
    
    # Extract volatility and buffer
    volatilities = recent_data['volatility'].values
    buffers = recent_data['buffer_rate'].values
    
    # Create features
    features = create_features(volatilities)
    
    # Split data
    split_idx = int(0.8 * len(features))
    X_train = features.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_train = buffers[:split_idx]
    y_test = buffers[split_idx:]
    
    # Train models
    with st.spinner("Training multiple ML models..."):
        results = train_multiple_models(X_train, y_train, X_test, y_test)
    
    # Sort by test R²
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
    
    # Display results
    st.markdown(f"### {selected_token} - ML Model Results")
    
    # Best model
    best_model_name, best_model_data = sorted_results[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Model", best_model_name)
    with col2:
        st.metric("Test R²", f"{best_model_data['test_r2']:.3f}")
    with col3:
        st.metric("Train R²", f"{best_model_data['train_r2']:.3f}")
    with col4:
        st.metric("RMSE", f"{best_model_data['rmse']:.4f}%")
    
    # Model comparison table
    st.markdown("### Model Comparison")
    
    comparison_df = pd.DataFrame([
        {
            'Model': name,
            'Train R²': data['train_r2'],
            'Test R²': data['test_r2'],
            'MAE (%)': data['mae'],
            'RMSE (%)': data['rmse']
        }
        for name, data in sorted_results
    ])
    
    st.dataframe(
        comparison_df.style.format({
            'Train R²': '{:.3f}',
            'Test R²': '{:.3f}',
            'MAE (%)': '{:.4f}',
            'RMSE (%)': '{:.4f}'
        }).background_gradient(subset=['Test R²'], cmap='RdYlGn')
    )
    
    # Visualization
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f"Best Model ({best_model_name}) Predictions",
            "Prediction Error"
        )
    )
    
    # Actual vs Predicted
    test_indices = recent_data.index[split_idx:]
    
    fig.add_trace(
        go.Scatter(
            x=test_indices,
            y=y_test,
            mode='lines',
            name='Actual Buffer',
            line=dict(color='green', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=test_indices,
            y=best_model_data['predictions'],
            mode='lines',
            name=f'Predicted ({best_model_name})',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Error plot
    error = y_test - best_model_data['predictions']
    fig.add_trace(
        go.Scatter(
            x=test_indices,
            y=error,
            mode='lines',
            name='Error',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=700,
        title_text=f"{selected_token} - ML Model Performance"
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Buffer Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Error (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance for tree-based models
    if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Decision Tree']:
        st.markdown("### Feature Importance")
        
        if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
            importances = best_model_data['model'].feature_importances_
        else:  # XGBoost
            importances = best_model_data['model'].feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'Feature': features.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        fig_importance = go.Figure(go.Bar(
            x=feature_importance_df['Importance'][:10],
            y=feature_importance_df['Feature'][:10],
            orientation='h'
        ))
        fig_importance.update_layout(
            title="Top 10 Feature Importances",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400
        )
        st.plotly_chart(fig_importance, use_container_width=True)

else:
    st.error("No data available for the selected token")

# Analyze all tokens
if st.session_state.get('analyze_all', False):
    st.markdown("---")
    st.markdown("### Analyzing All Tokens")
    
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, token in enumerate(all_tokens):
        status_text.text(f"Processing {token}... ({i+1}/{len(all_tokens)})")
        progress_bar.progress((i + 1) / len(all_tokens))
        
        try:
            data = get_combined_data(token)
            
            if data is not None and len(data) > 100:
                recent_data = data.tail(1000)
                volatilities = recent_data['volatility'].values
                buffers = recent_data['buffer_rate'].values
                
                # Create features
                features = create_features(volatilities)
                
                # Split data
                split_idx = int(0.8 * len(features))
                X_train = features.iloc[:split_idx]
                X_test = features.iloc[split_idx:]
                y_train = buffers[:split_idx]
                y_test = buffers[split_idx:]
                
                # Train models
                results = train_multiple_models(X_train, y_train, X_test, y_test)
                
                # Get best model
                best_model = max(results.items(), key=lambda x: x[1]['test_r2'])
                
                all_results.append({
                    'Token': token,
                    'Best Model': best_model[0],
                    'Test R²': best_model[1]['test_r2'],
                    'Train R²': best_model[1]['train_r2'],
                    'RMSE (%)': best_model[1]['rmse'],
                    'Data Points': len(recent_data)
                })
                
        except Exception as e:
            all_results.append({
                'Token': token,
                'Best Model': 'Error',
                'Test R²': np.nan,
                'Train R²': np.nan,
                'RMSE (%)': np.nan,
                'Data Points': 0,
                'Error': str(e)
            })
    
    status_text.text("Analysis complete!")
    progress_bar.progress(1.0)
    
    # Display results
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('Test R²', ascending=False)
    
    st.markdown("### All Tokens - Best ML Models")
    st.dataframe(
        results_df.style.format({
            'Test R²': '{:.3f}',
            'Train R²': '{:.3f}',
            'RMSE (%)': '{:.4f}',
            'Data Points': '{:.0f}'
        }).background_gradient(subset=['Test R²'], cmap='RdYlGn')
    )
    
    # Summary statistics
    st.markdown("### Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_r2 = results_df['Test R²'].mean()
        st.metric("Average Test R²", f"{avg_r2:.3f}")
    
    with col2:
        high_r2 = len(results_df[results_df['Test R²'] > 0.8])
        st.metric("High R² (>0.8)", f"{high_r2}/{len(results_df)}")
    
    with col3:
        med_r2 = results_df['Test R²'].median()
        st.metric("Median Test R²", f"{med_r2:.3f}")
    
    with col4:
        best_overall = results_df.iloc[0]
        st.metric("Best Overall", f"{best_overall['Token']} ({best_overall['Test R²']:.3f})")
    
    # Model frequency
    model_counts = results_df['Best Model'].value_counts()
    
    fig_models = go.Figure(go.Bar(
        x=model_counts.values,
        y=model_counts.index,
        orientation='h'
    ))
    fig_models.update_layout(
        title="Most Frequently Best Models",
        xaxis_title="Count",
        yaxis_title="Model",
        height=400
    )
    st.plotly_chart(fig_models, use_container_width=True)
    
    st.session_state.analyze_all = False