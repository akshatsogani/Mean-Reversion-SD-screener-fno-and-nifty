# sd_screener.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Constants
SD_PERIODS = 8
SD_LOOKBACK_DAILY = 3 * 252 + 100
SD_LOOKBACK_HOURLY = 60
SD_BANDS = [1.5, 2.0, 2.5]
MAX_WORKERS = 10
CACHE_TTL = 3600

# Page Configuration
st.set_page_config(
    page_title="SD Screener | Mean Reversion Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background: #0E1117;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00D9FF 0%, #0099FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #8B92A5;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .config-card {
        background: #262730;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background: #262730;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
    }
    
    .stock-card {
        background: #262730;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.2s;
        border: 1px solid #333;
    }
    
    .stock-card:hover {
        border-color: #00D9FF;
        box-shadow: 0 6px 12px rgba(0,217,255,0.2);
    }
    
    .signal-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .signal-long {
        background: #10B981;
        color: white;
    }
    
    .signal-short {
        background: #EF4444;
        color: white;
    }
    
    .signal-wait {
        background: #F59E0B;
        color: white;
    }
    
    .signal-breakout {
        background: #3B82F6;
        color: white;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FAFAFA;
        font-family: 'Monaco', monospace;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #8B92A5;
        margin-top: 0.5rem;
    }
    
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #00D9FF 0%, #0099FF 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
        width: 100%;
    }
    
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,217,255,0.4);
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #00D9FF, #0099FF);
        border-radius: 4px;
        height: 4px;
    }
</style>
""", unsafe_allow_html=True)

def load_fno_stocks():
    """Load FNO stocks from Excel file"""
    try:
        df = pd.read_excel('FNO_Stock.xlsx')
        if 'Symbol' in df.columns:
            symbols = df['Symbol'].dropna().tolist()
            return [f"{symbol}.NS" for symbol in symbols]
        else:
            st.error("Excel file must have a 'Symbol' column")
            return []
    except FileNotFoundError:
        st.warning("FNO_Stock.xlsx not found. Using Nifty Index only.")
        return []
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return []

@st.cache_data(ttl=CACHE_TTL)
def fetch_data(symbol, timeframe):
    """Fetch historical data for a symbol"""
    try:
        if timeframe == '1D':
            period = f"{SD_LOOKBACK_DAILY}d"
            interval = '1d'
        else:
            period = '60d'
            interval = '1h'
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty or len(df) < SD_PERIODS:
            return None
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except:
        return None

def calculate_indicators(df):
    """Calculate mean and SD bands"""
    if df is None or len(df) < SD_PERIODS:
        return None
    
    df = df.copy()
    df['Mean'] = df['Close'].rolling(window=SD_PERIODS).mean()
    
    # Calculate rolling standard deviation
    rolling_std = df['Close'].rolling(window=len(df)).std()
    current_std = rolling_std.iloc[-1]
    
    # Calculate SD bands
    for band in SD_BANDS:
        df[f'Upper_{band}'] = df['Mean'] + (band * current_std)
        df[f'Lower_{band}'] = df['Mean'] - (band * current_std)
    
    # Calculate current position in SD
    current_price = df['Close'].iloc[-1]
    current_mean = df['Mean'].iloc[-1]
    
    if pd.notna(current_std) and current_std > 0:
        df['SD_Position'] = (df['Close'] - df['Mean']) / current_std
    else:
        df['SD_Position'] = 0
    
    return df

def determine_signal(df):
    """Determine trading signal based on SD position"""
    if df is None or df.empty:
        return 'NO_DATA'
    
    current_sd = df['SD_Position'].iloc[-1]
    recent_sd = df['SD_Position'].tail(3).values
    
    # Check for extreme zones
    if current_sd >= 2.5:
        return 'EXTREME_SHORT'
    elif current_sd <= -2.5:
        return 'EXTREME_LONG'
    
    # Ready to trade signals
    elif current_sd >= 1.5:
        return 'SHORT'
    elif current_sd <= -1.5:
        return 'LONG'
    
    # Check for recent breakouts
    elif any(abs(sd) >= 1.5 for sd in recent_sd[:-1]) and abs(current_sd) < 1.5:
        return 'BREAKOUT'
    
    # Consolidation
    elif abs(current_sd) <= 0.5:
        return 'WAIT'
    
    # Trending
    elif current_sd > 0.5:
        return 'BULLISH_TREND'
    else:
        return 'BEARISH_TREND'

def calculate_metrics(df, symbol):
    """Calculate all metrics for a stock"""
    if df is None or df.empty:
        return None
    
    current_price = df['Close'].iloc[-1]
    current_mean = df['Mean'].iloc[-1]
    current_sd = df['SD_Position'].iloc[-1]
    
    # Distance from mean
    distance_points = current_price - current_mean
    distance_pct = (distance_points / current_mean) * 100
    
    # Stop loss at 2.5 SD
    if current_sd > 0:
        stop_loss = df['Upper_2.5'].iloc[-1]
    else:
        stop_loss = df['Lower_2.5'].iloc[-1]
    
    # Risk reward calculation
    potential_profit = abs(distance_points)
    risk = abs(current_price - stop_loss)
    rr_ratio = potential_profit / risk if risk > 0 else 0
    
    # Days in current zone
    zone_days = 0
    for i in range(len(df)-1, -1, -1):
        if abs(df['SD_Position'].iloc[i]) >= 1.5:
            zone_days += 1
        else:
            break
    
    signal = determine_signal(df)
    
    return {
        'symbol': symbol,
        'price': current_price,
        'mean': current_mean,
        'sd_position': current_sd,
        'distance_points': distance_points,
        'distance_pct': distance_pct,
        'stop_loss': stop_loss,
        'rr_ratio': rr_ratio,
        'zone_days': zone_days,
        'signal': signal,
        'df': df
    }

def create_sparkline(df):
    """Create mini sparkline chart"""
    if df is None or len(df) < 20:
        return None
    
    df_mini = df.tail(30)
    
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=df_mini.index,
        y=df_mini['Close'],
        mode='lines',
        line=dict(color='#00D9FF', width=2),
        showlegend=False,
        hovertemplate='%{y:.2f}<extra></extra>'
    ))
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=df_mini.index,
        y=df_mini['Mean'],
        mode='lines',
        line=dict(color='#F59E0B', width=1, dash='dot'),
        showlegend=False,
        hovertemplate='Mean: %{y:.2f}<extra></extra>'
    ))
    
    # Update layout for compact view
    fig.update_layout(
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode='x'
    )
    
    return fig

def create_detailed_chart(df, symbol):
    """Create detailed interactive chart"""
    if df is None or df.empty:
        return None
    
    df_chart = df.tail(90)
    
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart['Open'],
        high=df_chart['High'],
        low=df_chart['Low'],
        close=df_chart['Close'],
        name='Price',
        increasing_line_color='#10B981',
        decreasing_line_color='#EF4444'
    ))
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=df_chart.index,
        y=df_chart['Mean'],
        mode='lines',
        name='Mean (8 SMA)',
        line=dict(color='#F59E0B', width=2)
    ))
    
    # Add SD bands
    colors = ['#EF4444', '#10B981', '#8B43F6']
    for i, band in enumerate(SD_BANDS):
        fig.add_trace(go.Scatter(
            x=df_chart.index,
            y=df_chart[f'Upper_{band}'],
            mode='lines',
            name=f'+{band} SD',
            line=dict(color=colors[i], width=1, dash='dash'),
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=df_chart.index,
            y=df_chart[f'Lower_{band}'],
            mode='lines',
            name=f'-{band} SD',
            line=dict(color=colors[i], width=1, dash='dash'),
            opacity=0.5
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - Standard Deviation Analysis",
        height=500,
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        legend=dict(x=0, y=1)
    )
    
    return fig

def screen_stocks(universe, timeframe, filters=None):
    """Main screening function"""
    results = []
    symbols = []
    
    # Determine symbols based on universe
    if universe == 'Nifty Index Only':
        symbols = ['^NSEI']
    elif universe == 'FNO Stocks Only':
        symbols = load_fno_stocks()
    else:  # Both
        symbols = ['^NSEI'] + load_fno_stocks()
    
    if not symbols:
        return []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    failed_count = 0
    
    # Fetch and analyze data
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_data, symbol, timeframe): symbol 
                  for symbol in symbols}
        
        completed = 0
        for future in as_completed(futures):
            symbol = futures[future]
            completed += 1
            
            progress = completed / len(symbols)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing {completed}/{len(symbols)} stocks...")
            
            try:
                df = future.result()
                if df is not None:
                    df = calculate_indicators(df)
                    metrics = calculate_metrics(df, symbol)
                    if metrics:
                        results.append(metrics)
                else:
                    failed_count += 1
            except:
                failed_count += 1
    
    progress_bar.empty()
    status_text.empty()
    
    if failed_count > 0:
        st.info(f"ℹ️ {failed_count} symbols failed to fetch or had insufficient data")
    
    # Apply filters if specified
    if filters and 'signal_filter' in filters:
        signal_filter = filters['signal_filter']
        if signal_filter == 'Long Only':
            results = [r for r in results if 'LONG' in r['signal']]
        elif signal_filter == 'Short Only':
            results = [r for r in results if 'SHORT' in r['signal']]
        elif signal_filter == 'Extreme Zones Only':
            results = [r for r in results if 'EXTREME' in r['signal']]
    
    # Sort results
    if filters and 'sort_by' in filters:
        sort_by = filters['sort_by']
        if sort_by == 'Distance from Mean ↓':
            results.sort(key=lambda x: abs(x['distance_pct']), reverse=True)
        elif sort_by == 'Risk-Reward ↓':
            results.sort(key=lambda x: x['rr_ratio'], reverse=True)
        elif sort_by == 'Days in Zone ↓':
            results.sort(key=lambda x: x['zone_days'], reverse=True)
    
    return results

def render_config_screen():
    """Render initial configuration screen"""
    st.markdown('<h1 class="main-header">Standard Deviation Screener</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Institutional-Grade Mean Reversion Analytics</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.markdown('<div class="config-card">', unsafe_allow_html=True)
            
            st.markdown("### 📊 Universe Selection")
            universe = st.radio(
                "",
                options=['Nifty Index Only', 'FNO Stocks Only', 'Both Nifty + FNO Stocks'],
                index=2,
                label_visibility='collapsed'
            )
            
            st.markdown("### ⏰ Timeframe")
            timeframe = st.radio(
                "",
                options=['Daily (1D)', 'Hourly (1H)'],
                index=0,
                label_visibility='collapsed'
            )
            timeframe_code = '1D' if 'Daily' in timeframe else '1H'
            
            st.markdown("### 🎯 Additional Filters (Optional)")
            col_a, col_b = st.columns(2)
            with col_a:
                signal_filter = st.selectbox(
                    "Signal Filter",
                    options=['All Signals', 'Long Only', 'Short Only', 'Extreme Zones Only']
                )
            with col_b:
                sort_by = st.selectbox(
                    "Sort By",
                    options=['Distance from Mean ↓', 'Risk-Reward ↓', 'Days in Zone ↓']
                )
            
            st.markdown("---")
            
            num_stocks = 1 if universe == 'Nifty Index Only' else len(load_fno_stocks()) if universe == 'FNO Stocks Only' else len(load_fno_stocks()) + 1
            st.markdown(f"<p style='text-align:center; color:#8B92A5;'>This will analyze {num_stocks} stocks/index based on your selection</p>", unsafe_allow_html=True)
            
            if st.button("🚀 RUN ANALYSIS", use_container_width=True):
                st.session_state['ran'] = True
                st.session_state['universe'] = universe
                st.session_state['timeframe'] = timeframe_code
                st.session_state['filters'] = {
                    'signal_filter': signal_filter,
                    'sort_by': sort_by
                }
                
                with st.spinner("Initializing analysis engine..."):
                    time.sleep(1)
                    results = screen_stocks(universe, timeframe_code, st.session_state['filters'])
                    st.session_state['results'] = results
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

def render_stock_card(metrics):
    """Render individual stock card"""
    signal_class = {
        'LONG': 'signal-long',
        'EXTREME_LONG': 'signal-long',
        'SHORT': 'signal-short',
        'EXTREME_SHORT': 'signal-short',
        'WAIT': 'signal-wait',
        'BREAKOUT': 'signal-breakout',
        'BULLISH_TREND': 'signal-wait',
        'BEARISH_TREND': 'signal-wait'
    }.get(metrics['signal'], 'signal-wait')
    
    signal_text = metrics['signal'].replace('_', ' ').title()
    
    with st.container():
        st.markdown('<div class="stock-card">', unsafe_allow_html=True)
        
        # Header
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"### {metrics['symbol']}")
            st.markdown(f"**₹{metrics['price']:.2f}**")
        with col2:
            st.markdown(f'<span class="signal-badge {signal_class}">{signal_text}</span>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"₹{metrics['mean']:.2f}")
        with col2:
            st.metric("SD Position", f"{metrics['sd_position']:.2f} SD")
        with col3:
            st.metric("Distance", f"{metrics['distance_pct']:.1f}%")
        
        # Visual SD position bar
        sd_pos = metrics['sd_position']
        bar_position = (sd_pos + 2.5) / 5.0 * 100  # Normalize to 0-100%
        bar_position = max(0, min(100, bar_position))
        
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #10B981 0%, #F59E0B 50%, #EF4444 100%); 
                    height: 8px; border-radius: 4px; position: relative; margin: 1rem 0;">
            <div style="position: absolute; left: {bar_position}%; top: -4px; 
                        width: 16px; height: 16px; background: white; 
                        border-radius: 50%; border: 2px solid #00D9FF;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sparkline
        fig = create_sparkline(metrics['df'])
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Action row
        col1, col2, col3 = st.columns(3)
        with col1:
            direction = "↓" if sd_pos > 0 else "↑"
            st.markdown(f"**Expected:** {direction} {abs(metrics['distance_points']):.0f} pts")
        with col2:
            st.markdown(f"**R:R:** 1:{metrics['rr_ratio']:.1f}")
        with col3:
            st.markdown(f"**Zone Days:** {metrics['zone_days']}")
        
        # Expand button
        with st.expander("📊 View Detailed Analysis"):
            render_detailed_view(metrics)
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_detailed_view(metrics):
    """Render detailed view for a stock"""
    # Chart
    fig = create_detailed_chart(metrics['df'], metrics['symbol'])
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Price Metrics")
        st.write(f"**Current Price:** ₹{metrics['price']:.2f}")
        st.write(f"**Mean (8 SMA):** ₹{metrics['mean']:.2f}")
        st.write(f"**Distance to Mean:** ₹{abs(metrics['distance_points']):.2f} ({abs(metrics['distance_pct']):.1f}%)")
        st.write(f"**Current Position:** {metrics['sd_position']:.2f} SD")
    
    with col2:
        st.markdown("#### Risk Management")
        st.write(f"**Stop-Loss Level:** ₹{metrics['stop_loss']:.2f}")
        st.write(f"**Risk-Reward Ratio:** 1:{metrics['rr_ratio']:.2f}")
        st.write(f"**Days in Zone:** {metrics['zone_days']} days")
        st.write(f"**Signal:** {metrics['signal'].replace('_', ' ').title()}")

def render_dashboard(results):
    """Render main dashboard"""
    st.markdown('<h1 class="main-header">Standard Deviation Screener</h1>', unsafe_allow_html=True)
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_analyzed = len(results)
    long_count = len([r for r in results if 'LONG' in r['signal']])
    short_count = len([r for r in results if 'SHORT' in r['signal']])
    extreme_count = len([r for r in results if 'EXTREME' in r['signal']])
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{total_analyzed}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Analyzed</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card" style="border-left: 3px solid #10B981;">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{long_count}</div>', unsafe_allow_html=True)
        pct = (long_count/total_analyzed*100) if total_analyzed > 0 else 0
        st.markdown(f'<div class="metric-label">Ready to LONG ({pct:.0f}%)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card" style="border-left: 3px solid #EF4444;">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{short_count}</div>', unsafe_allow_html=True)
        pct = (short_count/total_analyzed*100) if total_analyzed > 0 else 0
        st.markdown(f'<div class="metric-label">Ready to SHORT ({pct:.0f}%)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card" style="border-left: 3px solid #8B43F6;">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{extreme_count}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Extreme Opportunities</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("### 🎯 Quick Filters")
        
        signal_type = st.selectbox(
            "Signal Type",
            options=['All', 'Long Signals', 'Short Signals', 'Extreme Only', 'Breakouts']
        )
        
        min_rr = st.slider("Min Risk-Reward", 0.0, 5.0, 0.0, 0.5)
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.session_state['results'] = screen_stocks(
                st.session_state['universe'],
                st.session_state['timeframe'],
                st.session_state['filters']
            )
            st.rerun()
        
        if st.button("⚙️ Reset Configuration"):
            for key in ['ran', 'results', 'universe', 'timeframe', 'filters']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Filter results based on sidebar
    filtered_results = results.copy()
    
    if signal_type == 'Long Signals':
        filtered_results = [r for r in filtered_results if 'LONG' in r['signal']]
    elif signal_type == 'Short Signals':
        filtered_results = [r for r in filtered_results if 'SHORT' in r['signal']]
    elif signal_type == 'Extreme Only':
        filtered_results = [r for r in filtered_results if 'EXTREME' in r['signal']]
    elif signal_type == 'Breakouts':
        filtered_results = [r for r in filtered_results if r['signal'] == 'BREAKOUT']
    
    filtered_results = [r for r in filtered_results if r['rr_ratio'] >= min_rr]
    
    # Results grid
    if filtered_results:
        st.markdown("### 📊 Screening Results")
        
        # Responsive columns
        num_cols = 3 if st.session_state.get('wide_mode', True) else 2
        cols = st.columns(num_cols)
        
        for idx, result in enumerate(filtered_results[:50]):  # Limit to 50 for performance
            with cols[idx % num_cols]:
                render_stock_card(result)
    else:
        st.info("No stocks match your filter criteria. Try adjusting the filters.")
    
    # Summary insights
    st.markdown("---")
    st.markdown("### 💡 Market Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        overbought = len([r for r in results if r['sd_position'] > 1.5])
        oversold = len([r for r in results if r['sd_position'] < -1.5])
        neutral = total_analyzed - overbought - oversold
        
        st.markdown(f"""
        **Market Overview:**  
        {overbought} stocks overbought  
        {oversold} stocks oversold  
        {neutral} stocks neutral
        """)
    
    with col2:
        top_rr = sorted(results, key=lambda x: x['rr_ratio'], reverse=True)[:3]
        st.markdown("**Top Risk-Reward Setups:**")
        for stock in top_rr:
            st.write(f"• {stock['symbol']}: 1:{stock['rr_ratio']:.1f}")
    
    with col3:
        extreme_stocks = [r for r in results if abs(r['sd_position']) >= 2.5]
        st.markdown("**⚠️ Extreme Zones:**")
        for stock in extreme_stocks[:3]:
            st.write(f"• {stock['symbol']}: {stock['sd_position']:.1f} SD")
    
    # Footer
    st.markdown("---")
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M IST")
    st.markdown(f"<p style='text-align:center; color:#8B92A5;'>Last updated: {timestamp}</p>", unsafe_allow_html=True)

def main():
    """Main application flow"""
    if not st.session_state.get('ran', False):
        render_config_screen()
    else:
        if 'results' in st.session_state:
            render_dashboard(st.session_state['results'])
        else:
            st.error("No results found. Please run analysis again.")
            if st.button("Back to Configuration"):
                st.session_state['ran'] = False
                st.rerun()

if __name__ == "__main__":
    main()
