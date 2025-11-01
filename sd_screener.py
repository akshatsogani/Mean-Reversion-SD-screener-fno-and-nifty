import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="SD Mean Reversion Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Professional Styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1e88e5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .dataframe {
        font-size: 14px;
    }
    .signal-long {
        background-color: #4caf50;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .signal-short {
        background-color: #f44336;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .signal-wait {
        background-color: #ffc107;
        color: black;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    h1 {
        text-align: center;
        color: #1e88e5;
    }
    h3 {
        color: #424242;
    }
</style>
""", unsafe_allow_html=True)

# Constants
SD_LOOKBACK = 756  # 3 years of trading days
MEAN_PERIOD = 8
SD_MULTIPLIERS = [1.5, 2.0, 2.5]

@st.cache_data(ttl=3600)
def load_fno_symbols() -> List[str]:
    """Load FNO stock symbols from Excel file."""
    try:
        df = pd.read_excel("FNO_Stock.xlsx")
        if 'Symbol' not in df.columns:
            st.error("❌ Excel file must have a 'Symbol' column")
            return []
        symbols = df['Symbol'].dropna().tolist()
        return [f"{symbol}.NS" for symbol in symbols if symbol]
    except FileNotFoundError:
        st.error("❌ FNO_Stock.xlsx not found. Please ensure the file is in the root directory.")
        return []
    except Exception as e:
        st.error(f"❌ Error loading FNO symbols: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch historical data for a single stock."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="4y", interval="1d")
        if len(df) < SD_LOOKBACK + 1:
            return None
        return df[['Close']]
    except:
        return None

def calculate_sd_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate SD bands using exact methodology."""
    if len(df) < SD_LOOKBACK + 1:
        return df
    
    # Get last 3 years + 1 day for difference calculation
    lookback_df = df.tail(SD_LOOKBACK + 1).copy()
    
    # Calculate close-to-close differences
    differences = lookback_df['Close'].diff().dropna()
    
    # Calculate SD using population formula (ddof=0)
    sd_value = np.std(differences.values, ddof=0)
    
    # Calculate 8-period SMA as mean
    df['Mean'] = df['Close'].rolling(window=MEAN_PERIOD).mean()
    
    # Store SD value
    df['SD_Value'] = sd_value
    
    # Calculate all SD bands
    df['Upper_1.5SD'] = df['Mean'] + (1.5 * sd_value)
    df['Upper_2SD'] = df['Mean'] + (2.0 * sd_value)
    df['Upper_2.5SD'] = df['Mean'] + (2.5 * sd_value)
    df['Lower_1.5SD'] = df['Mean'] - (1.5 * sd_value)
    df['Lower_2SD'] = df['Mean'] - (2.0 * sd_value)
    df['Lower_2.5SD'] = df['Mean'] - (2.5 * sd_value)
    
    # Calculate current SD position
    df['Current_SD_Position'] = (df['Close'] - df['Mean']) / sd_value
    
    return df

def determine_signal(df: pd.DataFrame) -> str:
    """Determine trading signal based on SD position."""
    if len(df) < 2:
        return "WAIT"
    
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    current_sd = latest['Current_SD_Position']
    prev_sd = previous['Current_SD_Position']
    
    if current_sd > 2.5:
        return "EXTREME SHORT"
    elif current_sd < -2.5:
        return "EXTREME LONG"
    elif current_sd < 1.5 and prev_sd >= 1.5:
        return "SHORT"
    elif current_sd > -1.5 and prev_sd <= -1.5:
        return "LONG"
    elif current_sd > 1.5:
        return "SHORT ZONE"
    elif current_sd < -1.5:
        return "LONG ZONE"
    else:
        return "WAIT"

def calculate_days_in_zone(df: pd.DataFrame) -> int:
    """Count consecutive days where abs(Current_SD_Position) > 1.5."""
    count = 0
    for i in range(len(df)-1, -1, -1):
        if abs(df.iloc[i]['Current_SD_Position']) > 1.5:
            count += 1
        else:
            break
    return count

def calculate_risk_reward(current_price: float, mean: float, current_sd: float, sd_value: float) -> float:
    """Calculate risk-reward ratio."""
    distance_to_mean = abs(current_price - mean)
    
    if current_sd > 0:  # Overbought - SHORT setup
        stop_loss_level = mean + (2.5 * sd_value)
        risk = abs(stop_loss_level - current_price)
    else:  # Oversold - LONG setup
        stop_loss_level = mean - (2.5 * sd_value)
        risk = abs(current_price - stop_loss_level)
    
    if risk == 0:
        return 0
    
    return distance_to_mean / risk

def process_single_stock(symbol: str) -> Optional[Dict[str, Any]]:
    """Process a single stock and return its metrics."""
    df = fetch_stock_data(symbol)
    if df is None or len(df) < SD_LOOKBACK + 1:
        return None
    
    df = calculate_sd_bands(df)
    if df['Mean'].iloc[-1] != df['Mean'].iloc[-1]:  # Check for NaN
        return None
    
    signal = determine_signal(df)
    
    # Extract metrics
    latest = df.iloc[-1]
    current_price = latest['Close']
    mean_value = latest['Mean']
    sd_value = latest['SD_Value']
    current_sd_pos = latest['Current_SD_Position']
    
    distance_to_mean = current_price - mean_value
    distance_pct = (distance_to_mean / current_price) * 100
    expected_move = -distance_to_mean
    potential_return = abs(distance_pct)
    
    days_in_zone = calculate_days_in_zone(df)
    rr_ratio = calculate_risk_reward(current_price, mean_value, current_sd_pos, sd_value)
    
    return {
        'Symbol': symbol.replace('.NS', ''),
        'Current Price': current_price,
        'Mean (8 SMA)': mean_value,
        'SD Value': sd_value,
        'Current Position': current_sd_pos,
        'Signal': signal,
        'Distance to Mean': distance_to_mean,
        'Distance %': distance_pct,
        'Days in Zone': days_in_zone,
        '+1.5 SD Level': latest['Upper_1.5SD'],
        '-1.5 SD Level': latest['Lower_1.5SD'],
        '+2.5 SD Level': latest['Upper_2.5SD'],
        '-2.5 SD Level': latest['Lower_2.5SD'],
        'Expected Move': expected_move,
        'Potential Return %': potential_return,
        'Risk-Reward': rr_ratio
    }

def screen_stocks(universe_choice: str, signal_filter: str, sort_by: str) -> List[Dict[str, Any]]:
    """Main screening function with parallel processing."""
    # Build stock list
    if universe_choice == "🔵 Nifty Index (^NSEI) Only":
        symbols = ["^NSEI"]
    elif universe_choice == "📈 FNO Stocks Only":
        symbols = load_fno_symbols()
    else:  # Both
        symbols = ["^NSEI"] + load_fno_symbols()
    
    if not symbols:
        return []
    
    results = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Parallel processing
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_single_stock, symbol): symbol for symbol in symbols}
        completed = 0
        
        for future in as_completed(futures):
            completed += 1
            symbol = futures[future]
            status_text.text(f"Analyzing {symbol}... ({completed}/{len(symbols)})")
            progress_bar.progress(completed / len(symbols))
            
            try:
                result = future.result(timeout=10)
                if result:
                    # Apply signal filter
                    if signal_filter == "All Signals":
                        results.append(result)
                    elif signal_filter == "Long Setups Only" and "LONG" in result['Signal']:
                        results.append(result)
                    elif signal_filter == "Short Setups Only" and "SHORT" in result['Signal']:
                        results.append(result)
                    elif signal_filter == "Extreme Zones Only" and "EXTREME" in result['Signal']:
                        results.append(result)
            except:
                continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Sort results
    if sort_by == "Distance from Mean (High to Low)":
        results.sort(key=lambda x: abs(x['Distance to Mean']), reverse=True)
    elif sort_by == "Current SD Position (Extreme to Neutral)":
        results.sort(key=lambda x: abs(x['Current Position']), reverse=True)
    elif sort_by == "Days in Current Zone (High to Low)":
        results.sort(key=lambda x: x['Days in Zone'], reverse=True)
    else:  # Symbol A-Z
        results.sort(key=lambda x: x['Symbol'])
    
    return results

def style_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply styling to the results dataframe."""
    def color_signal(val):
        colors = {
            'LONG': 'background-color: #4caf50; color: white',
            'EXTREME LONG': 'background-color: #2e7d32; color: white',
            'LONG ZONE': 'background-color: #81c784; color: white',
            'SHORT': 'background-color: #f44336; color: white',
            'EXTREME SHORT': 'background-color: #c62828; color: white',
            'SHORT ZONE': 'background-color: #ef5350; color: white',
            'WAIT': 'background-color: #ffc107; color: black'
        }
        return colors.get(val, '')
    
    def color_sd_position(val):
        try:
            num_val = float(val.replace(' SD', ''))
            if num_val > 1.5:
                return 'color: #c62828; font-weight: bold'
            elif num_val < -1.5:
                return 'color: #2e7d32; font-weight: bold'
            else:
                return 'color: #757575'
        except:
            return ''
    
    styled = df.style.applymap(color_signal, subset=['Signal'])
    styled = styled.applymap(color_sd_position, subset=['Current Position'])
    
    # Format columns
    styled = styled.format({
        'Current Price': '₹{:.2f}',
        'Mean (8 SMA)': '₹{:.2f}',
        'SD Value': '{:.2f} pts',
        'Current Position': '{:.2f} SD',
        'Distance to Mean': '{:.2f} pts ({:.2f}%)',
        'Days in Zone': '{} days',
        '+1.5 SD Level': '₹{:.2f}',
        '-1.5 SD Level': '₹{:.2f}',
        '+2.5 SD Level': '₹{:.2f}',
        '-2.5 SD Level': '₹{:.2f}',
        'Expected Move': '{:.2f} pts',
        'Potential Return %': '{:.2f}%',
        'Risk-Reward': '1:{:.1f}'
    })
    
    return styled

def export_to_excel(df: pd.DataFrame) -> Tuple[io.BytesIO, str]:
    """Export dataframe to Excel with formatting."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='SD Screener Results')
    
    output.seek(0)
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    filename = f"SD_Screener_Results_{timestamp}.xlsx"
    return output, filename

def export_to_csv(df: pd.DataFrame) -> Tuple[io.BytesIO, str]:
    """Export dataframe to CSV."""
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    filename = f"SD_Screener_Results_{timestamp}.csv"
    return output, filename

def render_summary_cards(results: List[Dict]) -> None:
    """Display summary metrics cards."""
    total_stocks = len(results)
    long_setups = len([r for r in results if 'LONG' in r['Signal']])
    short_setups = len([r for r in results if 'SHORT' in r['Signal']])
    extreme_zones = len([r for r in results if 'EXTREME' in r['Signal']])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Analyzed", f"{total_stocks} stocks")
    
    with col2:
        st.metric("📈 Long Setups", f"{long_setups} stocks", 
                 f"{(long_setups/total_stocks*100):.1f}%" if total_stocks > 0 else "0%")
    
    with col3:
        st.metric("📉 Short Setups", f"{short_setups} stocks",
                 f"{(short_setups/total_stocks*100):.1f}%" if total_stocks > 0 else "0%")
    
    with col4:
        st.metric("⚠️ Extreme Zones", f"{extreme_zones} stocks")

def create_stock_chart(symbol: str, df: pd.DataFrame) -> go.Figure:
    """Create a detailed chart for a stock."""
    recent_df = df.tail(90)
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=recent_df.index,
        y=recent_df['Close'],
        name='Price',
        line=dict(color='#1e88e5', width=2)
    ))
    
    # Mean line
    fig.add_trace(go.Scatter(
        x=recent_df.index,
        y=recent_df['Mean'],
        name='Mean (8 SMA)',
        line=dict(color='black', width=1.5, dash='dash')
    ))
    
    # SD bands
    colors = ['#ff9800', '#f44336', '#d32f2f']
    for i, mult in enumerate([1.5, 2.0, 2.5]):
        fig.add_trace(go.Scatter(
            x=recent_df.index,
            y=recent_df[f'Upper_{mult}SD'],
            name=f'+{mult} SD',
            line=dict(color=colors[i], width=1, dash='dot'),
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=recent_df.index,
            y=recent_df[f'Lower_{mult}SD'],
            name=f'-{mult} SD',
            line=dict(color=colors[i], width=1, dash='dot'),
            showlegend=True
        ))
    
    fig.update_layout(
        title=f"{symbol} - SD Band Analysis (90 Days)",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def main():
    # Initialize session state
    if 'screen_results' not in st.session_state:
        st.session_state.screen_results = None
    if 'config_submitted' not in st.session_state:
        st.session_state.config_submitted = False
    
    # Title
    st.markdown("<h1>📊 Standard Deviation Screener</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Daily Mean Reversion Analysis for Nifty & FNO Stocks</p>", unsafe_allow_html=True)
    
    # Configuration screen
    if not st.session_state.config_submitted:
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Configuration")
            
            universe_choice = st.radio(
                "**Select Universe:**",
                ["🔵 Nifty Index (^NSEI) Only", "📈 FNO Stocks Only", "🎯 Both Nifty + FNO Stocks"],
                index=0
            )
            
            signal_filter = st.selectbox(
                "**Signal Filter:**",
                ["All Signals", "Long Setups Only", "Short Setups Only", "Extreme Zones Only"],
                index=0
            )
            
            sort_by = st.selectbox(
                "**Sort Results By:**",
                ["Distance from Mean (High to Low)", "Current SD Position (Extreme to Neutral)", 
                 "Days in Current Zone (High to Low)", "Symbol Name (A to Z)"],
                index=0
            )
            
            st.markdown("---")
            
            if st.button("🚀 **RUN ANALYSIS**", type="primary"):
                with st.spinner("Running analysis..."):
                    results = screen_stocks(universe_choice, signal_filter, sort_by)
                    st.session_state.screen_results = results
                    st.session_state.config_submitted = True
                    st.rerun()
            
            st.markdown("<p style='text-align: center; color: #888; margin-top: 20px;'>Will analyze based on 3-year rolling Standard Deviation</p>", unsafe_allow_html=True)
    
    # Results screen
    else:
        results = st.session_state.screen_results
        
        if not results:
            st.warning("No stocks found matching the criteria. Please adjust filters and try again.")
            if st.button("⚙️ Change Configuration"):
                st.session_state.config_submitted = False
                st.rerun()
            return
        
        # Summary cards
        render_summary_cards(results)
        
        st.markdown("---")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Format columns for display
        display_df = results_df.copy()
        display_df['Current Position'] = display_df['Current Position'].apply(lambda x: f"{x:+.2f} SD")
        display_df['Distance to Mean'] = display_df.apply(
            lambda row: f"{row['Distance to Mean']:+.2f} pts ({row['Distance %']:+.2f}%)", axis=1
        )
        display_df['Days in Zone'] = display_df['Days in Zone'].apply(lambda x: f"{x} days")
        display_df['Risk-Reward'] = display_df['Risk-Reward'].apply(lambda x: f"1:{x:.1f}" if x > 0 else "N/A")
        
        # Select columns for display
        display_columns = [
            'Symbol', 'Current Price', 'Mean (8 SMA)', 'SD Value', 'Current Position',
            'Signal', 'Distance to Mean', 'Days in Zone', '+1.5 SD Level', '-1.5 SD Level',
            '+2.5 SD Level', '-2.5 SD Level', 'Expected Move', 'Potential Return %', 'Risk-Reward'
        ]
        
        display_df = display_df[display_columns]
        
        # Display main results table
        st.markdown("### 📋 Screening Results")
        st.dataframe(
            style_dataframe(display_df),
            height=600,
            use_container_width=True
        )
        
        # Export buttons
        st.markdown("---")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            excel_data, excel_filename = export_to_excel(display_df)
            st.download_button(
                label="📥 Download Excel",
                data=excel_data,
                file_name=excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            csv_data, csv_filename = export_to_csv(display_df)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv"
            )
        
        with col3:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.session_state.config_submitted = False
                st.session_state.screen_results = None
                st.rerun()
        
        with col4:
            if st.button("⚙️ Change Configuration"):
                st.session_state.config_submitted = False
                st.rerun()
        
        # Detailed charts (optional)
        with st.expander("📊 View Detailed Charts for Selected Stocks"):
            selected_symbols = st.multiselect(
                "Select stocks to view charts:",
                options=[r['Symbol'] for r in results]
            )
            
            if selected_symbols:
                for symbol in selected_symbols:
                    symbol_with_suffix = f"{symbol}.NS" if symbol != "^NSEI" else symbol
                    df = fetch_stock_data(symbol_with_suffix)
                    if df is not None:
                        df = calculate_sd_bands(df)
                        fig = create_stock_chart(symbol, df)
                        st.plotly_chart(fig, use_container_width=True)
        
        # Bottom summary
        st.markdown("---")
        st.markdown("### 📊 Market Overview")
        
        total = len(results)
        overbought = len([r for r in results if r['Current Position'] > 1.5])
        oversold = len([r for r in results if r['Current Position'] < -1.5])
        neutral = total - overbought - oversold
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Overbought:** {overbought} stocks ({overbought/total*100:.1f}%)")
        with col2:
            st.success(f"**Oversold:** {oversold} stocks ({oversold/total*100:.1f}%)")
        with col3:
            st.warning(f"**Neutral:** {neutral} stocks ({neutral/total*100:.1f}%)")
        
        # Top opportunities
        st.markdown("### 🎯 Top 5 Opportunities")
        top_opps = sorted(results, key=lambda x: x['Risk-Reward'] if x['Risk-Reward'] > 0 else 0, reverse=True)[:5]
        
        top_df = pd.DataFrame(top_opps)[['Symbol', 'Signal', 'Current Position', 'Potential Return %', 'Risk-Reward']]
        top_df['Current Position'] = top_df['Current Position'].apply(lambda x: f"{x:+.2f} SD")
        top_df['Risk-Reward'] = top_df['Risk-Reward'].apply(lambda x: f"1:{x:.1f}" if x > 0 else "N/A")
        
        st.dataframe(top_df, use_container_width=True)
        
        # Extreme zones alert
        extreme_stocks = [r['Symbol'] for r in results if 'EXTREME' in r['Signal']]
        if extreme_stocks:
            st.warning(f"⚠️ **Stocks at Extreme Levels (±2.5 SD):** {', '.join(extreme_stocks)}")
        
        # Footer
        st.markdown("---")
        st.caption(f"Last Updated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')} IST")
        
        with st.expander("ℹ️ About This Strategy"):
            st.markdown("""
            **Standard Deviation Mean Reversion Strategy:**
            
            This screener identifies stocks that have deviated significantly from their mean and are likely to revert:
            
            - **Mean Line:** 8-period Simple Moving Average of closing prices
            - **Standard Deviation:** Calculated using 3 years of close-to-close price differences
            - **Entry Signals:**
              - LONG: Price crosses above -1.5 SD (oversold reversal)
              - SHORT: Price crosses below +1.5 SD (overbought reversal)
            - **Stop Loss:** Set at ±2.5 SD levels
            - **Target:** Mean reversion (8 SMA)
            
            **Risk Disclaimer:** This is for educational purposes only. Always conduct your own research before trading.
            """)

if __name__ == "__main__":
    main()
