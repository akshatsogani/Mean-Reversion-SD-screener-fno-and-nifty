import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
from typing import List, Dict, Optional, Tuple
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SD Mean Reversion Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CUSTOM_CSS = """
<style>
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .main-header {
        font-size: 48px;
        font-weight: 700;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 18px;
        text-align: center;
        color: #666;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .signal-long { background-color: #28a745; color: white; padding: 2px 8px; border-radius: 4px; }
    .signal-short { background-color: #dc3545; color: white; padding: 2px 8px; border-radius: 4px; }
    .signal-wait { background-color: #ffc107; color: black; padding: 2px 8px; border-radius: 4px; }
    .signal-extreme { background-color: #6610f2; color: white; padding: 2px 8px; border-radius: 4px; }
    div[data-testid="stDataFrame"] {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f0f2f6;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
"""

SD_LOOKBACK = 756
MEAN_PERIOD = 8
SD_MULTIPLIERS = [1.5, 2.0, 2.5]

@st.cache_data(ttl=300)
def load_fno_symbols() -> List[str]:
    try:
        df = pd.read_excel("FNO_Stock.xlsx")
        if 'Symbol' not in df.columns:
            st.error("Error: 'Symbol' column not found in FNO_Stock.xlsx")
            return []
        symbols = df['Symbol'].dropna().tolist()
        return [f"{symbol}.NS" for symbol in symbols]
    except FileNotFoundError:
        st.error("Error: FNO_Stock.xlsx not found. Please ensure the file is in the root directory.")
        return []
    except Exception as e:
        st.error(f"Error loading FNO symbols: {str(e)}")
        return []

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(symbol: str, date_key: str = None) -> Optional[pd.DataFrame]:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="4y", interval="1d")
        if len(df) < SD_LOOKBACK + 1:
            return None
        return df[['Close']]
    except Exception:
        return None

def calculate_sd_bands(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < SD_LOOKBACK + 1:
        return df
    
    lookback_df = df.tail(SD_LOOKBACK + 1)
    differences = lookback_df['Close'].diff().dropna()
    sd_value = np.std(differences.values, ddof=0)
    
    df['Mean'] = df['Close'].rolling(window=MEAN_PERIOD).mean()
    df['SD_Value'] = sd_value
    
    for multiplier in SD_MULTIPLIERS:
        df[f'Upper_{multiplier}SD'] = df['Mean'] + (multiplier * sd_value)
        df[f'Lower_{multiplier}SD'] = df['Mean'] - (multiplier * sd_value)
    
    df['Current_SD_Position'] = (df['Close'] - df['Mean']) / sd_value
    
    return df

def determine_signal(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "INSUFFICIENT_DATA"
    
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    current_sd = latest['Current_SD_Position']
    prev_sd = previous['Current_SD_Position']
    
    if current_sd > 2.5:
        return "EXTREME SHORT"
    elif current_sd < -2.5:
        return "EXTREME LONG"
    elif prev_sd > 1.5 and current_sd <= 1.5:
        return "SHORT"
    elif prev_sd < -1.5 and current_sd >= -1.5:
        return "LONG"
    elif current_sd > 1.5:
        return "SHORT ZONE"
    elif current_sd < -1.5:
        return "LONG ZONE"
    else:
        return "WAIT"

def calculate_days_in_zone(df: pd.DataFrame) -> int:
    count = 0
    for i in range(len(df)-1, -1, -1):
        if abs(df.iloc[i]['Current_SD_Position']) > 1.5:
            count += 1
        else:
            break
    return count

def calculate_risk_reward(current_price: float, mean: float, current_sd: float, sd_value: float) -> float:
    distance_to_mean = abs(current_price - mean)
    
    if current_sd > 0:
        stop_loss_level = mean + (2.5 * sd_value)
        risk = abs(stop_loss_level - current_price)
    else:
        stop_loss_level = mean - (2.5 * sd_value)
        risk = abs(current_price - stop_loss_level)
    
    if risk == 0:
        return 0
    
    return distance_to_mean / risk

def process_single_stock(symbol: str, date_key: str = None) -> Optional[Dict]:
    df = fetch_stock_data(symbol, date_key)
    if df is None or len(df) < SD_LOOKBACK + 1:
        return None
    
    df = calculate_sd_bands(df)
    if pd.isna(df['Mean'].iloc[-1]):
        return None
    
    signal = determine_signal(df)
    
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
        'Symbol': symbol.replace('.NS', '').replace('^NSEI', 'NIFTY'),
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
        'Risk-Reward': rr_ratio,
        '_df': df
    }

def screen_stocks(universe_choice: str, signal_filter: str, sort_by: str) -> List[Dict]:
    results = []
    date_key = datetime.now().strftime('%Y-%m-%d')
    
    if universe_choice == "🔵 Nifty Index (^NSEI) Only":
        symbols = ["^NSEI"]
    elif universe_choice == "📈 FNO Stocks Only":
        symbols = load_fno_symbols()
    else:
        symbols = ["^NSEI"] + load_fno_symbols()
    
    if not symbols:
        return []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_single_stock, symbol, date_key): symbol for symbol in symbols}
        completed = 0
        
        for future in as_completed(futures):
            completed += 1
            symbol = futures[future]
            status_text.text(f"Analyzing {symbol}... ({completed}/{len(symbols)})")
            progress_bar.progress(completed / len(symbols))
            
            try:
                result = future.result(timeout=10)
                if result:
                    if signal_filter == "All Signals":
                        results.append(result)
                    elif signal_filter == "Long Setups Only" and "LONG" in result['Signal']:
                        results.append(result)
                    elif signal_filter == "Short Setups Only" and "SHORT" in result['Signal']:
                        results.append(result)
                    elif signal_filter == "Extreme Zones Only" and "EXTREME" in result['Signal']:
                        results.append(result)
            except Exception:
                continue
    
    progress_bar.empty()
    status_text.empty()
    
    if sort_by == "Distance from Mean (High to Low)":
        results.sort(key=lambda x: abs(x['Distance to Mean']), reverse=True)
    elif sort_by == "Current SD Position (Extreme to Neutral)":
        results.sort(key=lambda x: abs(x['Current Position']), reverse=True)
    elif sort_by == "Days in Current Zone (High to Low)":
        results.sort(key=lambda x: x['Days in Zone'], reverse=True)
    else:
        results.sort(key=lambda x: x['Symbol'])
    
    return results

def style_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    def color_signal(val):
        if 'LONG' in str(val):
            return 'background-color: #28a745; color: white'
        elif 'SHORT' in str(val):
            return 'background-color: #dc3545; color: white'
        elif val == 'WAIT':
            return 'background-color: #ffc107; color: black'
        return ''
    
    def color_position(val):
        try:
            num_val = float(str(val).replace(' SD', ''))
            if num_val > 1.5:
                return 'color: #dc3545; font-weight: bold'
            elif num_val < -1.5:
                return 'color: #28a745; font-weight: bold'
            return 'color: #666666'
        except:
            return ''
    
    styled = df.style.applymap(color_signal, subset=['Signal'])
    styled = styled.applymap(color_position, subset=['Current Position'])
    
    styled = styled.format({
        'Current Price': '₹{:.2f}',
        'Mean (8 SMA)': '₹{:.2f}',
        'SD Value': '{:.2f} pts',
        'Current Position': '{:+.2f} SD',
        'Distance to Mean': lambda x: f"{x:+.2f} pts ({x/df.loc[df['Distance to Mean'] == x, 'Current Price'].values[0]*100:+.2f}%)" if len(df.loc[df['Distance to Mean'] == x]) > 0 else "",
        'Days in Zone': '{} days',
        '+1.5 SD Level': '₹{:.2f}',
        '-1.5 SD Level': '₹{:.2f}',
        '+2.5 SD Level': '₹{:.2f}',
        '-2.5 SD Level': '₹{:.2f}',
        'Expected Move': '{:+.2f} pts',
        'Potential Return %': '{:.2f}%',
        'Risk-Reward': lambda x: f"1:{x:.1f}" if x > 0 else "N/A"
    })
    
    return styled

def export_to_excel(df: pd.DataFrame) -> Tuple[io.BytesIO, str]:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='SD Screener Results')
        
        workbook = writer.book
        worksheet = writer.sheets['SD Screener Results']
        
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 30)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    output.seek(0)
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    filename = f"SD_Screener_Results_{timestamp}.xlsx"
    return output, filename

def export_to_csv(df: pd.DataFrame) -> Tuple[io.BytesIO, str]:
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    filename = f"SD_Screener_Results_{timestamp}.csv"
    return output, filename

def render_config_screen():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">📊 Standard Deviation Screener</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Daily Mean Reversion Analysis for Nifty & FNO Stocks</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Configuration")
        
        universe_choice = st.radio(
            "Select Universe",
            ["🔵 Nifty Index (^NSEI) Only", "📈 FNO Stocks Only", "🎯 Both Nifty + FNO Stocks"],
            index=2
        )
        
        signal_filter = st.selectbox(
            "Signal Filter",
            ["All Signals", "Long Setups Only", "Short Setups Only", "Extreme Zones Only"]
        )
        
        sort_by = st.selectbox(
            "Sort Results By",
            ["Distance from Mean (High to Low)", "Current SD Position (Extreme to Neutral)", 
             "Days in Current Zone (High to Low)", "Symbol Name (A to Z)"]
        )
        
        st.markdown("---")
        
        if st.button("🚀 RUN ANALYSIS", use_container_width=True):
            return universe_choice, signal_filter, sort_by
        
        st.info("📌 Will analyze based on 3-year rolling Standard Deviation with 8-period SMA mean")
    
    return None, None, None

def render_summary_cards(results: List[Dict]):
    total = len(results)
    long_count = sum(1 for r in results if 'LONG' in r['Signal'])
    short_count = sum(1 for r in results if 'SHORT' in r['Signal'])
    extreme_count = sum(1 for r in results if 'EXTREME' in r['Signal'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Analyzed", f"{total} stocks")
    with col2:
        st.metric("📈 Long Setups", f"{long_count} ({long_count/total*100:.1f}%)" if total > 0 else "0")
    with col3:
        st.metric("📉 Short Setups", f"{short_count} ({short_count/total*100:.1f}%)" if total > 0 else "0")
    with col4:
        st.metric("⚠️ Extreme Zones", f"{extreme_count} stocks")

def render_results_table(results: List[Dict]):
    if not results:
        st.warning("No results found. Please adjust your filters.")
        return None
    
    df = pd.DataFrame(results)
    
    display_columns = [
        'Symbol', 'Current Price', 'Mean (8 SMA)', 'SD Value', 'Current Position',
        'Signal', 'Distance to Mean', 'Days in Zone', '+1.5 SD Level', '-1.5 SD Level',
        '+2.5 SD Level', '-2.5 SD Level', 'Expected Move', 'Potential Return %', 'Risk-Reward'
    ]
    
    df_display = df[display_columns].copy()
    
    styled_df = style_dataframe(df_display)
    st.dataframe(styled_df, height=600, use_container_width=True)
    
    return df_display

def render_detail_charts(results: List[Dict], selected_symbols: List[str]):
    if not selected_symbols:
        return
    
    cols = st.columns(2)
    for idx, symbol in enumerate(selected_symbols[:4]):
        result = next((r for r in results if r['Symbol'] == symbol), None)
        if not result or '_df' not in result:
            continue
        
        df = result['_df'].tail(90)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            mode='lines', name='Price',
            line=dict(color='black', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Mean'],
            mode='lines', name='Mean (8 SMA)',
            line=dict(color='blue', width=1.5, dash='dash')
        ))
        
        for multiplier in [1.5, 2.0, 2.5]:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[f'Upper_{multiplier}SD'],
                mode='lines', name=f'+{multiplier} SD',
                line=dict(color='red', width=0.5),
                showlegend=multiplier==1.5
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df[f'Lower_{multiplier}SD'],
                mode='lines', name=f'-{multiplier} SD',
                line=dict(color='green', width=0.5),
                showlegend=multiplier==1.5
            ))
        
        fig.update_layout(
            title=f"{symbol} - 90 Day SD Analysis",
            xaxis_title="Date",
            yaxis_title="Price",
            height=400,
            hovermode='x unified'
        )
        
        with cols[idx % 2]:
            st.plotly_chart(fig, use_container_width=True)

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    if 'results' not in st.session_state:
        universe, signal_filter, sort_by = render_config_screen()
        
        if universe and signal_filter and sort_by:
            with st.spinner("Analyzing stocks..."):
                results = screen_stocks(universe, signal_filter, sort_by)
                st.session_state.results = results
                st.session_state.universe = universe
                st.session_state.signal_filter = signal_filter
                st.session_state.sort_by = sort_by
                st.rerun()
    else:
        st.markdown('<h1 class="main-header">📊 Standard Deviation Screener</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Analysis Results</p>', unsafe_allow_html=True)
        
        render_summary_cards(st.session_state.results)
        
        st.markdown("---")
        st.markdown("### 📋 Screening Results")
        
        df = render_results_table(st.session_state.results)
        
        if df is not None:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                excel_data, excel_filename = export_to_excel(df)
                st.download_button(
                    label="📥 Download as Excel",
                    data=excel_data,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                csv_data, csv_filename = export_to_csv(df)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_data.getvalue(),
                    file_name=csv_filename,
                    mime="text/csv"
                )
            
            with col3:
                if st.button("🔄 New Analysis"):
                    del st.session_state.results
                    st.rerun()
            
            with st.expander("📊 View Detailed Charts for Selected Stocks"):
                available_symbols = [r['Symbol'] for r in st.session_state.results]
                selected = st.multiselect(
                    "Select stocks to view charts (max 4)",
                    available_symbols,
                    max_selections=4
                )
                if selected:
                    render_detail_charts(st.session_state.results, selected)
            
            st.markdown("---")
            
            with st.expander("ℹ️ About Standard Deviation Strategy"):
                st.markdown("""
                **Strategy Overview:**
                - Uses 8-period SMA as the mean line
                - Calculates 3-year rolling standard deviation from close-to-close price differences
                - Identifies mean reversion opportunities when price deviates beyond ±1.5 SD
                
                **Signal Interpretation:**
                - **LONG/EXTREME LONG**: Price below -1.5/-2.5 SD, expect bounce back to mean
                - **SHORT/EXTREME SHORT**: Price above +1.5/+2.5 SD, expect pullback to mean
                - **WAIT**: Price within ±1.5 SD bands, no clear signal
                
                **Risk Management:**
                - Stop loss at ±2.5 SD levels
                - Target at mean (8 SMA)
                - Risk-Reward calculated based on distance to mean vs distance to stop
                
                **Disclaimer:** This is for educational purposes only. Always do your own research before trading.
                """)
            
            st.markdown("---")
            st.caption(f"Last Updated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')} IST")

if __name__ == "__main__":
    main()
