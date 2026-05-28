# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. 台股中文名稱對照 ---
def get_chinese_stock_name(stock_id):
    """獲取台股中文名稱"""
    stock_id = stock_id.strip()
    
    # 常見台股中文名稱對照表
    chinese_names = {
        # 大型權值股
        "2330": "台積電",
        "2454": "聯發科", 
        "2317": "鴻海",
        "2308": "台達電",
        "2882": "中信金",
        "2881": "富邦金",
        "2885": "元大金",
        "2884": "玉山金",
        "2880": "華南金",
        "2892": "中信金控",
        
        # 半導體相關
        "3037": "欣興",
        "2379": "瑞昱",
        "8086": "台半",
        "3231": "迅杰",
        "6776": "祥碩",
        "3443": "創意",
        "4938": "景硕",
        "5269": "祥邦",
        
        # 電子股
        "3008": "大立光",
        "2357": "華碩",
        "2365": "鴻準",
        "2382": "廣達",
        "2324": "仁寶",
        "2362": "藍天",
        "2344": "華擎",
        "2371": "華映",
        "2345": "華科",
        
        # 金融股
        "2886": "兆豐金",
        "2887": "台新金",
        "2890": "永豐金",
        "2891": "中信銀",
        "2893": "凱基",
        "2883": "開發金",
        "2845": "鴻準",
        
        # 傳產股
        "1301": "台塑",
        "1303": "南亞",
        "1326": "台化",
        "2002": "中鋼",
        "2006": "南鋼",
        "1101": "台泥",
        "1102": "亞泥",
        
        # 櫃買指數成分股
        "0050": "元大台灣50",
        "0056": "元大高股息",
        "006208": "元大台灣50反1",
        "00632L": "元大台灣50正2",
        
        # 其他熱門股
        "8069": "世芯-KY",
        "6415": "矽力-KY",
        "3260": "日月光投控",
        "3711": "日月光",
        "2388": "威強電",
        "2412": "中華電",
        "9910": "豐泰",
        "9933": "康那香",
        "9943": "佳格",
        "9958": "呈益",
        "9937": "旺旺",
        "9914": "美利達"
    }
    
    return chinese_names.get(stock_id, None)

# --- 2. 股票資訊獲取 ---
@st.cache_data(ttl=3600)
def get_stock_info(stock_id):
    """獲取股票基本資訊"""
    stock_id = stock_id.strip()
    
    # 嘗試多種台股代號格式
    symbols_to_try = [
        f"{stock_id}.TW",  # 台灣上市
        f"{stock_id}.TWO", # 台灣上櫃
        f"{stock_id}.TAI", # 另一種格式
        stock_id           # 直接嘗試原始代號
    ]
    
    for symbol in symbols_to_try:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # 檢查是否成功獲取資訊
            if info and 'shortName' in info:
                return {
                    'symbol': symbol,
                    'shortName': info.get('shortName', 'N/A'),
                    'longName': info.get('longName', 'N/A'),
                    'currency': info.get('currency', 'TWD'),
                    'market': info.get('market', 'N/A'),
                    'industry': info.get('industry', 'N/A')
                }
        except Exception as e:
            continue
    
    return None

# --- 3. 串接真實市場資料 ---
@st.cache_data(ttl=3600)
def fetch_stock_data(stock_id):
    # 台股代號處理：上市加 .TW，上櫃加 .TWO
    end_date = datetime.today()
    start_date = end_date - timedelta(days=60)

    # 清理股票代號，移除可能的空格或特殊字符
    stock_id = stock_id.strip()
    
    # 嘗試多種台股代號格式
    symbols_to_try = [
        f"{stock_id}.TW",  # 台灣上市
        f"{stock_id}.TWO", # 台灣上櫃
        f"{stock_id}.TAI", # 另一種格式
        stock_id           # 直接嘗試原始代號
    ]
    
    df = None
    for symbol in symbols_to_try:
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not df.empty:
                break
        except Exception as e:
            continue
    
    if df is None or df.empty:
        return None

    # 確保欄位為 1D Series 並處理可能的 MultiIndex (yfinance v0.2.x 特性)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 確保只返回單一欄位的 DataFrame
    result = df[['Close', 'Volume']].copy()
    return result

# --- 4. 處置股預警運算邏輯 ---
def analyze_disposition_risk(df):
    if df is None or len(df) < 30:
        return ["數據量不足，無法計算風險 (需至少30個交易日)"], None

    # 創建副本避免修改原始數據
    data = df.copy()
    
    # 計算關鍵指標
    data['Return_6D'] = (data['Close'] / data['Close'].shift(6) - 1) * 100
    data['Return_10D'] = (data['Close'] / data['Close'].shift(10) - 1) * 100
    data['Return_30D'] = (data['Close'] / data['Close'].shift(30) - 1) * 100
    data['Vol_10D_MA'] = data['Volume'].shift(1).rolling(window=10).mean()
    data['Vol_30D_MA'] = data['Volume'].shift(1).rolling(window=30).mean()
    data['Vol_Ratio_10D'] = data['Volume'] / data['Vol_10D_MA']
    data['Vol_Ratio_30D'] = data['Volume'] / data['Vol_30D_MA']
    
    # 計算週轉率 (假設發行股數需要外部數據，這裡用成交量估算)
    data['Turnover_Rate'] = (data['Volume'] / data['Volume'].rolling(window=30).mean()) * 10
    data['Turnover_3D_MA'] = data['Turnover_Rate'].rolling(window=3).mean()
    
    # 計算價格振幅
    data['Daily_Range'] = ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1)) * 100
    data['Range_3D_MA'] = abs(data['Daily_Range']).rolling(window=3).mean()

    latest = data.iloc[-1]
    warnings = []
    risk_level = 0

    # 1. 累積漲跌幅異常檢測
    if abs(latest['Return_6D']) >= 25:
        warnings.append(f"🔴 6日累積漲跌幅達 {latest['Return_6D']:.2f}% (注意股門檻 ±25%)")
        risk_level += 2
    
    if abs(latest['Return_10D']) >= 50:
        warnings.append(f"🔴 10日累積漲跌幅達 {latest['Return_10D']:.2f}% (注意股門檻 ±50%)")
        risk_level += 2
        
    if abs(latest['Return_30D']) >= 100:
        warnings.append(f"🔴 30日累積漲跌幅達 {latest['Return_30D']:.2f}% (處置股門檻 ±100%)")
        risk_level += 3

    # 2. 成交量異常放大檢測
    if latest['Vol_Ratio_10D'] >= 3:
        warnings.append(f"🟡 今日成交量異常！為前10日均量之 {latest['Vol_Ratio_10D']:.2f} 倍")
        risk_level += 1
        
    if latest['Vol_Ratio_30D'] >= 5:
        warnings.append(f"🔴 今日成交量嚴重異常！為前30日均量之 {latest['Vol_Ratio_30D']:.2f} 倍")
        risk_level += 2

    # 3. 週轉率過高檢測
    if latest['Turnover_Rate'] >= 10:
        warnings.append(f"🟡 週轉率過高！達 {latest['Turnover_Rate']:.2f}%")
        risk_level += 1
        
    if latest['Turnover_3D_MA'] >= 10:
        warnings.append(f"🔴 連續3日平均週轉率過高！達 {latest['Turnover_3D_MA']:.2f}%")
        risk_level += 2

    # 4. 價格振幅異常檢測
    if latest['Range_3D_MA'] >= 6:
        warnings.append(f"🟡 3日平均價格振幅過高！達 {latest['Range_3D_MA']:.2f}%")
        risk_level += 1

    # 5. 綜合判斷 - 處置股判定
    consecutive_warning_days = 0
    for i in range(min(5, len(data))):
        if i == 0:
            continue
        prev_data = data.iloc[-(i+1)]
        if (abs(prev_data['Return_6D']) >= 25 or 
            prev_data['Vol_Ratio_10D'] >= 3 or 
            prev_data['Turnover_Rate'] >= 10):
            consecutive_warning_days += 1
        else:
            break

    if consecutive_warning_days >= 3:
        warnings.append(f"🔴 連續 {consecutive_warning_days} 日達到注意股標準，可能被處置！")
        risk_level += 3

    # 6. 處置股等級判定
    if risk_level >= 5:
        warnings.insert(0, "🚨 **高風險處置股** - 多項指標異常，建議謹慎投資！")
    elif risk_level >= 3:
        warnings.insert(0, "⚠️ **潛在處置股** - 達到多項注意股標準")
    elif risk_level >= 1:
        warnings.insert(0, "📊 **注意股** - 部分指標異常")

    if not warnings:
        warnings.append("✅ 目前市場指標尚在正常範圍內。")

    return warnings, latest

# --- 5. Streamlit 網頁前端設計 ---
st.set_page_config(page_title="台股處置股預警系統", layout="wide")
st.title("🚨 台股處置股預警系統")
st.markdown("本系統串接真實市場數據，計算該股是否符合注意或處置警示條件。")

with st.sidebar:
    st.header("設定追蹤目標")
    stock_input = st.text_input("請輸入股票代號 (如: 2330, 2454, 8069)", value="2330")
    
    # 添加快速測試按鈕
    st.markdown("**快速測試股票**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("測試 2330"):
            stock_input = "2330"
    with col2:
        if st.button("測試高風險股"):
            stock_input = "6776"  # 通常高波動性股票
    
    analyze_btn = st.button("開始分析", type="primary")
    
    st.markdown("---")
    st.markdown("**處置股判斷標準**")
    st.markdown("- 6日漲跌幅 ±25%")
    st.markdown("- 10日漲跌幅 ±50%") 
    st.markdown("- 30日漲跌幅 ±100%")
    st.markdown("- 成交量異常放大")
    st.markdown("- 週轉率過高")
    st.markdown("- 連續3日達標")

if analyze_btn and stock_input:
    with st.spinner(f"正在從 Yahoo Finance 抓取 {stock_input} 資料..."):
        # 獲取股票基本資訊
        stock_info = get_stock_info(stock_input)
        df = fetch_stock_data(stock_input)

        if df is not None and stock_info is not None:
            warnings, latest_data = analyze_disposition_risk(df)

            # 顯示股票資訊
            chinese_name = get_chinese_stock_name(stock_input)
            if chinese_name:
                display_name = chinese_name
            else:
                # 如果沒有中文名稱，使用英文名稱
                stock_name = stock_info['shortName']
                if len(stock_name) > 30:  # 如果英文名稱太長，使用完整名稱
                    stock_name = stock_info['longName']
                display_name = stock_name
            
            st.success(f"📈 **{stock_input} - {display_name}**")
            
            # 顯示額外股票資訊
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("市場", stock_info['market'])
            with info_col2:
                st.metric("行業", stock_info['industry'][:20] + "..." if len(stock_info['industry']) > 20 else stock_info['industry'])
            with info_col3:
                st.metric("貨幣", stock_info['currency'])

            st.markdown("---")

            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("📊 今日收盤數據")
                st.metric(label="最新收盤價", value=f"${latest_data['Close']:.2f}")
                st.metric(label="今日成交量", value=f"{int(latest_data['Volume']):,}")

            with col2:
                st.subheader("⚠️ 風險預警判定")
                if warnings:
                    for w in warnings:
                        st.warning(w)
                else:
                    st.success("✅ 目前市場指標尚在正常範圍內。")

            st.subheader("📈 近期價格走勢 (Real Data)")
            st.line_chart(df['Close'])
        else:
            if stock_info is None:
                st.error(f"無法找到股票代碼 '{stock_input}' 的基本資訊，請檢查代碼是否正確。")
            else:
                st.error(f"無法找到股票代碼 '{stock_input}' 的價格資料，請檢查代碼是否正確。")
