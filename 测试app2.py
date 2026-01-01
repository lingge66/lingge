# 合约策略回测代码 V24.0 (修复增强版)
# By LingGe_CTO (已修复 KeyError: 'initial_capital' 问题)

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Any
import time
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import io
import base64
import os
import json
from collections import defaultdict
import scipy.stats as scipy_stats
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

# 新增优化算法库
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import deap
from deap import base, creator, tools, algorithms
# ==========================================
# 🔥 新增：全局实时日志系统
# ==========================================
import re

def ui_log(msg: str):
    """
    双向日志函数 (V27.1 显示修复版)
    修复了 HTML 源码直接显示在网页上的问题
    """
    # 1. 后台打印
    print(msg)
    
    # 2. 初始化
    if 'ui_log_queue' not in st.session_state:
        st.session_state.ui_log_queue = []
    
    # 3. 文案修正
    msg = msg.replace("利润回吐", "利润")
    
    # 4. 基础样式 (白底、大字、加粗、无背景色)
    # 注意：这里去掉了多余的空格，压缩成一行
    base_style = "padding:12px 15px; margin-bottom:8px; background-color:#ffffff; border-bottom:1px solid #f0f0f0; font-family:'Segoe UI',monospace; font-size:16px; font-weight:700; line-height:1.6; color:#333333;"
    
    # 5. 颜色与逻辑判断
    left_border = "5px solid #ccc" # 默认灰色
    
    pnl_pct = 0.0
    try:
        match = re.search(r'\(([\+\-]?\d+\.\d+)%\)', msg)
        if match: pnl_pct = float(match.group(1))
    except: pass

    # --- 状态判断 ---
    if "开多" in msg:
        left_border = "6px solid #2196f3" # 蓝
    elif "开空" in msg:
        left_border = "6px solid #9c27b0" # 紫
    elif "平仓" in msg:
        if pnl_pct > 0:
            left_border = "6px solid #4caf50" # 绿
            if pnl_pct >= 5.0: 
                left_border = "8px solid #2e7d32" # 深绿
        else:
            left_border = "6px solid #ef5350" # 红
            if pnl_pct <= -3.0: 
                left_border = "8px solid #c62828" # 深红
                
    elif "熔断" in msg or "拒绝" in msg or "💀" in msg:
        left_border = "6px solid #ff9800" # 橙
    elif "接力" in msg:
        left_border = "6px solid #673ab7" # 深紫
    elif "贪婪" in msg:
        left_border = "6px solid #ffc107" # 金

    # --- 6. 关键数字着色 (正则替换) ---
    def highlight_pnl(m):
        text = m.group(0)
        # 只要是正数(包含+号)就绿，负数就红
        if "+" in text: 
            return f'<span style="color:#2e7d32; font-size:18px; font-weight:900;">{text}</span>'
        else: 
            return f'<span style="color:#c62828; font-size:18px; font-weight:900;">{text}</span>'
    
    # 替换规则
    msg = re.sub(r'💰.*?\)', highlight_pnl, msg)
    msg = re.sub(r'利润:\s*[\+\-]?\d+\.?\d*%', lambda m: f'<span style="color:#666;">{m.group(0)}</span>', msg)

    # 7. 组装最终 HTML (🔥关键修复：使用 f-string 紧凑拼接，不留缩进)
    msg_html = msg.replace('\n', '<br>')
    
    html = (
        f'<div style="{base_style} border-left: {left_border};">'
        f'{msg_html}'
        f'</div>'
    )

    # 8. 插入队列 (最新在最上面)
    st.session_state.ui_log_queue.insert(0, html)
    
    # 限制长度
    if len(st.session_state.ui_log_queue) > 1000:
        st.session_state.ui_log_queue.pop()
        
    # 9. 刷新显示 (高度增加到 800px)
    if 'log_placeholder' in st.session_state:
        full_content = "".join(st.session_state.ui_log_queue)
        st.session_state.log_placeholder.markdown(
            f'<div style="height: 800px; overflow-y: auto; padding: 5px;">{full_content}</div>', 
            unsafe_allow_html=True
        )
import plotly.express as px

def display_trade_analysis_ui(trades_list: List[Dict]):
    """
    高级交易复盘分析 UI (V1.0)
    功能：多维筛选、亏损归因、可视化分布
    """
    if not trades_list:
        st.warning("📭 暂无交易记录，无法分析")
        return

    st.markdown("---")
    st.subheader("📊 交易深度复盘 (Deep Dive)")

    # 1. 数据转换：List -> DataFrame
    df_trades = pd.DataFrame(trades_list)
    
    # 确保有必要的列，没有则补默认值
    required_cols = ['symbol', 'direction', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl', 'pnl_percent', 'exit_reason']
    for col in required_cols:
        if col not in df_trades.columns:
            df_trades[col] = 0 if 'pnl' in col else None
            
    # 计算辅助列
    if 'duration' not in df_trades.columns:
        df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
        df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
        df_trades['duration_hours'] = (df_trades['exit_time'] - df_trades['entry_time']).dt.total_seconds() / 3600
    
    # 2. 侧边栏筛选器
    with st.expander("🔍 筛选条件 (Filter Options)", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            filter_result = st.selectbox("盈亏状态", ["全部", "✅ 盈利单", "❌ 亏损单", "🛡️ 保本/微利"])
        
        with col2:
            # 如果你在 trade 对象里记录了 entry_adx，这里可以用
            min_adx, max_adx = st.slider("ADX 范围 (假设已记录)", 0, 100, (0, 100))
            
        with col3:
            min_dur, max_dur = st.slider("持仓时间 (小时)", 0.0, 100.0, (0.0, 100.0))
            
        with col4:
            search_reason = st.text_input("搜索离场原因 (如: 止损)", "")

    # 3. 执行筛选
    df_filtered = df_trades.copy()
    
    if filter_result == "✅ 盈利单":
        df_filtered = df_filtered[df_filtered['pnl'] > 0]
    elif filter_result == "❌ 亏损单":
        df_filtered = df_filtered[df_filtered['pnl'] <= 0]
    elif filter_result == "🛡️ 保本/微利":
        df_filtered = df_filtered[(df_filtered['pnl'] > 0) & (df_filtered['pnl_percent'] < 0.5)]

    df_filtered = df_filtered[
        (df_filtered['duration_hours'] >= min_dur) & 
        (df_filtered['duration_hours'] <= max_dur)
    ]
    
    if search_reason:
        df_filtered = df_filtered[df_filtered['exit_reason'].str.contains(search_reason, na=False, case=False)]

    # 4. 统计看板
    st.markdown(f"### 🎯 筛选结果: 共 {len(df_filtered)} 笔交易")
    
    if not df_filtered.empty:
        m1, m2, m3, m4 = st.columns(4)
        total_pnl = df_filtered['pnl'].sum()
        avg_pnl = df_filtered['pnl'].mean()
        win_rate = (df_filtered[df_filtered['pnl']>0].shape[0] / len(df_filtered)) * 100
        avg_dur = df_filtered['duration_hours'].mean()
        
        m1.metric("累计盈亏 (Filtered PnL)", f"${total_pnl:.2f}", delta_color="normal")
        m2.metric("平均单笔", f"${avg_pnl:.2f}")
        m3.metric("区间胜率", f"{win_rate:.1f}%")
        m4.metric("平均持仓", f"{avg_dur:.1f}h")

        # 5. 可视化分析
        tab1, tab2 = st.tabs(["📋 交易明细表", "📈 盈亏分布图"])
        
        with tab1:
            # 使用 Streamlit 的高级数据表格，自带排序
            st.dataframe(
                df_filtered.style.format({
                    'entry_price': '{:.4f}', 
                    'exit_price': '{:.4f}', 
                    'pnl': '{:.2f}', 
                    'pnl_percent': '{:.2f}%',
                    'duration_hours': '{:.1f}h'
                }).background_gradient(subset=['pnl'], cmap='RdYlGn', vmin=-100, vmax=100),
                use_container_width=True,
                height=400
            )
            
        with tab2:
            # 散点图：横轴=持仓时间，纵轴=盈亏百分比，颜色=方向
            fig = px.scatter(
                df_filtered, 
                x="duration_hours", 
                y="pnl_percent",
                color="direction",
                size=abs(df_filtered['pnl']), # 气泡大小代表金额大小
                hover_data=['exit_reason', 'entry_time'],
                title="持仓时间 vs 盈亏分布 (大泡泡=大盈亏)"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        # ==========================================
            # ==========================================
    # 🔥 [修复] 新增：导出“病历单”给 AI 分析 (修复 .get 报错)
    # ==========================================
    st.markdown("---")
    st.subheader("📤 导出数据给 AI 分析")
    
    export_data = []
    for t in trades_list:
        row = {}
        
        # --- 1. 智能提取器 (兼容 Dict 和 Object) ---
        def safe_get(item, key, default='N/A'):
            if isinstance(item, dict):
                return item.get(key, default)
            else:
                return getattr(item, key, default)

        # 基础字段
        for col in ['symbol', 'direction', 'entry_price', 'exit_price', 'pnl_percent', 'exit_reason', 'score']:
            row[col] = safe_get(t, col)
        # 🔥 新增诊断字段提取
        row['sl_source'] = getattr(t, 'sl_source', 'N/A')
        row['btc_env'] = getattr(t, 'btc_env', 'N/A')
        # 时间与持仓
        entry_t = safe_get(t, 'entry_time', None)
        exit_t = safe_get(t, 'exit_time', None)
        row['entry_time'] = str(entry_t)
        row['exit_time'] = str(exit_t)
        
        if entry_t and exit_t:
            try:
                # 兼容字符串和datetime对象
                t1 = pd.to_datetime(entry_t)
                t2 = pd.to_datetime(exit_t)
                row['hours'] = round((t2 - t1).total_seconds() / 3600, 1)
            except:
                row['hours'] = 0
            
        # 关键环境数据
        # 修复：先获取 smc_info 字典，再取内部值
        smc_info = safe_get(t, 'smc_info', {})
        # 修复：Trade 对象可能没有 market_regime 属性，给默认值
        row['regime'] = safe_get(t, 'market_regime', 'N/A') 
        row['smc'] = 1 if smc_info.get('smc_score', 0) > 0 else 0
        
        export_data.append(row)
        
    df_export = pd.DataFrame(export_data)
    
    # 转换为 JSON 字符串
    json_str = df_export.to_json(orient="records", date_format="iso")
    
    st.text_area("📋 复制下方 JSON 数据发给我 (前50条):", json_str[:50000], height=100)
    st.download_button(
        label="📥 下载完整交易记录 (JSON)",
        data=json_str,
        file_name="trade_history_for_ai.json",
        mime="application/json"
    )
class RollingVsIndependentValidator:
    def __init__(self):
        self.independent = {}
        self.rolling = {}
    
    def collect_independent(self, config, data_cache, optimizer_results, data_range_str):
        """收集 Tab 3 手动优化的证据"""
        self.independent = {
            'config': config.copy(),
            'data_keys': self._get_data_fingerprint(data_cache),
            'results_top1': optimizer_results[0] if optimizer_results else None,
            'data_range': data_range_str,
            'timestamp': datetime.now()
        }
    
    def collect_rolling(self, config, data_cache, optimizer_results, data_range_str):
        """收集 Tab 7 滚动回测的证据"""
        self.rolling = {
            'config': config.copy(),
            'data_keys': self._get_data_fingerprint(data_cache),
            'results_top1': optimizer_results[0] if optimizer_results else None,
            'data_range': data_range_str,
            'timestamp': datetime.now()
        }
    def _get_data_fingerprint(self, data_cache):
        """生成数据指纹（检查数据量和指标列）"""
        fingerprint = {}
        for sym, tfs in data_cache.items():
            if '4h' in tfs:
                df = tfs['4h']
                # 记录行数和最后一行的时间，以及是否有 ema_trend 列
                fingerprint[sym] = f"Rows:{len(df)}|End:{df.index[-1]}|HasEma:{'ema_trend' in df.columns}"
        return fingerprint

    def compare(self):
        """开庭审判：对比两者是否一致"""
        report = []
        passed = True
        
        # 🔥🔥🔥 修复点：先检查双方数据是否都存在 🔥🔥🔥
        # 如果手动优化数据为空，或者滚动回测数据为空，直接停止对比
        if not self.independent or not self.independent.get('config'):
            return False, ["⚠️ 无法对比: 缺少【手动优化(Tab3)】数据。请先去 Tab 3 运行一次优化，再来运行滚动回测。"]
            
        if not self.rolling or not self.rolling.get('config'):
            return False, ["⚠️ 无法对比: 缺少【滚动回测(Tab7)】数据。"]

        # 1. 获取配置 (现在安全了，因为上面检查过了)
        c1 = self.independent.get('config')
        c2 = self.rolling.get('config')
        
        # 2. 对比数据范围
        t1 = self.independent.get('data_range')
        t2 = self.rolling.get('data_range')
        if t1 == t2:
            report.append(f"✅ [时间窗口] 完全一致: {t1}")
        else:
            report.append(f"❌ [时间窗口] 不一致! 手动:{t1} vs 滚动:{t2}")
            passed = False

        # 3. 对比数据指纹
        d1 = self.independent.get('data_keys')
        d2 = self.rolling.get('data_keys')
        if d1 == d2:
            report.append(f"✅ [数据指纹] 完全一致 (预计算状态相同)")
        else:
            report.append(f"❌ [数据指纹] 不一致! \n手动:{d1}\n滚动:{d2}")
            passed = False

        # 4. 对比配置参数
        keys_to_check = ['initial_capital', 'position_mode', 'target_position_value']
        for k in keys_to_check:
            # 这里的 .get 也就安全了
            if c1.get(k) == c2.get(k):
                report.append(f"✅ [参数:{k}] 一致: {c1.get(k)}")
            else:
                report.append(f"❌ [参数:{k}] 不一致! 手动:{c1.get(k)} vs 滚动:{c2.get(k)}")
                passed = False

        # 5. 对比结果
        r1 = self.independent.get('results_top1')
        r2 = self.rolling.get('results_top1')
        if r1 and r2:
            score1 = r1.get('score', 0)
            score2 = r2.get('score', 0)
            if abs(score1 - score2) / (score1 + 0.001) < 0.05:
                 report.append(f"✅ [最终结果] 高度接近! 手动分:{score1:.2f} vs 滚动分:{score2:.2f}")
            else:
                 report.append(f"⚠️ [最终结果] 存在差异 (可能是随机性导致): 手动:{score1:.2f} vs 滚动:{score2:.2f}")
        
        return passed, report

# 初始化全局验证器
if 'global_validator' not in st.session_state:
    st.session_state.global_validator = RollingVsIndependentValidator()

PARAM_CN_MAP = {
    # --- 动态风控 ---
    'sideways_threshold': '震荡市-信号门槛',
    'sideways_rr': '震荡市-盈亏比',
    'trend_threshold': '趋势市-信号门槛',
    'trend_rr': '趋势市-盈亏比',
    
    # --- 均线系统 ---
    'ema_fast': 'EMA快线',
    'ema_medium': 'EMA中线',
    'ema_slow': 'EMA慢线',
    'ema_trend': 'EMA大势线',
    
    # --- 核心指标 ---
    'rsi_period': 'RSI周期',
    'atr_period': 'ATR周期',
    'bb_period': '布林带周期',
    'bb_std': '布林带宽度',
    'adx_period': 'ADX周期',
    'volume_ma': '成交量均线',
    
    # --- 门槛阈值 ---
    'min_rr_ratio': '基础盈亏比',
    'min_signal_score': '基础信号分',
    'min_adx': '最小趋势强度(ADX)',
    'max_volatility': '最大波动率限制',
    
    # --- 开关与SMC ---
    'use_smc_logic': '启用SMC逻辑',
    'use_dynamic_risk': '启用动态风控',
    'fvg_lookback': 'FVG回溯',
    'swing_lookback': '波段回溯',
    'rs_period': '相对强弱周期',
    
    # --- 资金管理 (新增) ---
    'stop_loss_amount': '总亏损止损',
    'min_continue_capital': '最小继续资金',
    'position_mode': '仓位模式',
    'leverage': '杠杆倍数',
    'compounding_ratio': '复利比例',
    'target_position_value': '目标单仓价值',
    
    # --- 优化权重因子 ---
    'screening_weights': '筛选权重组合'
}
def render_trading_memo():
    """在侧边栏显示实盘交易备忘录"""
    with st.sidebar.expander("📝 实盘作战备忘录 (重点必读)", expanded=True):
        
        st.markdown("### 🗓️ 运维节奏 (SOP)")
        st.info("""
        * **常规优化**: 每月 1 号 (雷打不动)
        * **数据窗口**: 
            * 训练集: 过去 6-9 个月 (找参数)
            * 验证集: 过去 3 年 (验抗压)
        * **紧急熔断**: 回撤 > 10% 或 连亏 6 单 -> **立即停止**
        """)

        st.markdown("### ⚙️ 参数双模式切换")
        
        st.success("""
        **🟢 A组 (牛市/进攻模式)**
        * **适用**: 趋势顺畅，均线完美发散，ADX > 30
        * **信号分**: `65 - 70`
        * **盈亏比**: `2.0` (吃鱼身，容忍小止损)
        """)

        st.warning("""
        **🛡️ B组 (震荡/防御模式)**
        * **适用**: 上下插针，频繁磨损，当前行情
        * **信号分**: `75 - 80` (不见兔子不撒鹰)
        * **盈亏比**: `3.0` (只做高赔率)
        """)

        st.markdown("### ⛔ 铁血风控")
        st.error("""
        1. **连跪降权**: 连亏 3 单 ➡️ **仓位减半**
        2. **硬止损**: 开单必须挂交易所 **STOP_MARKET**
        3. **不手痒**: 没信号坚决不开，不要因为无聊而交易
        """)
        
        st.caption("💡 记住：量化的核心是执行力，不是预测。")
warnings.filterwarnings('ignore')

# ==========================================
# 代理配置
# ==========================================

# 默认代理设置
DEFAULT_PROXY = {
    'http': 'http://127.0.0.1:10808',
    'https': 'http://127.0.0.1:10808'
}

# ==========================================
# 默认配置
# ==========================================
# 获取当前日期
current_date = datetime.now()
half_year_ago = current_date - timedelta(days=180)
DEFAULT_CONFIG = {
    'symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'],
    'start_date': half_year_ago.strftime('%Y-%m-%d'),
    'end_date': current_date.strftime('%Y-%m-%d'),
    'initial_capital': 10000,
    'target_position_value': 30000,
    'fee_rate': 0.0004,
    'slippage': 0.0010,
    'max_positions': 1,
    'check_interval_hours': 1,
    # 🔥 新增：BTC 大盘共振与止损融合配置
    'use_btc_protection': False,  # 【开关】是否开启 BTC 防崩盘保护 (默认先关，方便对比)
    'btc_protection_strict': False, # False=只防暴跌(推荐), True=必须暴涨才做多(太严)
    
    'use_fusion_stop_loss': True,   # 【开关】是否开启 融合止损 (ATR + 结构)
    # 风险管理参数
    'max_portfolio_risk': 0.1,
    'margin_maintenance': 0.5,
    'min_liquidity': 1000000,
    
    # 代币筛选配置
    'enable_token_screening': True,
    'select_best_token_only': True,
    'min_signal_threshold': 80,
    'screening_weights': {
        'signal_score': 0.40,
        'trend_strength': 0.25,
        'momentum': 0.15,
        'risk_reward': 0.10,
        'volume_confirmation': 0.10,
    },
    
    # 策略参数
    'min_rr_ratio': 2.5,
    'max_volatility': 0.04,
    'min_signal_score': 70,
    'min_adx': 25,
    
    # 聪明钱与动态风控参数
    'use_smc_logic': False,
    'use_dynamic_risk': False,
    'fvg_lookback': 3,
    'rs_period': 20,
    'swing_lookback': 10,
    
    # 技术指标参数
    'ema_fast': 9,
    'ema_medium': 21,
    'ema_slow': 50,
    'ema_trend': 200,
    'rsi_period': 14,
    'atr_period': 14,
    'bb_period': 20,
    'bb_std': 2.0,
    'volume_ma': 20,
    'adx_period': 14,
    
    # 🚀 新增资金管理参数
    'stop_loss_amount': 10000,  # 总亏损达到此金额停止交易
    'min_continue_capital': 1000,  # 最小继续交易资金
    'position_mode': 'dynamic_leverage',  # 'fixed', 'dynamic_leverage', 'compounding'
    'leverage_ratio': 2.5,  # 杠杆倍数
    'risk_per_trade': 0.03,  # 单笔交易风险比例（5%） 账户最大风险
    'compounding_ratio': 1.0,  # 复合增长模式下使用资金比例
    'max_position_value': 50000,  # 最大仓位价值限制
}
# ==========================================
# 🧠 [新增] 智能代币管理器 (集成了 历史回溯/分类/选币)
# ==========================================
class SmartTokenManager:
    def __init__(self, exchange_id='binance', proxies=None):
        try:
            # 配置项
            exchange_config = {
                'timeout': 30000,
                'enableRateLimit': True
            }
            # 🔥 关键：如果传入了代理，就挂载上去
            if proxies:
                exchange_config['proxies'] = proxies
                
            self.exchange = getattr(ccxt, exchange_id)(exchange_config)
            
        except Exception as e:
            print(f"❌ 交易所初始化失败: {e}")
            self.exchange = ccxt.binance()
            
        # 1. 硬编码的历史热点 (解决幸存者偏差)
        self.history_presets = {
            2020: ['AAVE/USDT', 'UNI/USDT', 'YFI/USDT', 'LINK/USDT', 'SNX/USDT', 'SUSHI/USDT'], # DeFi Summer
            2021: ['SOL/USDT', 'LUNA/USDT', 'AXS/USDT', 'MATIC/USDT', 'FTM/USDT', 'SAND/USDT', 'DOGE/USDT'], # 公链 & GameFi
            2022: ['GMT/USDT', 'APE/USDT', 'GALA/USDT', 'OP/USDT', 'LDO/USDT'], # 熊市中的阿尔法
            2023: ['ORDI/USDT', 'PEPE/USDT', 'INJ/USDT', 'TIA/USDT', 'TRB/USDT', 'WLD/USDT'], # 铭文 & AI
            2024: ['WIF/USDT', 'PEPE/USDT', 'SOL/USDT', 'RNDR/USDT', 'FET/USDT', 'FLOKI/USDT', 'ONDO/USDT', 'NOT/USDT'], # Meme & AI
            2025: ['IP/USDT', 'TRUMP/USDT', 'SUI/USDT', 'HYPE/USDT'] # 假设的未来热点
        }
        
        # 2. 简单的板块分类 (用于分析，不用于交易权重)
        self.sector_map = {
            'BTC': 'Core', 'ETH': 'Core', 'BNB': 'Core', 'SOL': 'L1', 'AVAX': 'L1', 'SUI': 'L1', 'SEI': 'L1',
            'PEPE': 'Meme', 'DOGE': 'Meme', 'WIF': 'Meme', 'BONK': 'Meme', 'SHIB': 'Meme',
            'RNDR': 'AI', 'FET': 'AI', 'WLD': 'AI', 'ARKM': 'AI',
            'UNI': 'DeFi', 'AAVE': 'DeFi', 'LDO': 'DeFi', 'ENA': 'DeFi',
            'ORDI': 'BRC20', 'SATS': 'BRC20'
        }

    def get_history_pool(self, year):
        """获取历史年份代币池"""
        base = ['BTC/USDT', 'ETH/USDT']
        hot_tokens = self.history_presets.get(year, [])
        # 去重并保持顺序
        return list(dict.fromkeys(base + hot_tokens))

    def classify_token(self, symbol):
        """简单分类"""
        base_symbol = symbol.split('/')[0]
        return self.sector_map.get(base_symbol, 'Others')

    def check_data_quality(self, df, timeframe='1h', min_bars=200):
        """
        数据质量安检门 (修复版：支持多周期适配)
        :param timeframe: 当前数据的周期 (1h, 4h, 1d)
        :param min_bars: 计算指标所需的最少 K 线数量 (至少要有200根才能算 EMA200)
        """
        if df is None or df.empty:
            return False, "数据为空"
        
        # 1. 获取当前数据行数
        current_bars = len(df)
        
        # 2. 核心检查：无论什么周期，K线数量必须能够计算出核心指标 (如 EMA200)
        # 如果数据少于 200 行，EMA200 就是空的，策略会报错
        if current_bars < min_bars:
            return False, f"数据长度不足 (只有 {current_bars} 行，策略至少需要 {min_bars} 行计算 EMA200)"
            
        # 3. 检查缺失值
        if df['close'].isnull().sum() > current_bars * 0.1:
            return False, "缺失值过多 (>10%)"
            
        # 4. 检查死价 (流动性枯竭)
        # 如果最高价等于最低价的情况超过 50%，说明是死币
        if (df['high'] == df['low']).mean() > 0.5:
            return False, "价格长期无波动 (僵尸币)"
            
        return True, "合格"

    def fetch_dynamic_hot_tokens(self, top_n=15, min_vol_m=10):
        """
        🚀 增强版自动选币：基于 成交量(50%) + 波动率(50%) 综合打分
        """
        try:
            tickers = self.exchange.fetch_tickers()
            candidates = []
            
            for s, d in tickers.items():
                if '/USDT' not in s: continue
                if any(bad in s for bad in ['UP/', 'DOWN/', 'BEAR', 'BULL', 'USDC']): continue
                
                vol = d.get('quoteVolume', 0)
                change = abs(d.get('percentage', 0))
                
                if vol < min_vol_m * 1_000_000: continue
                
                candidates.append({
                    'symbol': s,
                    'volume': vol,
                    'change': change,
                    # 简单评分: 成交量越大越好，波动越大越好 (趋势策略喜欢波动)
                    'score': (vol / 100_000_000) * 0.5 + (change * 10) * 0.5 
                })
            
            # 按综合评分排序
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 💡 相关性过滤 (简单版)：
            # 如果选了太多 Meme，可以手动在这里限制，比如 "Meme" 类不超过 3 个
            # (由于没有实时板块数据，这里暂不做 API 请求，保持回测速度)
            
            hot_list = [c['symbol'] for c in candidates[:top_n]]
            return list(set(['BTC/USDT', 'ETH/USDT'] + hot_list))
            
        except Exception as e:
            st.error(f"选币失败: {e}")
            return ['BTC/USDT', 'ETH/USDT']

# ==========================================
# 参数中文映射 (带功能备注版)
# ==========================================

PARAM_CHINESE_NAMES = {
    # --- 资金与风控 ---
    'initial_capital': '初始本金 (本钱)',
    'target_position_value': '目标仓位 (含杠杆总额)',
    'fee_rate': '手续费率 (交易所抽水)',
    'slippage': '滑点 (进场价差)',
    'max_positions': '最大持仓 (防单边风险)',
    'check_interval_hours': '检查频率 (K线周期)',
    # 🚀 新增资金管理参数中文映射
    'stop_loss_amount': '总亏损止损 (亏多少U停止)',
    'min_continue_capital': '最小继续资金 (低于此值停止)',
    'position_mode': '仓位模式 (固定/动态/复合)',
    'leverage_ratio': '杠杆倍数',
    'risk_per_trade': '单笔风险比例 (占资金%)',
    'compounding_ratio': '复利比例 (0.0-1.0)',
    'max_position_value': '最大仓位价值 (U)',
    # --- 趋势指标 (判断方向) ---
    'min_rr_ratio': '盈亏比',
    'min_signal_score': '信号分',
    'ema_fast': 'EMA快线 (短势线)',
    'ema_medium': 'EMA中线 (中支撑线)',
    'ema_slow': 'EMA慢线 (长趋势线)',
    'ema_trend': 'EMA大势线 (牛熊线)',
    'adx_period': 'ADX周期 (趋势强度)',

    # 🔥🔥🔥 【新增】 把这一块加进去 🔥🔥🔥
    'sideways_threshold': '震荡门槛 (防御)',
    'sideways_rr': '震荡盈亏比',
    'trend_threshold': '趋势门槛 (进攻)',
    'trend_rr': '趋势盈亏比',
    'enable_dynamic_params': '启用动态参数',
    
    # --- 震荡与入场 (找买点) ---
    'rsi_period': 'RSI周期 (超买超卖)',
    'atr_period': 'ATR周期 (计算止损)',
    'bb_period': '布林带周期 (价格通道)',
    'bb_std': '布林带宽度 (波动范围)',
    'volume_ma': '成交量均线 (量能)',
    
    # --- 策略核心阈值 (过滤信号) ---
    
    'max_volatility': '最大波动率 (防极端)',
    
    'min_adx': '最小趋势强度 (过滤震荡市)',
    
    # --- 聪明钱 SMC (机构行为) ---
    'use_smc_logic': '启用聪明钱 (机构订单流)',
    'use_dynamic_risk': '动态风控 (结构止损)',
    'fvg_lookback': 'FVG回溯 (找未成交缺口)',
    'rs_period': '相对强弱周期 (对比BTC强弱)',
    'swing_lookback': '波段回溯 (找前高前低)',
    
    # --- 代币筛选 ---
    'enable_token_screening': '启用选币 (只做最强/最弱)',
    'select_best_token_only': '只做第一名 (资金集中)',
    'min_signal_threshold': '筛选门槛 (垃圾币过滤)',
    'screening_weights.signal_score': '权重:形态得分',
    'screening_weights.trend_strength': '权重:趋势强度',
    'screening_weights.momentum': '权重:冲刺动能',
    'screening_weights.risk_reward': '权重:盈亏比',
    'screening_weights.volume_confirmation': '权重:成交量'
}

# ==========================================
# 数据类
# ==========================================

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any

@dataclass
class Trade:
    """
    交易记录对象 (V24.2 修复增强版)
    功能：
    1. 完整记录交易生命周期 (开仓->持仓->平仓)
    2. 追踪动态风控状态 (分批止盈、保本损、移动止损)
    3. 提供多维度绩效分析 (MAE/MFE/风险占比)
    4. 🔥 新增：初始止损记录、止损来源诊断、大盘环境记录
    """
    # ==========================================
    # 1. 核心必填字段 (无默认值，必须在最前面)
    # ==========================================
    id: int
    symbol: str
    direction: TradeDirection
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    leverage: float
    margin_used: float
    liquidation_price: float
    position_value: float
    signal_score: int
    signal_reasons: List[str]

    # ==========================================
    # 2. 扩展字段 (有默认值)
    # ==========================================
    
    # 🔥🔥🔥 [核心修复] 新增初始止损字段 (用于计算准确的 R 值) 🔥🔥🔥
    initial_stop_loss: float = 0.0

    # --- 交易诊断字段 ---
    mfe: float = 0.0  # Maximum Favorable Excursion (最大浮盈/最高光时刻)
    mae: float = 0.0  # Maximum Adverse Excursion (最大浮亏/最危险时刻)

    # --- 代币筛选与SMC信息 ---
    token_rank: int = 0
    screening_score: float = 0.0
    smc_info: Dict[str, Any] = field(default_factory=dict)

    # --- 仓位元数据 ---
    position_data: Dict[str, Any] = field(default_factory=dict)

    # --- 动态止盈止损与价格追踪 ---
    trailing_stop: float = 0.0
    highest_price: float = 0.0        # 持仓期间最高价
    lowest_price: float = float('inf') # 持仓期间最低价

    # 🔥 [新增关键字段] 分批止盈与风控状态
    tp1_hit: bool = False          # 是否已触及第一止盈位 (如 1.5R)
    remaining_ratio: float = 1.0   # 剩余仓位比例 (初始1.0，触发TP1后变为0.5)
    is_breakeven: bool = False     # 是否已触发保本损 (止损移至开仓价)

    # --- 离场结算信息 ---
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0           # 实现盈亏 (含分批止盈的累计)
    pnl_percent: float = 0.0   # 盈亏百分比
    fees: float = 0.0          # 交易手续费
    funding_fees: float = 0.0  # 资金费率
    
    # 🔥 新增诊断字段 (止损融合与BTC共振)
    sl_source: str = "N/A"   # 记录止损来源: 'ATR', 'Structure', 'Fusion'
    btc_env: str = "N/A"     # 记录开单时的 BTC 环境: 'Safe', 'Crash', 'Bull'

    def __post_init__(self):
        """初始化后的数据完整性校验与预处理"""
        # 1. 初始化 MFE/MAE 为开仓价
        if self.mfe == 0: self.mfe = self.entry_price
        if self.mae == 0: self.mae = self.entry_price
        
        # 2. 初始化极值追踪
        if self.highest_price == 0: 
            self.highest_price = self.entry_price
        if self.lowest_price == float('inf'): 
            self.lowest_price = self.entry_price

        # 3. 🔥 如果初始止损未被赋值（比如老代码调用），强制用当前止损兜底
        if self.initial_stop_loss == 0.0:
            self.initial_stop_loss = self.stop_loss

    @property
    def duration_hours(self) -> float:
        """持仓时长 (小时)"""
        if self.exit_time and self.entry_time:
            return (self.exit_time - self.entry_time).total_seconds() / 3600
        return 0.0

    @property
    def risk_reward_ratio(self) -> float:
        """当前设置的盈亏比 (Reward/Risk Ratio)"""
        if self.direction == TradeDirection.LONG:
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
        
        if risk > 0:
            return reward / risk
        return 0.0

    @property
    def is_closed(self) -> bool:
        """判断交易是否已结束"""
        return self.exit_time is not None

    @property
    def risk_percentage(self) -> float:
        """计算单笔交易风险占占用保证金的百分比"""
        if self.margin_used <= 0: return 0.0
        
        if self.direction == TradeDirection.LONG:
            risk_amount = (self.entry_price - self.stop_loss) * self.position_size
        else:
            risk_amount = (self.stop_loss - self.entry_price) * self.position_size
            
        return max(0.0, risk_amount / self.margin_used * 100)

    @property
    def leverage_used(self) -> float:
        """计算实际使用的有效杠杆"""
        return self.position_value / self.margin_used if self.margin_used > 0 else 0.0

    def get_position_summary(self) -> str:
        """获取结构化的仓位摘要信息 (用于日志和UI展示)"""
        if not self.position_data:
            return "⚠️ 无详细仓位数据"

        data = self.position_data
        
        # 动态获取当前状态标记
        status_flags = []
        if self.tp1_hit: status_flags.append("💰已减仓")
        if self.is_breakeven: status_flags.append("🛡️已保本")
        status_str = " | ".join(status_flags) if status_flags else "持有中"

        summary = f"""
🎯 仓位信息摘要 [{status_str}]:
├── 模式: {data.get('mode_info', 'N/A')}
├── 方向: {'🟢 做多' if self.direction == TradeDirection.LONG else '🔴 做空'} ({self.symbol})
├── 入场价: ${data.get('entry_price', 0):.4f}
├── 止损价: ${self.stop_loss:.4f} (原始: ${data.get('stop_loss', 0):.4f})
├── 仓位价值: ${self.position_value:.2f}U (剩余: {self.remaining_ratio*100:.0f}%)
├── 保证金: ${self.margin_used:.2f}U
├── 杠杆: {data.get('actual_leverage', 0):.1f}倍
├── 风险敞口: ${data.get('risk_amount_value', 0):.2f}U ({data.get('risk_percent', 0):.1f}%)
├── 爆仓价: ${data.get('liquidation_price', 0):.4f}
├── 安全边际: {data.get('safety_margin_percent', 0):.1f}%
└── 盈亏比: {self.risk_reward_ratio:.2f}:1
"""
        return summary.strip()

    def get_safety_margin(self) -> float:
        """获取安全边际百分比 (距离爆仓价的距离)"""
        if not self.position_data:
            return 0.0
        return self.position_data.get('safety_margin_percent', 0.0)
class DiffDetective:
    """
    全量参数与状态捕获器
    用于对比 Manual (手动) 和 Rolling (滚动) 的每一个原子级细节
    """
    def __init__(self):
        self.manual_snapshot = None
        self.rolling_snapshots = {} # Key: period_num or date_str

    def capture_manual(self, config, data_cache, stats):
        """捕获手动回测的现场"""
        # 计算首个代币的第一行数据的指标值（用于检测预热偏差）
        indicator_sample = {}
        first_symbol = config['symbols'][0]
        if first_symbol in data_cache and '4h' in data_cache[first_symbol]:
            df = data_cache[first_symbol]['4h']
            # 取中间某一行的数据做指纹（取最后一行容易受切片影响，取中间比较稳）
            mid_idx = len(df) // 2
            row = df.iloc[mid_idx]
            indicator_sample = {
                'sample_time': row.name,
                'ema_fast': row.get('ema_fast', 0),
                'ema_slow': row.get('ema_slow', 0),
                'rsi': row.get('rsi', 0),
                'data_start_date': df.index[0], # 数据集的物理起点
                'data_end_date': df.index[-1]   # 数据集的物理终点
            }

        self.manual_snapshot = {
            'type': 'MANUAL',
            'timestamp': datetime.now(),
            'config': config.copy(), # 深拷贝配置
            'stats': {
                'total_trades': stats.get('total_trades'),
                'total_return': stats.get('total_return'),
                'initial_capital': stats.get('initial_capital') # 关键！
            },
            'indicator_fingerprint': indicator_sample
        }

    def capture_rolling(self, period_num, config, data_cache, stats):
        """捕获某一轮滚动回测的现场"""
        indicator_sample = {}
        first_symbol = config['symbols'][0]
        
        # 注意：这里的 data_cache 应该是被切片过的
        if first_symbol in data_cache and '4h' in data_cache[first_symbol]:
            df = data_cache[first_symbol]['4h']
            if not df.empty:
                # 尝试找跟手动回测相同时间点的指纹，如果找不到就取中间
                mid_idx = len(df) // 2
                row = df.iloc[mid_idx]
                indicator_sample = {
                    'sample_time': row.name,
                    'ema_fast': row.get('ema_fast', 0),
                    'ema_slow': row.get('ema_slow', 0),
                    'rsi': row.get('rsi', 0),
                    'data_start_date': df.index[0],
                    'data_end_date': df.index[-1]
                }

        self.rolling_snapshots[period_num] = {
            'type': f'ROLLING_WIN_{period_num}',
            'timestamp': datetime.now(),
            'config': config.copy(),
            'stats': {
                'total_trades': stats.get('total_trades'),
                'total_return': stats.get('total_return'),
                'initial_capital': stats.get('initial_capital')
            },
            'indicator_fingerprint': indicator_sample
        }

# 初始化全局侦探
if 'diff_detective' not in st.session_state:
    st.session_state.diff_detective = DiffDetective()
# ==========================================
# 数据管理器
# ==========================================

class DataManager:
    """数据管理器，支持多时间框架本地缓存"""
    
    def __init__(self, data_dir: str = "crypto_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def get_cache_key(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
        """生成缓存键"""
        key_str = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cache_path(self, cache_key: str) -> str:
        """获取缓存路径"""
        return os.path.join(self.data_dir, f"{cache_key}.pkl")
    
    def save_data(self, symbol: str, timeframe: str, start_date: str, end_date: str, data: pd.DataFrame):
        """保存数据到本地"""
        cache_key = self.get_cache_key(symbol, timeframe, start_date, end_date)
        cache_path = self.get_cache_path(cache_key)
        
        cache_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'data': data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def load_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从本地加载数据"""
        cache_key = self.get_cache_key(symbol, timeframe, start_date, end_date)
        cache_path = self.get_cache_path(cache_key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # 验证数据是否匹配
                if (cache_data['symbol'] == symbol and 
                    cache_data['timeframe'] == timeframe and
                    cache_data['start_date'] == start_date and
                    cache_data['end_date'] == end_date):
                    
                    # 检查数据是否足够新（30天内）
                    cache_time = datetime.strptime(cache_data['timestamp'], '%Y-%m-%d %H:%M:%S')
                    if (datetime.now() - cache_time).days < 30:
                        return cache_data['data']
            except Exception as e:
                st.warning(f"缓存加载失败 {symbol} {timeframe}: {str(e)}")
                return None
        
        return None
    
    def get_all_timeframes_data(self, symbol: str, start_date: str, end_date: str, 
                               timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """获取指定代币的所有时间框架数据"""
        result = {}
        for timeframe in timeframes:
            data = self.load_data(symbol, timeframe, start_date, end_date)
            if data is not None:
                result[timeframe] = data
        return result
    
    def save_all_timeframes_data(self, symbol: str, start_date: str, end_date: str,
                               data_dict: Dict[str, pd.DataFrame]):
        """保存所有时间框架数据"""
        for timeframe, data in data_dict.items():
            self.save_data(symbol, timeframe, start_date, end_date, data)
    
    def clear_cache(self, days_old: int = 30):
        """清除旧缓存"""
        cutoff_time = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        deleted_count += 1
                except Exception as e:
                    st.warning(f"删除缓存文件失败 {filename}: {str(e)}")
        
        return deleted_count
    
    def get_data_stats(self) -> Dict:
        """获取缓存数据统计信息"""
        stats = {
            'total_files': 0,
            'symbols': set(),
            'timeframes': set(),
            'total_size_mb': 0
        }
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.pkl'):
                stats['total_files'] += 1
                filepath = os.path.join(self.data_dir, filename)
                
                # 获取文件大小
                stats['total_size_mb'] += os.path.getsize(filepath) / (1024 * 1024)
                
                # 解析文件名获取symbol和timeframe
                try:
                    parts = filename.replace('.pkl', '').split('_')
                    if len(parts) >= 4:
                        symbol = parts[0] + '/' + parts[1]
                        stats['symbols'].add(symbol)
                        stats['timeframes'].add(parts[2])
                except:
                    pass
        
        return stats
# ==========================================
# 🔥 【新增】 全局主数据管理器 (解决预热偏差的核心)
# ==========================================
class MasterDataManager:
    """
    上帝视角数据管理器 (Master Data Manager)
    核心职责：
    1. 接收原始 OHLCV 数据。
    2. 基于全量历史计算所有技术指标 (EMA, RSI, ATR等)。
    3. 提供“只读切片”服务，确保切片后的数据保留了基于历史计算的指标值。
    """
    def __init__(self, config: Dict, data_cache: Dict):
        self.config = config
        self.raw_cache = data_cache  # 原始数据
        self.processed_cache = {}    # 计算好指标的全量数据
        # 实例化信号检测器用于计算指标
        self.signal_detector = SmartMoneySignalDetector(config)
        self._is_prepared = False

    def prepare_all_indicators(self):
        """基于全量历史计算指标，确保存入缓存"""
        if self._is_prepared:
            return self.processed_cache

        print("⚡ [MasterData] 开始全量指标预计算 (消除预热偏差)...")
        for symbol, timeframes in self.raw_cache.items():
            self.processed_cache[symbol] = {}
            for tf, df in timeframes.items():
                if tf in ['1h', '4h'] and not df.empty:
                    try:
                        # 1. 深度复制原始数据，防止污染源数据
                        df_calc = df.copy()
                        
                        # 2. 全量计算指标 (关键！这里计算的是整个历史长河的指标)
                        # 这样保证了即使是切片的第1行，其 EMA200 也是基于过去200天算出来的
                        df_calc = self.signal_detector.calculate_indicators(df_calc)
                        
                        # 3. 存入处理后的缓存
                        self.processed_cache[symbol][tf] = df_calc
                    except Exception as e:
                        print(f"❌ [MasterData] {symbol} {tf} 指标计算失败: {e}")
                        self.processed_cache[symbol][tf] = df.copy() # 失败则回退
                else:
                    self.processed_cache[symbol][tf] = df
                    
        self._is_prepared = True
        print("✅ [MasterData] 全量指标计算完成，数据指纹已锁定。")
        return self.processed_cache

    def get_slice(self, start_date: str, end_date: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        获取带指标的时间切片 (只读)
        """
        if not self._is_prepared:
            self.prepare_all_indicators()

        sliced_cache = {}
        # 将字符串日期转换为 datetime 对象 (包含当天的最后一秒)
        s_dt = pd.to_datetime(start_date)
        e_dt = pd.to_datetime(end_date) + timedelta(hours=23, minutes=59, seconds=59)
        
        for sym, tfs in self.processed_cache.items():
            sliced_cache[sym] = {}
            for tf, df in tfs.items():
                if df.empty:
                    sliced_cache[sym][tf] = df
                    continue
                    
                # 使用布尔索引进行切片
                mask = (df.index >= s_dt) & (df.index <= e_dt)
                # copy() 是必须的，防止回测引擎修改切片影响主数据
                sliced_cache[sym][tf] = df.loc[mask].copy()
                
        return sliced_cache    
# ==========================================
# 优化版：实盘风控管理器 (平衡风险与交易机会)
# ==========================================
class RealTimeRiskManager:
    """优化版风控管理器：更合理的参数"""
    
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_loss_limit = 0.7  # 7%单日亏损限制 (原5%)
        self.max_consecutive_losses = 4  # 4次连败 (原3次)
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.cooldown_until = None
        
        # 🔥 新增：最小恢复时间（避免频繁熔断）
        self.min_recovery_time = timedelta(hours=4)
        
        # 🔥 新增：交易计数器
        self.trades_today = 0
        self.max_trades_per_day = 5  # 每日最多5单
    
    # 🔥🔥🔥 必须添加这个方法！ 🔥🔥🔥
    def calculate_total_portfolio_risk(self, active_trades: List['Trade']) -> float:
        """计算当前所有持仓的总风险敞口"""
        total_risk_amount = 0
        for trade in active_trades:
            if hasattr(trade, 'entry_price') and hasattr(trade, 'stop_loss') and hasattr(trade, 'position_size'):
                risk = abs(trade.entry_price - trade.stop_loss) * trade.position_size
                total_risk_amount += risk
        return total_risk_amount
    
    def can_open_position(self, position_risk: float, active_trades: List['Trade'], current_time: datetime) -> bool:
        """优化版开仓检查"""
        
        # 1. 检查冷却期
        if self.cooldown_until and current_time < self.cooldown_until:
            return False
        elif self.cooldown_until and current_time >= self.cooldown_until:
            # 冷却期结束
            self.cooldown_until = None
            self.consecutive_losses = max(0, self.consecutive_losses - 2)  # 部分恢复
        
        # 2. 检查单日亏损限制
        daily_loss_limit_amount = self.initial_capital * self.daily_loss_limit
        if self.daily_pnl <= -daily_loss_limit_amount:
            # 🔥 优化：只冷却2小时（原24小时）
            self.cooldown_until = current_time + timedelta(hours=2)
            return False
        
        # 3. 检查连败限制（带弹性）
        if self.consecutive_losses >= self.max_consecutive_losses:
            # 🔥 优化：冷却时间与连败次数成比例
            cooldown_hours = min(8, self.consecutive_losses * 2)  # 最多8小时
            self.cooldown_until = current_time + timedelta(hours=cooldown_hours)
            return False
        
        # 4. 检查总风险敞口（保持原10%限制）
        total_risk = self.calculate_total_portfolio_risk(active_trades)  # 🔥 这里调用了！
        if (total_risk + position_risk) > self.current_capital * 0.10:
            return False
            
        # 5. 检查每日交易次数限制
        if self.trades_today >= self.max_trades_per_day:
            return False
            
        return True
    
    def update_after_trade(self, pnl: float):
        """平仓后更新（添加交易计数）"""
        self.daily_pnl += pnl
        self.current_capital += pnl
        self.trades_today += 1
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            # 🔥 优化：盈利时减少连败计数（最多减到0）
            self.consecutive_losses = max(0, self.consecutive_losses - 1)
    
    def reset_daily(self):
        """每日重置"""
        self.daily_pnl = 0
        self.trades_today = 0
        # 🔥 优化：不清空连败计数，但给予恢复
        self.consecutive_losses = max(0, self.consecutive_losses - 1)

# ==========================================
# 智能仓位计算器（支持亏损后动态调整）
# ==========================================

class SmartPositionManager:
    """
    🔥 终极版：智能仓位管理 (Smart Position Manager v3)
    融合特性：
    1. 🐊 鳄鱼策略 (利润垫 + 趋势强度)
    2. 🛡️ 风格偏好 (Conservative/Aggressive)
    3. 📉 ADX斜率判断 (防止高位接盘)
    4. 🌊 波动率阻尼 (防止高波被秒爆)
    5. ❤️ 生命线保护 (防止缩量过度无法回本)
    """
    def __init__(self, config: Dict):
        self.config = config
        self.initial_capital = config.get('initial_capital', 10000)
        self.fee_rate = config.get('fee_rate', 0.0004)
        # 获取用户选择的风格，默认为平衡
        self.profile = config.get('risk_preference', 'Balanced')
        
        # 为了兼容旧代码调用的属性，设置一些默认值或别名
        self.target_position_value = config.get('target_position_value', 30000)
        self.compounding_ratio = config.get('compounding_ratio', 0.5)
        self.position_mode = config.get('position_mode', 'fixed')
        
        self._init_risk_profile()

    def _init_risk_profile(self):
        """根据风格初始化核心参数"""
        if self.profile == 'Conservative': 
            # 🛡️ 保守模式
            self.base_leverage = 1.0    # 基础1倍
            self.max_leverage = 3.0     # 封顶3倍
            self.profit_sensitivity = 0.5 # 利润加成慢
            self.floor_leverage = 0.8   # 最低0.8倍
        elif self.profile == 'Aggressive': 
            # 🦁 激进模式

            self.base_leverage = 3.0    # 基础3倍
            self.max_leverage = 10.0    # 封顶10倍
            self.profit_sensitivity = 1.5 # 利润加成快
            self.floor_leverage = 1.0   # 绝不低于1倍(保留翻身火种)
        else: 
            # ⚖️ 平衡模式 (Balanced)
            self.base_leverage = 2.0
            self.max_leverage = 5.0
            self.profit_sensitivity = 1.0
            self.floor_leverage = 0.8

    def calculate_position(self, entry_price: float, stop_loss: float, 
                           direction: Any, current_capital: float, 
                           adx_value: float = 0, prev_adx_value: float = 0, atr_value: float = 0) -> Dict:
        """
        🔥 全参数计算：引入 ADX斜率 和 ATR波动率
        """
        if current_capital <= 0: return {'can_trade': False, 'reason': '破产'}

        # ---------------------------------------------------
        # 1. 利润垫逻辑 (Profit Cushion) + 生命线保护 (Recovery Trap)
        # ---------------------------------------------------
        profit_ratio = (current_capital - self.initial_capital) / self.initial_capital
        
        if profit_ratio < 0:
            # 亏损状态：降仓，但有地板价 (floor_leverage)
            # 逻辑：即使亏损50%，杠杆系数也不会无限降低，给翻身留生机
            target_drop = 1.0 + profit_ratio
            min_drop = self.floor_leverage / self.base_leverage
            cushion_factor = max(min_drop, target_drop)
        else:
            # 盈利状态：放大，但有封顶 (3.0倍系数)
            cushion_factor = min(3.0, 1.0 + (profit_ratio * self.profit_sensitivity))

        # ---------------------------------------------------
        # 2. 趋势强度 + 斜率判断 (ADX Slope - 防止接盘)
        # ---------------------------------------------------
        trend_factor = 1.0
        adx_slope = adx_value - prev_adx_value
        
        if adx_value < 20:
            trend_factor = 0.5 # 🈚 无趋势：减半
        elif adx_value > 50:
            if adx_slope < 0:
                trend_factor = 0.8 # 📉 高位拐头：严重减仓，防止山顶接盘！
            else:
                trend_factor = 1.5 # 🚀 高位加速：重仓出击，主升浪！
        elif adx_value > 25:
            if adx_slope > 0:
                trend_factor = 1.2 # 📈 趋势增强：微加
            else:
                trend_factor = 1.0 # ➡️ 趋势减弱：保持

        # ---------------------------------------------------
        # 3. 波动率阻尼 (Volatility Damper - 防止秒爆)
        # ---------------------------------------------------
        # 计算当前的波动率百分比 (例如 ATR=50, Price=1000 => 5%)
        # 如果 ATR 为 0 (未传入)，则给一个默认安全值
        if atr_value <= 0: atr_value = entry_price * 0.02

        volatility_pct = (atr_value / entry_price) * 100 if entry_price > 0 else 0
        
        # 物理限制：安全杠杆 = 100% / (波动率 * 安全系数2.0)
        # 含义：必须能扛住 2倍 ATR 的反向波动而不爆仓
        safe_leverage_limit = 100 / (volatility_pct * 2.0 + 0.1) # +0.1防除零
        
        # ---------------------------------------------------
        # 4. 综合计算最终杠杆
        # ---------------------------------------------------
        # 这里的 base_leverage 来自用户的配置或 Config，但在 SmartManager 中我们用 self.base_leverage
        # 为了尊重用户在 UI 上滑动的 "base leverage"，我们可以取两者较小值或加权
        # 这里 V3 逻辑是完全接管，所以主要依靠 profit cushion 和 trend
        
        raw_leverage = self.base_leverage * cushion_factor * trend_factor
        
        # 应用三重限制
        # A. 风格上限 (Aggressive max 10x)
        lev_1 = min(raw_leverage, self.max_leverage)
        # B. 物理波动上限 (防止秒爆)
        lev_2 = min(lev_1, safe_leverage_limit)
        # C. 生命线底限 (防止死得太透)
        final_leverage = max(lev_2, self.floor_leverage)

        # ---------------------------------------------------
        # 5. 仓位落地
        # ---------------------------------------------------
        price_risk_dist = abs(entry_price - stop_loss)
        if price_risk_dist == 0: return {'can_trade': False, 'reason': '止损为0'}

        # 动态风险比例：杠杆越大，允许的单笔本金亏损比例也适当放大
        # 但设置硬顶 10% (激进模式)
        leverage_ratio_calc = final_leverage / self.base_leverage
        base_risk_cap = 0.02 if self.profile != 'Aggressive' else 0.04
        dynamic_risk_per_trade = min(base_risk_cap * leverage_ratio_calc, 0.10) 

        # 风险倒推价值 (Risk Based Value)
        risk_limit_amt = current_capital * dynamic_risk_per_trade
        position_value_risk = (risk_limit_amt / price_risk_dist) * entry_price
        
        # 杠杆硬顶价值 (Leverage Based Value)
        max_pos_by_lev = current_capital * final_leverage
        
        # 模式兼容 (Position Mode Check)
        if self.position_mode == 'fixed':
            # 如果是固定模式，尝试去接近 target_value，但受制于 max_pos_by_lev
            mode_value = self.target_position_value
        else:
            # 复合模式
            mode_value = current_capital * self.compounding_ratio * final_leverage

        # 取三者最小值：风险限制、杠杆限制、模式设定
        final_pos_value = min(position_value_risk, max_pos_by_lev, mode_value)

        if final_pos_value < 10: return {'can_trade': False, 'reason': '仓位过小'}

        position_size = final_pos_value / entry_price
        margin_used = final_pos_value / final_leverage

        # 爆仓价估算
        mmr = 0.005
        dir_val = direction.value if hasattr(direction, 'value') else str(direction)
        if dir_val == "LONG":
            liq_price = entry_price * (1 - (1/final_leverage) + mmr)
        else:
            liq_price = entry_price * (1 + (1/final_leverage) - mmr)

        return {
            'can_trade': True,
            'position_size': position_size,
            'position_value': final_pos_value,
            'margin_used': margin_used,
            'liquidation_price': liq_price,
            'actual_leverage': final_leverage,
            'risk_percent': dynamic_risk_per_trade * 100,
            'risk_amount_value': risk_limit_amt, # 兼容接口
            'open_fee': final_pos_value * self.fee_rate,
            'mode_info': f"{self.profile} V3",
            'debug_info': f"Lev:{final_leverage:.1f}x (ADX:{adx_value:.0f}|Slope:{adx_slope:+.1f}|ATR:{volatility_pct:.1f}%)"
        }
            
    
    def get_position_summary(self, position_data: Dict) -> str:
        """生成仓位摘要信息"""
        if not position_data.get('can_trade', True):
            return f"❌ 无法开仓: {position_data.get('reason', '未知原因')}"
        
        summary = f"""
🎯 仓位信息摘要:
├── 模式: {position_data.get('mode_info', 'N/A')}
├── 当前总资金: ${position_data.get('current_total_capital', 0):.0f}U
├── 入场价: ${position_data.get('entry_price', 0):.2f}
├── 止损价: ${position_data.get('stop_loss', 0):.2f}
├── 止盈价: ${position_data.get('take_profit_price', 0):.2f}
├── 仓位价值: ${position_data.get('position_value', 0):.2f}U
├── 保证金: ${position_data.get('margin_used', 0):.2f}U
├── 杠杆: {position_data.get('actual_leverage', 0):.1f}倍
├── 风险: ${position_data.get('risk_amount_value', 0):.2f}U ({position_data.get('risk_percent', 0):.1f}%保证金)
├── 风险/总资金: {position_data.get('risk_vs_capital', 0):.1f}%
├── 爆仓价: ${position_data.get('liquidation_price', 0):.2f}
├── 安全边际: {position_data.get('safety_margin_percent', 0):.1f}%
└── 盈亏比: {position_data.get('risk_reward_ratio', 0):.2f}:1
"""
        return summary
    
    def validate_position(self, position_data: Dict) -> Tuple[bool, str]:
        """验证仓位参数是否合理"""
        if not position_data.get('can_trade', True):
            return False, f"无法开仓: {position_data.get('reason', '未知原因')}"
        
        errors = []
        
        # 检查保证金是否为正
        if position_data['margin_used'] <= 0:
            errors.append("保证金必须大于0")
        
        # 检查风险是否过大
        if position_data['risk_percent'] > 100:
            errors.append(f"单笔风险({position_data['risk_percent']:.1f}%)过高")
        
        # 检查安全边际是否足够
        if position_data['safety_margin_percent'] < 2:
            errors.append(f"安全边际({position_data['safety_margin_percent']:.1f}%)过低，爆仓风险高")
        
        # 检查杠杆是否过高
        if position_data['actual_leverage'] > 10:
            errors.append(f"杠杆({position_data['actual_leverage']:.1f}倍)过高")
        
        # 检查风险占资金比例
        if position_data['risk_vs_capital'] > self.risk_per_trade * 100 * 1.5:  # 允许1.5倍容差
            errors.append(f"风险占资金比例({position_data['risk_vs_capital']:.1f}%)过高")
        
        if errors:
            return False, " | ".join(errors)
        return True, "仓位参数合理"
    
    def simulate_liquidation_scenario(self, position_data: Dict, price_drop_percent: float) -> Dict:
        """模拟价格下跌时的爆仓风险"""
        if not position_data.get('can_trade', True):
            return {
                'can_simulate': False,
                'reason': position_data.get('reason', '无仓位数据')
            }
        
        entry_price = position_data['entry_price']
        liquidation_price = position_data['liquidation_price']
        
        # 假设是多头仓位
        current_price = entry_price * (1 - price_drop_percent/100)
        
        # 计算当前保证金余额
        initial_margin = position_data['margin_used']
        position_size = position_data['position_size']
        
        # 计算未实现盈亏
        unrealized_pnl = (current_price - entry_price) * position_size
        
        # 计算当前权益
        current_equity = initial_margin + unrealized_pnl
        
        # 计算维持保证金要求
        position_value = current_price * position_size
        maintenance_required = position_value * self.maintenance_margin_rate
        
        # 计算保证金率
        margin_ratio = (current_equity / position_value) * 100 if position_value > 0 else 0
        
        # 检查是否接近爆仓
        price_to_liquidation = abs(current_price - liquidation_price) / entry_price * 100
        
        # 判断风险等级
        if current_price <= liquidation_price:
            liquidation_warning_level = 'LIQUIDATED'  # 已爆仓
        elif price_to_liquidation < 2:
            liquidation_warning_level = 'CRITICAL'    # 临界爆仓（2%以内）
        elif price_to_liquidation < 5:
            liquidation_warning_level = 'WARNING'     # 警告（5%以内）
        elif price_to_liquidation < 10:
            liquidation_warning_level = 'CAUTION'     # 谨慎（10%以内）
        else:
            liquidation_warning_level = 'SAFE'        # 安全
        
        return {
            'can_simulate': True,
            'current_price': current_price,
            'unrealized_pnl': unrealized_pnl,
            'current_equity': current_equity,
            'margin_ratio': margin_ratio,
            'price_to_liquidation_pct': price_to_liquidation,
            'is_liquidated': current_price <= liquidation_price,
            'liquidation_warning_level': liquidation_warning_level,
            'liquidation_price': liquidation_price,
            'distance_to_liquidation': current_price - liquidation_price if current_price > liquidation_price else 0
        }
    
    def calculate_max_position_for_capital(self, current_capital: float, entry_price: float) -> Dict:
        """
        计算给定资金下的最大可能仓位
        
        Args:
            current_capital: 当前资金
            entry_price: 入场价格
            
        Returns:
            最大仓位信息
        """
        if current_capital <= 0 or entry_price <= 0:
            return {
                'max_position_value': 0,
                'max_position_size': 0,
                'max_margin': 0,
                'max_leverage': self.leverage_ratio
            }
        
        # 计算最大保证金
        max_margin = current_capital
        
        # 计算最大仓位价值
        max_position_value = max_margin * self.leverage_ratio
        
        # 计算最大合约数量
        max_position_size = max_position_value / entry_price if entry_price > 0 else 0
        
        return {
            'max_position_value': max_position_value,
            'max_position_size': max_position_size,
            'max_margin': max_margin,
            'max_leverage': self.leverage_ratio,
            'current_capital': current_capital,
            'entry_price': entry_price
        }

# ==========================================
# 聪明钱信号检测器（增强版）
# ==========================================

class SmartMoneySignalDetector:
    """聪明钱信号检测器，集成FVG、相对强弱、动态风控及VWAP机构视角 (V25.0 升级版)"""
    
    def __init__(self, config=None):
        if config is None:
            config = {}
        
        # 基础技术指标参数
        self.params = {
            'ema_fast': config.get('ema_fast', 9),
            'ema_medium': config.get('ema_medium', 21),
            'ema_slow': config.get('ema_slow', 50),
            'ema_trend': config.get('ema_trend', 200),
            'rsi_period': config.get('rsi_period', 14),
            'atr_period': config.get('atr_period', 14),
            'bb_period': config.get('bb_period', 20),
            'bb_std': config.get('bb_std', 2.0),
            'volume_ma': config.get('volume_ma', 20),
            'adx_period': config.get('adx_period', 14)
        }
        
        # 策略基础参数
        self.min_rr_ratio = config.get('min_rr_ratio', 2.5)
        self.max_volatility = config.get('max_volatility', 0.04)
        self.min_signal_score = config.get('min_signal_score', 70)
        self.min_adx = config.get('min_adx', 25)
        
        # 聪明钱与动态风控参数
        self.use_smc_logic = config.get('use_smc_logic', False)
        self.use_dynamic_risk = config.get('use_dynamic_risk', False)
        self.fvg_lookback = config.get('fvg_lookback', 3)
        self.rs_period = config.get('rs_period', 20)
        self.swing_lookback = config.get('swing_lookback', 10)
        
        # ====================================================
        # 🔥 【核心参数】 动态风控阈值 (支持贝叶斯优化)
        # ====================================================
        self.sideways_threshold = config.get('sideways_threshold', 75)
        self.sideways_rr = config.get('sideways_rr', 3.0)
        self.trend_threshold = config.get('trend_threshold', 65)
        self.trend_rr = config.get('trend_rr', 2.0)
        
        # 功能开关
        self.enable_dynamic_params = config.get('enable_dynamic_params', True)
        
        # BTC数据缓存（用于相对强弱计算）
        self.btc_data = {}
    
    def set_btc_data(self, btc_data: Dict[str, pd.DataFrame]):
        """设置BTC数据用于相对强弱计算"""
        self.btc_data = btc_data
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标（增强版：新增 VWAP）"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # 基础指标
        df['returns'] = df['close'].pct_change()
        
        # EMA系统
        df['ema_fast'] = df['close'].ewm(span=self.params['ema_fast'], min_periods=self.params['ema_fast']).mean()
        df['ema_medium'] = df['close'].ewm(span=self.params['ema_medium'], min_periods=self.params['ema_medium']).mean()
        df['ema_slow'] = df['close'].ewm(span=self.params['ema_slow'], min_periods=self.params['ema_slow']).mean()
        df['ema_trend'] = df['close'].ewm(span=self.params['ema_trend'], min_periods=self.params['ema_trend']).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.params['rsi_period'], min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.params['rsi_period'], min_periods=1).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.params['atr_period'], min_periods=1).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(self.params['bb_period'], min_periods=1).mean()
        bb_std = df['close'].rolling(self.params['bb_period'], min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + bb_std * self.params['bb_std']
        df['bb_lower'] = df['bb_middle'] - bb_std * self.params['bb_std']
        
        # 成交量
        df['volume_ma'] = df['volume'].rolling(self.params['volume_ma'], min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        
        # ==========================================
        # 🔥 新增：VWAP (成交量加权平均价)
        # ==========================================
        # 计算典型价格
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vol_price'] = df['typical_price'] * df['volume']
        
        # 使用 24周期滚动 (约等于过去24小时的机构成本)
        # 如果是 4H K线，建议改为 6周期；如果是 1H K线，用 24周期
        vwap_window = 24 
        df['vwap'] = (df['vol_price'].rolling(vwap_window).sum() / 
                      df['volume'].rolling(vwap_window).sum())
        
        # MACD
        exp1 = df['close'].ewm(span=12, min_periods=12).mean()
        exp2 = df['close'].ewm(span=26, min_periods=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ADX (调用辅助函数)
        df['adx'] = self._calculate_adx(df, self.params['adx_period'])
        
        # 价格动量
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)
        
        # 支撑阻力
        df['support'] = df['low'].rolling(20, min_periods=1).min()
        df['resistance'] = df['high'].rolling(20, min_periods=1).max()
        
        # 填充缺失值
        df = df.fillna(method='ffill').fillna(0)
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算ADX指标 (内部辅助函数 - 标准修正版)
        逻辑：修正 DM 计算，对齐标准技术分析定义
        """
        if len(df) < period:
            return pd.Series(index=df.index, data=0.0)
            
        high = df['high']
        low = df['low']
        # close = df['close'] # ATR计算会用到
        
        # 1. 计算方向变动 (Directional Movement)
        # plus_dm: 今天最高价比昨天最高价高出的部分
        plus_dm = high.diff()
        # minus_dm: 昨天最低价比今天最低价低出的部分 (注意这里是反过来的，代表向下的力度)
        minus_dm = -low.diff()
        
        # 2. 修正 DM 逻辑 (Smoothing Logic)
        # 如果 +DM > -DM 且 > 0，则取值，否则为0
        # 如果 -DM > +DM 且 > 0，则取值，否则为0
        # 这种写法利用了 pandas 的 where 逻辑：cond ? val : other
        
        # 暂存原始 diff 结果以免变量覆盖影响判断
        _plus = plus_dm.copy()
        _minus = minus_dm.copy()
        
        plus_dm = _plus.where((_plus > _minus) & (_plus > 0), 0.0)
        minus_dm = _minus.where((_minus > _plus) & (_minus > 0), 0.0)
        
        # 3. 计算 ATR (真实波幅)
        # TR = Max(H-L, |H-Cp|, |L-Cp|)
        tr1 = high - low
        tr2 = abs(high - df['close'].shift())
        tr3 = abs(low - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 4. 平滑处理 (Smoothing) - 使用 Wilder's Smoothing (alpha=1/period) 
        # 注意：很多库用简单的 rolling mean，但在 ADX 标准定义中通常用 ewm
        # 这里为了保持跟您原逻辑一致性且兼顾效率，使用 rolling mean 是可接受的近似
        atr = tr.rolling(period).mean()
        
        # 5. 计算 DI (+DI, -DI)
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        # 6. 计算 DX 和 ADX
        # 处理分母为0的情况
        sum_di = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / sum_di.replace(0, 1) # 避免除零
        
        adx = dx.rolling(period).mean()
        
        return adx.fillna(0)
    
    def check_market_regime(self, df: pd.DataFrame, idx: int) -> Dict:
        """
        全天候市场状态检测 (ATR动态版 - 最终确认版)
        逻辑：ADX判强度 + 价格位置判方向 + ATR判波动范围
        """
        if idx < 50 or idx >= len(df):
            return {'tradeable': False, 'regime': 'insufficient_data'}
        
        # 1. 基础数据准备
        close = df['close'].iloc[idx]
        ema_trend = df['ema_trend'].iloc[idx] if 'ema_trend' in df.columns else close
        
        # 获取 ADX (如果没算，给默认值)
        adx = df['adx'].iloc[idx] if 'adx' in df.columns else 0
        
        # 获取 ATR (关键！用于动态衡量"距离")
        # 如果 atr 不存在，使用 close * 1% 作为临时兜底，防止报错
        atr = df['atr'].iloc[idx] if 'atr' in df.columns else close * 0.01 
        if atr == 0: atr = close * 0.01 # 防止 ATR 为 0 导致除零错误
        
        # ==========================================
        # 🎯 核心判决逻辑
        # ==========================================

        # 1. 计算价格偏离均线的程度 (用 ATR 标准化)
        # 距离均线多少个 ATR？ (Distance in ATR units)
        dist_from_ema = abs(close - ema_trend)
        dist_in_atr = dist_from_ema / atr
        
        # 2. 判定【震荡 (Ranging)】
        # 逻辑：趋势指标 ADX 很弱 (<25) 且 价格像磁铁一样粘在均线附近 (距离 < 1.0 ATR)
        # 这种情况下，均线失去方向指引作用，容易来回打脸
        is_weak_trend = adx < 25
        is_price_sticky = dist_in_atr < 1.0 
        
        if is_weak_trend and is_price_sticky:
            return {
                'tradeable': True, 
                'regime': 'ranging', 
                'confidence': 0.8, 
                'desc': f"均线粘合(偏离{dist_in_atr:.1f}ATR)"
            }
            
        # 3. 判定【高波动/剧烈震荡 (High Volatility)】
        # 如果单根 K 线的 ATR 占比超过了设定的最大阈值 (比如 4%)，说明市场极不稳定
        atr_pct = atr / close
        if atr_pct > self.max_volatility:
             return {
                'tradeable': False, 
                'regime': 'high_volatility',
                'desc': f"波动剧烈({atr_pct*100:.1f}%)"
             }

        # 4. 判定【趋势 (Trend)】
        # 既然排除了震荡和高波动，那大概率就是趋势了
        # 我们用 ADX 的值作为置信度 (Confidence)
        if close > ema_trend:
            # ADX 越高，趋势越强，置信度越高
            confidence = min(adx / 50.0, 1.0) 
            return {
                'tradeable': True, 
                'regime': 'bullish', 
                'confidence': confidence,
                'desc': f"多头趋势(ADX:{adx:.0f})"
            }
            
        elif close < ema_trend:
            confidence = min(adx / 50.0, 1.0)
            return {
                'tradeable': True, 
                'regime': 'bearish', 
                'confidence': confidence,
                'desc': f"空头趋势(ADX:{adx:.0f})"
            }

        # 5. 兜底 (理论上走不到这，但在边界条件下有用)
        return {'tradeable': True, 'regime': 'ranging', 'confidence': 0.2, 'desc': 'Neutral'}
    
    def check_multi_timeframe_alignment(self, df_4h: pd.DataFrame, df_1h: pd.DataFrame, 
                                        idx_4h: int, idx_1h: int) -> Dict:
        """
        多周期趋势一致性检查 (V25.0 宽容共振版)
        逻辑：大周期定方向，小周期找位置。允许小周期有轻微回调，只要不破关键均线。
        """
        if idx_4h < 50 or idx_4h >= len(df_4h) or idx_1h >= len(df_1h):
            return {'aligned': False, 'reason': 'insufficient_data'}
        
        # 1. 4H 趋势判定 (大方向)
        close_4h = df_4h['close'].iloc[idx_4h]
        ema_fast_4h = df_4h['ema_fast'].iloc[idx_4h]
        ema_slow_4h = df_4h['ema_slow'].iloc[idx_4h]
        
        trend_4h = 'neutral'
        # 只要价格在慢线之上，且快线也在慢线之上，就是多头结构 (比单纯 Close > Fast > Slow 更稳健)
        if close_4h > ema_slow_4h and ema_fast_4h > ema_slow_4h:
            trend_4h = 'bullish'
        elif close_4h < ema_slow_4h and ema_fast_4h < ema_slow_4h:
            trend_4h = 'bearish'
        
        # 2. 1H 趋势判定 (进场周期)
        close_1h = df_1h['close'].iloc[idx_1h]
        ema_fast_1h = df_1h['ema_fast'].iloc[idx_1h]
        ema_slow_1h = df_1h['ema_slow'].iloc[idx_1h]
        ema_trend_1h = df_1h['ema_trend'].iloc[idx_1h] if 'ema_trend' in df_1h.columns else ema_slow_1h
        
        trend_1h = 'neutral'
        if close_1h > ema_slow_1h:
            trend_1h = 'bullish'
        elif close_1h < ema_slow_1h:
            trend_1h = 'bearish'
            
        # 3. 共振逻辑优化
        # 严格模式：两个周期必须完全同向
        # 宽容模式：如果 4H 极强 (比如在 EMA Trend 之上)，允许 1H 轻微跌破 EMA Fast 但必须在 EMA Slow 之上
        
        aligned = False
        reason = ""
        
        if trend_4h == 'bullish':
            # 1H 也是多头，完美
            if trend_1h == 'bullish':
                aligned = True
            # 特殊情况：4H 极强，1H 虽然价格跌破快线在回调，但还在长期趋势线(EMA200)之上 -> 视为“回调接多”机会
            elif close_1h > ema_trend_1h:
                aligned = True
                reason = "4H强多+1H回调不破位"
                
        elif trend_4h == 'bearish':
            if trend_1h == 'bearish':
                aligned = True
            elif close_1h < ema_trend_1h:
                aligned = True
                reason = "4H强空+1H反弹不过位"
                
        if aligned:
            return {'aligned': True, 'direction': trend_4h, 'note': reason}
        else:
            return {'aligned': False, 'reason': f'Mismatch: 4h({trend_4h}) vs 1h({trend_1h})'}

    def detect_fvg(self, df: pd.DataFrame, idx: int) -> Dict[str, Any]:
        """
        识别公允价值缺口 (SMC Logic) - 实战回踩版
        逻辑：寻找过去 N 根 K 线内形成的缺口，检查当前价格是否正在回踩（Mitigation）这些区域。
        """
        # 建议 fvg_lookback 至少设为 10-20，否则很难捕捉到好的回踩
        search_range = max(self.fvg_lookback, 10) 
        
        if idx < search_range + 3:
            return {}
        
        current_price = df['close'].iloc[idx]
        
        fvg_bullish = []
        fvg_bearish = []
        
        # 遍历历史寻找缺口
        # 注意：我们是往回找已经形成的结构
        for j in range(1, search_range + 1):
            mid_idx = idx - j
            left_idx = mid_idx - 1
            right_idx = mid_idx + 1
            
            if left_idx < 0: continue
            
            # --- 看涨 FVG (Bullish Gap) ---
            # 结构：K(Left).High < K(Right).Low
            prev_high = df['high'].iloc[left_idx]
            next_low = df['low'].iloc[right_idx]
            
            if next_low > prev_high:
                # 检查这个缺口是否已经被完全回补过了（如果是老缺口，可能早失效了）
                # 简化逻辑：暂不检查历史是否回补，只看当前是否在区间内
                 fvg_bullish.append({
                    'range': [prev_high, next_low], 
                    'age': j
                })

            # --- 看跌 FVG (Bearish Gap) ---
            # 结构：K(Left).Low > K(Right).High
            prev_low = df['low'].iloc[left_idx]
            next_high = df['high'].iloc[right_idx]
            
            if next_high < prev_low:
                fvg_bearish.append({
                    'range': [next_high, prev_low],
                    'age': j
                })
        
        # 判断当前价格是否在 历史 FVG 区域内 (回踩确认)
        in_bullish_fvg = False
        in_bearish_fvg = False
        current_fvg_direction = 'none'
        
        # 检查回踩看涨缺口 (做多信号)
        for fvg in fvg_bullish:
            low_bound, high_bound = fvg['range']
            # 价格进入缺口区域 (且没有跌破下沿太多)
            if low_bound * 0.998 <= current_price <= high_bound:
                in_bullish_fvg = True
                current_fvg_direction = 'bullish'
                break # 只要踩中一个有效缺口即可
        
        # 检查回补看跌缺口 (做空信号)
        for fvg in fvg_bearish:
            low_bound, high_bound = fvg['range']
            # 价格进入缺口区域 (且没有涨破上沿太多)
            if low_bound <= current_price <= high_bound * 1.002:
                in_bearish_fvg = True
                current_fvg_direction = 'bearish'
                break
                
        return {
            'in_bullish_fvg': in_bullish_fvg,
            'in_bearish_fvg': in_bearish_fvg,
            'current_fvg_direction': current_fvg_direction
        }

    def calculate_relative_strength(self, df_symbol: pd.DataFrame, timeframe: str, idx: int) -> Dict[str, Any]:
        """
        计算相对强弱 (SMC Logic) - 安全对齐版
        修复：防止 BTC 数据缺失导致的索引错位
        """
        if not self.btc_data or timeframe not in self.btc_data:
            return {'rs_trend': 'neutral', 'rs_above_ma': False}
            
        df_btc = self.btc_data[timeframe]
        
        # 安全检查：索引越界
        if idx >= len(df_symbol): return {'rs_trend': 'neutral'}
        
        # 🔥【关键修复】时间对齐检查
        # 尝试对比时间戳，如果错位太大，说明数据不同步
        try:
            ts_symbol = df_symbol.index[idx]
            # 如果是 Int64Index (RangeIndex)，说明不是时间索引，只能被迫用 iloc
            # 如果是 DatetimeIndex，则可以检查
            if isinstance(df_symbol.index, pd.DatetimeIndex) and isinstance(df_btc.index, pd.DatetimeIndex):
                # 尝试用时间戳找 BTC 对应位置 (容错查找)
                if ts_symbol in df_btc.index:
                    btc_row = df_btc.loc[ts_symbol]
                    price_btc = btc_row['close']
                else:
                    # 如果找不到对应时间，回退到 iloc，但风险较大
                    if idx < len(df_btc): price_btc = df_btc['close'].iloc[idx]
                    else: return {'rs_trend': 'neutral'}
            else:
                # 非时间索引，直接用位置
                if idx < len(df_btc): price_btc = df_btc['close'].iloc[idx]
                else: return {'rs_trend': 'neutral'}
                
        except Exception:
            # 发生任何异常，兜底处理
            if idx < len(df_btc): price_btc = df_btc['close'].iloc[idx]
            else: return {'rs_trend': 'neutral'}

        price_symbol = df_symbol['close'].iloc[idx]
        if price_btc == 0: return {'rs_trend': 'neutral'}
        
        rs_ratio = price_symbol / price_btc
        
        # 计算 RS 均线 (动态计算最近 N 根)
        lookback = self.rs_period
        
        # 优化：不需要循环，利用向量化计算会更快，但这里为了局部计算方便用切片
        # 我们假设 RS 也是连续的
        start_idx = max(0, idx - lookback)
        end_idx = idx
        
        # 这里为了不引入复杂的 pandas Series 操作，做个简易均值
        # 注意：这只是个估算，但足够有效
        rs_ma = rs_ratio # 默认值
        
        # 如果能获取到历史数据片段
        if end_idx > start_idx:
            # 简易取样：只取首尾和中间，避免大量循环
            p_s_hist = df_symbol['close'].iloc[start_idx:end_idx]
            
            # 再次注意：这里假设 BTC 数据也是对齐的，如果不对齐，历史 RS 均线会有偏差
            # 考虑到回测速度，我们这里做一个权衡：直接用当前比例 * 0.99/1.01 做判断，
            # 或者用简单的 RS 移动平均
            
            # 修正版逻辑：直接判断 RS 相对自身的趋势，而不是均线
            # RS Rising?
            prev_idx = max(0, idx - 5)
            prev_rs = 0
            if prev_idx < len(df_btc):
                 prev_rs = df_symbol['close'].iloc[prev_idx] / df_btc['close'].iloc[prev_idx]
            
            if prev_rs > 0:
                if rs_ratio > prev_rs * 1.02:
                    return {'rs_trend': 'strong', 'rs_ratio': rs_ratio, 'rs_above_ma': True}
                elif rs_ratio < prev_rs * 0.98:
                    return {'rs_trend': 'weak', 'rs_ratio': rs_ratio, 'rs_above_ma': False}
        
        return {
            'rs_ratio': rs_ratio,
            'rs_ma': rs_ratio,
            'rs_trend': 'neutral',
            'rs_above_ma': False
        }

    def find_swing_points(self, df: pd.DataFrame, idx: int) -> Dict[str, Any]:
        """寻找波段高低点 (SMC Logic)"""
        if idx < self.swing_lookback:
            return {'swing_high': None, 'swing_low': None}
            
        # 在过去 N 根 K 线中寻找最高点和最低点
        start_idx = max(0, idx - self.swing_lookback)
        
        # 优化：不要包含当前 K 线 (idx)，否则"最高点"永远是当前价格，止损就没有意义了
        # 应该找"之前的"高点作为阻力位
        window_highs = df['high'].iloc[start_idx:idx]
        window_lows = df['low'].iloc[start_idx:idx]
        
        swing_high = window_highs.max()
        swing_low = window_lows.min()
        
        return {
            'swing_high': swing_high,
            'swing_low': swing_low
        }

    def calculate_dynamic_stop_loss(self, direction, entry_price, swing_high, swing_low, atr):
        """计算动态止损 (结合 ATR 和 结构位)"""
        
        # 1. 基础 ATR 止损 (兜底)
        base_sl = 0
        sl_mult = self.params.get('stop_loss_atr', 2.0)
        
        if direction == TradeDirection.LONG:
            base_sl = entry_price - (atr * sl_mult)
        else:
            base_sl = entry_price + (atr * sl_mult)

        if not self.use_dynamic_risk:
            return base_sl
        
        # 2. SMC 结构止损 (更紧凑，盈亏比更高)
        structure_sl = base_sl # 默认回退到 ATR
        
        if direction == TradeDirection.LONG:
            # 如果有有效的波段低点，且这个低点在合理范围内 (不是太远也不是太近)
            if swing_low and swing_low < entry_price:
                # 止损放在波段低点下方 0.5%
                potential_sl = swing_low * 0.995
                # 风险控制：不能亏超过 10%，也不能太近(<0.5 ATR)
                if (entry_price - potential_sl) > (atr * 0.5) and (entry_price - potential_sl) < (entry_price * 0.1):
                    structure_sl = potential_sl
                else:
                    structure_sl = base_sl # 结构位不合理，用 ATR
            # 取两者中较优的 (离入场价较近的那个，提高盈亏比？还是较远的那个，防扫损？)
            # 既然我们要解决"高胜率负期望"，我们应该追求高盈亏比 -> 选离入场价近的！
            # 但为了防止被秒扫，我们选 min(base_sl, structure_sl) 其实是选更宽的...
            # 不，为了盈亏比，我们应该选 max(base_sl, structure_sl) 即更高的止损价
            return max(base_sl, structure_sl) 
            
        else: # SHORT
            if swing_high and swing_high > entry_price:
                potential_sl = swing_high * 1.005
                if (potential_sl - entry_price) > (atr * 0.5) and (potential_sl - entry_price) < (entry_price * 0.1):
                    structure_sl = potential_sl
                else:
                    structure_sl = base_sl
            # 做空止损选更低的 (closer to entry)
            return min(base_sl, structure_sl)

    def detect_smart_money_signal(self, df_4h: pd.DataFrame, df_1h: pd.DataFrame,
                                  idx_4h: int, idx_1h: int) -> Dict[str, Any]:
        """SMC 信号汇总"""
        if not self.use_smc_logic:
            return {'smc_score': 0, 'smc_reasons': []}
            
        smc_info = {
            'smc_score': 0,
            'smc_reasons': [],
            'has_fvg_1h': False,
            'has_fvg_4h': False
        }
        
        # 1. FVG 检测
        fvg_1h = self.detect_fvg(df_1h, idx_1h)
        # fvg_4h = self.detect_fvg(df_4h, idx_4h) # 4H FVG 暂时只做参考，不加分，节省计算
        
        smc_info['fvg_direction_1h'] = fvg_1h.get('current_fvg_direction', 'none')
        smc_info['has_fvg_1h'] = fvg_1h.get('in_bullish_fvg', False) or fvg_1h.get('in_bearish_fvg', False)
        
        # 2. RS 检测
        rs_1h = self.calculate_relative_strength(df_1h, '1h', idx_1h)
        smc_info['rs_trend_1h'] = rs_1h.get('rs_trend', 'neutral')
        
        # 3. 波段点
        swing_1h = self.find_swing_points(df_1h, idx_1h)
        smc_info['swing_high_1h'] = swing_1h.get('swing_high')
        smc_info['swing_low_1h'] = swing_1h.get('swing_low')
        
        # 4. 计算加分
        # FVG 回踩是极强的信号
        if smc_info['has_fvg_1h']:
            smc_info['smc_score'] += 20
            smc_info['smc_reasons'].append(f"回踩1H缺口({smc_info['fvg_direction_1h']})")
            
        # RS 强势是阿尔法来源
        if smc_info['rs_trend_1h'] == 'strong':
            smc_info['smc_score'] += 15
            smc_info['smc_reasons'].append("RS强于大盘")
        elif smc_info['rs_trend_1h'] == 'weak':
            smc_info['smc_score'] += 15
            smc_info['smc_reasons'].append("RS弱于大盘")
            
        return smc_info

    # ------------------------------------------------------------------------
    # 🔥 [新增方法] VPA 量价分析 (已优化)
    # ------------------------------------------------------------------------
    def _analyze_vpa(self, df: pd.DataFrame, idx: int, direction: str) -> bool:
        """
        VPA (Volume Price Analysis) 深度量价验证
        逻辑：只有当价格突破伴随着动能和成交量的双重确认时，才允许开单。
        拒绝：无量上涨、放量滞涨、缩量阴跌。
        """
        if idx < 1: return False
        
        # 获取当前和前一根K线数据
        curr_close = df['close'].iloc[idx]
        prev_close = df['close'].iloc[idx-1]
        curr_open = df['open'].iloc[idx]
        curr_vol = df['volume'].iloc[idx]
        vol_ma = df['volume_ma'].iloc[idx] if 'volume_ma' in df.columns else curr_vol
        
        # 1. 量能基础门槛：必须放量
        # 咱们设定为 1.2倍均量，确保是有增量资金进场
        if curr_vol < vol_ma * 1.2: 
            return False
            
        # 2. 量价配合 (Effort vs Result)
        if direction == 'bullish':
            # 做多要求：价格上涨 且 必须是阳线 (收盘 > 开盘)
            # 防止"天量见顶"的墓碑线 (虽然涨了但收出长上影阴线)
            is_bullish_candle = curr_close > curr_open
            is_price_up = curr_close > prev_close
            
            if not is_bullish_candle: return False
            return is_price_up
            
        elif direction == 'bearish':
            # 做空要求：价格下跌 且 必须是阴线 (收盘 < 开盘)
            # 防止"低位承接"的锤子线
            is_bearish_candle = curr_close < curr_open
            is_price_down = curr_close < prev_close
            
            if not is_bearish_candle: return False
            return is_price_down
            
        return False

    # ------------------------------------------------------------------------
    # 🔥 [替换方法] V32.0 双轨制策略 (Trend Long + Scalp Short)
    # ------------------------------------------------------------------------
    def detect_signal(self, df_4h: pd.DataFrame, df_1h: pd.DataFrame, 
                      idx_4h: int, idx_1h: int, base_capital: float) -> Optional[Dict]:
        """
        [V33.1 顺势双轨版 - 微调优化]
        逻辑：
        1. 牛市 (Price > EMA): 只做多 (趋势突破)。禁止做空。
        2. 熊市 (Price < EMA): 只做空 (反弹衰竭)。禁止做多。
        优化点：做空门槛提高 (RSI>55)，防止空在半山腰。
        """
        if idx_4h < 200 or idx_1h < 200 or idx_4h >= len(df_4h) or idx_1h >= len(df_1h):
            return None
        
        # 1. 基础数据
        current_price = df_1h['close'].iloc[idx_1h]
        ema_trend = df_1h['ema_trend'].iloc[idx_1h] if 'ema_trend' in df_1h.columns else current_price
        
        # 乖离率 (Bias)
        bias_pct = (current_price - ema_trend) / ema_trend if ema_trend > 0 else 0
        
        # 核心判别：牛熊分界
        is_bull_market = current_price >= ema_trend

        # 指标获取
        rsi = df_1h['rsi'].iloc[idx_1h]
        adx_series = df_1h['adx'].iloc[idx_1h-3:idx_1h+1]
        current_adx = adx_series.iloc[-1] if len(adx_series) > 0 else 0
        vol_ratio = df_1h['volume_ratio'].iloc[idx_1h]
        
        # SMC 信息
        smc_info = self.detect_smart_money_signal(df_4h, df_1h, idx_4h, idx_1h)

        # 初始化变量
        score = 0
        reasons = []
        direction = None
        stop_loss = 0.0
        take_profit = 0.0
        risk_weight = 1.0 # 默认满仓
        rr_target = 2.0
        
        # ======================================================
        # 🐂 牛市区域：只做多 (Long Only)
        # ======================================================
        if is_bull_market:
            # 策略：趋势突破 (ADX > 30)
            # 1. 乖离率保护：Bias > 10% 不追高
            if bias_pct < 0.10: 
                # 2. 动能：ADX 强
                if current_adx >= 30:
                    # 3. 趋势共振
                    alignment = self.check_multi_timeframe_alignment(df_4h, df_1h, idx_4h, idx_1h)
                    if alignment['aligned'] and alignment['direction'] == 'bullish':
                        # 4. 辅助过滤
                        if 1.2 <= vol_ratio <= 6.0 and rsi < 80:
                            # ---> 开多信号
                            score = 85
                            direction = TradeDirection.LONG
                            reasons = ["牛市:趋势多", f"ADX:{current_adx:.1f}", "位置安全"]
                            risk_weight = 1.0 # 牛市重拳出击
                            
                            # 止损：宽幅 (ATR)
                            swing_info = self.find_swing_points(df_1h, idx_1h)
                            atr = df_1h['atr'].iloc[idx_1h]
                            stop_loss = self.calculate_dynamic_stop_loss(direction, current_price, swing_info.get('swing_high'), swing_info.get('swing_low'), atr)
                            rr_target = self.min_rr_ratio

        # ======================================================
        # 🐻 熊市区域：只做空 (Short Only)
        # ======================================================
        else: # is_bull_market == False
            # 策略：反弹衰竭 (Sell the Rally)
            # 注意：这里不再用 ADX > 30 追空，而是等反弹
            
            # 1. 反弹确认 (V33.1 优化：RSI门槛从50提高到55)
            # 在熊市里，RSI > 55 意味着反弹比较充分了，此时衰竭概率大
            # 或者 价格回抽到了 EMA 附近 (Bias > -0.03, 即距离均线不到 3%)
            is_rebound = rsi > 55 or (bias_pct > -0.03)
            
            if is_rebound:
                # 2. 阻力确认：不能涨太猛 (RSI 不能 > 70，否则可能反转)
                if rsi < 70:
                    # 3. 形态确认 (简单版)：收阴线 (当前收盘 < 开盘) 表示反弹受阻
                    # (需要在 df_1h 里取 open, 假设 df_1h 有 'open' 列，通常都有)
                    current_open = df_1h['open'].iloc[idx_1h]
                    if current_price < current_open:
                        # ---> 开空信号
                        score = 80
                        direction = TradeDirection.SHORT
                        reasons = ["熊市:反弹空", f"RSI:{rsi:.1f}", "阻力确认"]
                        risk_weight = 0.5 # 熊市轻仓喝汤 (半仓)
                        
                        # 止损：(V33.1 优化：稍微收紧止损 ATR 1.5 -> 1.2)
                        # 让止损更灵敏，不对劲就跑
                        atr = df_1h['atr'].iloc[idx_1h]
                        stop_loss = current_price + (atr * 1.2)
                        rr_target = 2.0 # 正常盈亏比

        # ======================================================
        # 🚦 最终信号生成
        # ======================================================
        if direction is not None:
            risk_dist = abs(current_price - stop_loss)
            if risk_dist == 0: risk_dist = current_price * 0.01
            
            if direction == TradeDirection.LONG:
                take_profit = current_price + (risk_dist * rr_target)
            else:
                take_profit = current_price - (risk_dist * rr_target)
            
            rr = abs(take_profit - current_price) / risk_dist
            
            return {
                'direction': direction,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'score': score,
                'reasons': reasons,
                'smc_info': smc_info,
                'rr_ratio': rr,
                'market_regime': 'Bull' if is_bull_market else 'Bear',
                'atr': df_1h['atr'].iloc[idx_1h],
                'risk_factor': risk_weight, # 🔥 风险权重 (多1.0, 空0.5)
                # 日志字段
                'adx': current_adx,
                'rsi': rsi,
                'vol_ratio': vol_ratio,
                'is_bull': is_bull_market
            }
            
        return None
    
   
    def detect_signal_with_realistic_entry(self, df_1h: pd.DataFrame, idx_1h: int, slippage: float) -> Dict[str, float]:
        """
        模拟真实交易环境：获取下一根K线的 Open 价作为入场价
        """
        # 检查是否有下一根K线
        if idx_1h + 1 >= len(df_1h):
            return None # 已经是最后一根K线，无法得知下一根开盘价，放弃开仓
            
        # 决策价格 = 下一根K线的 Open
        next_open = df_1h['open'].iloc[idx_1h + 1]
        
        # 加上滑点
        # 如果前面判断是做多，这里加滑点；如果是做空，减滑点。
        # 可以在外部判断，这里简单处理统一返回基础价格，外部处理滑点。
        return next_open
# ==========================================
# 代币筛选器（调试增强版：完整逻辑+日志）
# ==========================================

class SmartMoneyTokenScreener:
    """聪明钱代币筛选器 (带调试日志功能)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.screening_weights = config.get('screening_weights', {
            'signal_score': 0.40,
            'trend_strength': 0.25,
            'momentum': 0.15,
            'risk_reward': 0.10,
            'volume_confirmation': 0.10,
        })
        self.min_signal_threshold = config.get('min_signal_threshold', 80)
        self.use_smc_logic = config.get('use_smc_logic', False)
        
        # 记录上一次打印日志的时间，避免同一小时重复打印
        self.last_log_time = None

    def calculate_token_score(self, symbol: str, signal: Dict[str, Any], 
                            df_4h: pd.DataFrame, df_1h: pd.DataFrame,
                            idx_4h: int, idx_1h: int) -> Dict[str, Any]:
        """计算代币的综合评分 (包含完整原版逻辑)"""
        if signal is None:
            return None
        
        signal_score = signal['score']
        signal_score_normalized = min(signal_score / 100.0, 1.0)
        
        # 安全获取指标值
        adx_1h = df_1h['adx'].iloc[idx_1h] if 'adx' in df_1h.columns and idx_1h < len(df_1h) else 0
        adx_4h = df_4h['adx'].iloc[idx_4h] if 'adx' in df_4h.columns and idx_4h < len(df_4h) else 0
        
        close_1h = df_1h['close'].iloc[idx_1h] if idx_1h < len(df_1h) else 0
        ema_fast_1h = df_1h['ema_fast'].iloc[idx_1h] if 'ema_fast' in df_1h.columns and idx_1h < len(df_1h) else close_1h
        ema_slow_1h = df_1h['ema_slow'].iloc[idx_1h] if 'ema_slow' in df_1h.columns and idx_1h < len(df_1h) else close_1h
        ema_trend_1h = df_1h['ema_trend'].iloc[idx_1h] if 'ema_trend' in df_1h.columns and idx_1h < len(df_1h) else close_1h
        
        close_4h = df_4h['close'].iloc[idx_4h] if idx_4h < len(df_4h) else 0
        ema_fast_4h = df_4h['ema_fast'].iloc[idx_4h] if 'ema_fast' in df_4h.columns and idx_4h < len(df_4h) else close_4h
        ema_slow_4h = df_4h['ema_slow'].iloc[idx_4h] if 'ema_slow' in df_4h.columns and idx_4h < len(df_4h) else close_4h
        ema_trend_4h = df_4h['ema_trend'].iloc[idx_4h] if 'ema_trend' in df_4h.columns and idx_4h < len(df_4h) else close_4h
        
        trend_score = 0
        reasons = []
        
        # --- 1. 趋势强度评分 ---
        if signal['direction'] == TradeDirection.LONG:
            if close_1h > ema_fast_1h > ema_slow_1h > ema_trend_1h:
                trend_score += 40
                reasons.append("1H完美多头排列")
            elif close_1h > ema_fast_1h > ema_slow_1h:
                trend_score += 30
                reasons.append("1H多头排列")
            
            if close_4h > ema_fast_4h > ema_slow_4h > ema_trend_4h:
                trend_score += 40
                reasons.append("4H完美多头排列")
            elif close_4h > ema_fast_4h > ema_slow_4h:
                trend_score += 30
                reasons.append("4H多头排列")
        else:
            if close_1h < ema_fast_1h < ema_slow_1h < ema_trend_1h:
                trend_score += 40
                reasons.append("1H完美空头排列")
            elif close_1h < ema_fast_1h < ema_slow_1h:
                trend_score += 30
                reasons.append("1H空头排列")
            
            if close_4h < ema_fast_4h < ema_slow_4h < ema_trend_4h:
                trend_score += 40
                reasons.append("4H完美空头排列")
            elif close_4h < ema_fast_4h < ema_slow_4h:
                trend_score += 30
                reasons.append("4H空头排列")
        
        adx_score = min((adx_1h + adx_4h) / 2.0 / 50.0, 1.0) * 20 if adx_1h > 0 and adx_4h > 0 else 0
        trend_score += adx_score
        
        trend_strength_normalized = min(trend_score / 100.0, 1.0)
        
        # --- 2. 动量评分 ---
        momentum_5_1h = df_1h['momentum_5'].iloc[idx_1h] if 'momentum_5' in df_1h.columns and idx_1h < len(df_1h) else 0
        momentum_10_1h = df_1h['momentum_10'].iloc[idx_1h] if 'momentum_10' in df_1h.columns and idx_1h < len(df_1h) else 0
        momentum_5_4h = df_4h['momentum_5'].iloc[idx_4h] if 'momentum_5' in df_4h.columns and idx_4h < len(df_4h) else 0
        
        momentum_score = 0
        if signal['direction'] == TradeDirection.LONG:
            if momentum_5_1h > 0: momentum_score += 20
            if momentum_10_1h > 0: momentum_score += 15
            if momentum_5_4h > 0: momentum_score += 25
        else:
            if momentum_5_1h < 0: momentum_score += 20
            if momentum_10_1h < 0: momentum_score += 15
            if momentum_5_4h < 0: momentum_score += 25
        
        momentum_normalized = min(momentum_score / 60.0, 1.0) if momentum_score > 0 else 0
        
        # --- 3. 盈亏比评分 ---
        rr_ratio = signal.get('rr_ratio', 1.0)
        rr_normalized = min(rr_ratio / 5.0, 1.0) if rr_ratio > 0 else 0
        
        # --- 4. 成交量评分 ---
        volume_ratio_1h = df_1h['volume_ratio'].iloc[idx_1h] if 'volume_ratio' in df_1h.columns and idx_1h < len(df_1h) else 1.0
        volume_ratio_4h = df_4h['volume_ratio'].iloc[idx_4h] if 'volume_ratio' in df_4h.columns and idx_4h < len(df_4h) else 1.0
        
        volume_score = 0
        if volume_ratio_1h > 1.2: volume_score += 30
        elif volume_ratio_1h > 1.0: volume_score += 15
        
        if volume_ratio_4h > 1.2: volume_score += 30
        elif volume_ratio_4h > 1.0: volume_score += 15
        
        volume_normalized = min(volume_score / 60.0, 1.0) if volume_score > 0 else 0
        
        # --- 5. 计算总分 ---
        composite_score = (
            signal_score_normalized * self.screening_weights['signal_score'] +
            trend_strength_normalized * self.screening_weights['trend_strength'] +
            momentum_normalized * self.screening_weights['momentum'] +
            rr_normalized * self.screening_weights['risk_reward'] +
            volume_normalized * self.screening_weights['volume_confirmation']
        ) * 100
        
        # --- 6. SMC 额外加分 ---
        smc_bonus = 0
        if self.use_smc_logic and 'smc_info' in signal:
            smc_info = signal['smc_info']
            if smc_info.get('has_fvg_1h', False):
                smc_bonus += 20
                direction = smc_info.get('fvg_direction_1h', 'none')
                reasons.append(f"1H {direction} FVG")
            if smc_info.get('has_fvg_4h', False):
                smc_bonus += 15
                direction = smc_info.get('fvg_direction_4h', 'none')
                reasons.append(f"4H {direction} FVG")
            if smc_info.get('rs_trend_1h') == 'strong' and signal['direction'] == TradeDirection.LONG:
                smc_bonus += 15
                reasons.append("1H 相对强弱强势")
            elif smc_info.get('rs_trend_1h') == 'weak' and signal['direction'] == TradeDirection.SHORT:
                smc_bonus += 15
                reasons.append("1H 相对强弱弱势")
            
            if self.config.get('use_dynamic_risk', False):
                smc_bonus += 10
                reasons.append("动态风控启用")
        
        composite_score += smc_bonus
        
        return {
            'symbol': symbol,
            'direction': signal['direction'],
            'signal': signal,
            'composite_score': composite_score,
            'smc_bonus': smc_bonus,
            'component_scores': {
                'signal_score': signal_score,
                'trend_strength': trend_score,
                'momentum': momentum_score,
                'risk_reward': rr_ratio,
                'volume_score': volume_score
            },
            'reasons': reasons,
            'original_signal': signal
        }

    def select_best_token(self, token_scores: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """从多个代币中选出最佳的交易机会"""
        if not token_scores:
            return None
        
        token_scores.sort(key=lambda x: x['composite_score'], reverse=True)
        best_token = token_scores[0]
        
        if best_token['composite_score'] >= self.min_signal_threshold:
            # 调试输出：找到了一个合格的币
            # print(f"✅ 选中: {best_token['symbol']} 分数: {best_token['composite_score']:.2f} (>{self.min_signal_threshold})")
            return best_token
        
        return None

    def screen_tokens(self, data_cache: Dict[str, Any], check_time: datetime, 
                      signal_detector: Any) -> Optional[Dict[str, Any]]:
        """筛选所有代币，找出最佳交易机会 (带调试日志 - 增强版)"""
        token_scores = []
        
        # 控制日志频率：每天只打印一次关键检查信息 (用于低分信号)
        should_print = False
        if check_time.hour == 0 and check_time.minute == 0:
            should_print = True
            # print(f"\n--- 检查时间点: {check_time} ---")

        for symbol in self.config['symbols']:
            if symbol not in data_cache: continue
            if '4h' not in data_cache[symbol] or '1h' not in data_cache[symbol]: continue
            
            df_4h = data_cache[symbol]['4h']
            df_1h = data_cache[symbol]['1h']
            
            mask_4h = df_4h.index <= check_time
            mask_1h = df_1h.index <= check_time
            
            if mask_4h.sum() < 210 or mask_1h.sum() < 210:
                continue
            
            idx_4h = mask_4h.sum() - 1
            idx_1h = mask_1h.sum() - 1
            
            if idx_4h >= len(df_4h) or idx_1h >= len(df_1h): continue
            
            # 1. 检测信号
            signal = signal_detector.detect_signal(
                df_4h, df_1h, idx_4h, idx_1h, 0
            )
            
            if signal:
                # 2. 检查基础信号分数
                if signal.get('score', 0) >= signal_detector.min_signal_score:
                    # 3. 计算综合筛选分数
                    token_score = self.calculate_token_score(
                        symbol, signal, df_4h, df_1h, idx_4h, idx_1h
                    )
                    
                    if token_score:
                        token_scores.append(token_score)
                else:
                    # 🔥 [新增] 记录被分数卡住的信号 (差一点就开单的)
                    score = signal.get('score', 0)
                    
                    # 策略：如果分数不错 (>60)，即使不是 0点 也打印出来，方便复盘
                    if score > 60:
                        reasons_list = signal.get('reasons', [])
                        reasons_str = "+".join(reasons_list)
                        print(f"🚫 [过滤] {check_time} {symbol} 分数不足: {score} < {signal_detector.min_signal_score} (理由: {reasons_str})")
                    
                    # 只有在每天 0点 时才打印低分垃圾信号
                    elif should_print:
                        print(f"[{symbol}] 信号分数不足: {score} < {signal_detector.min_signal_score}")
            else:
                # 调试：无信号
                # if should_print: print(f"[{symbol}] 无信号 (趋势/指标不满足)")
                pass

        if not token_scores:
            return None
        
        best_token = self.select_best_token(token_scores)
        
        if best_token:
            # 排序找到排名
            sorted_tokens = sorted(token_scores, key=lambda x: x['composite_score'], reverse=True)
            for i, token in enumerate(sorted_tokens):
                if token['symbol'] == best_token['symbol']:
                    best_token['rank'] = i + 1
                    best_token['total_tokens'] = len(token_scores)
                    break
            
            # 打印选中的代币详情
            # print(f"🚀 [{check_time}] 开仓: {best_token['symbol']} 方向:{best_token['direction']} 分数:{best_token['composite_score']:.1f}")
            return best_token
            
        return None

# ==========================================
# ➕ 【新增】 动态风控与漂移检测模块 (完整版)
# ==========================================

class DynamicRiskBudget:
    """动态风险预算管理器：根据年度绩效调整次年仓位"""
    def __init__(self, initial_capital: float, lookback_years: int = 1):
        self.initial_capital = initial_capital
        self.lookback_years = lookback_years
        self.yearly_pnl = {}  # 记录每年的盈亏 {2021: 0.5, 2022: -0.1}
        
    def record_year_performance(self, year: int, pnl_ratio: float):
        """记录年度表现"""
        self.yearly_pnl[year] = pnl_ratio
        
    def adjust_for_year_performance(self, current_time: datetime, data_available_until: datetime = None) -> float:
        """根据可用的历史数据调整仓位 (加权平均版)"""
        current_year = current_time.year
        
        # 1. 获取过去已完成年份 (严格防止未来函数)
        available_years = sorted([y for y in self.yearly_pnl.keys() if y < current_year])
        
        # 2. 数据不足保护：如果历史不足 2 年，不调整，保持中性
        if len(available_years) < 2:
            return 1.0
            
        # 3. 选取最近 3 年
        recent_years = available_years[-3:]
        returns = [self.yearly_pnl[y] for y in recent_years]
        
        # 4. 计算加权平均 (越近的年份权重越大)
        if len(returns) == 3:
            weights = [0.2, 0.3, 0.5]
        elif len(returns) == 2:
            weights = [0.4, 0.6]
        else:
            weights = [1.0]
            
        avg_return = sum(r * w for r, w in zip(returns, weights))
        
        # 5. 调整系数
        if avg_return > 0.25:   # 平均年化 > 25%
            return 1.2          # 激进模式
        elif avg_return < -0.05: # 平均年化 < -5%
            return 0.5          # 防御模式
        elif avg_return < 0.10: # 平均年化 < 10%
            return 0.8          # 稍微降仓
            
        return 1.0

    def can_trade(self, risk_amount: float) -> bool:
        return True 

class ParameterDriftDetector:
    """参数漂移检测器：检测策略是否失效"""
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.pnl_history = []
        
    def update(self, trade_pnl_percent: float):
        self.pnl_history.append(trade_pnl_percent)
        if len(self.pnl_history) > self.window_size:
            self.pnl_history.pop(0)
            
    def check_performance(self) -> Dict:
        if len(self.pnl_history) < 20:
            return {'status': 'OK', 'recommendation': 'Wait for data'}
            
        wins = sum(1 for x in self.pnl_history if x > 0)
        win_rate = wins / len(self.pnl_history)
        avg_pnl = sum(self.pnl_history) / len(self.pnl_history)
        
        # 漂移判定：胜率极低且期望为负
        if win_rate < 0.25 and avg_pnl < 0:
            return {'status': 'WARNING', 'recommendation': '策略可能失效，建议减半'}
        elif win_rate < 0.15:
            return {'status': 'CRITICAL', 'recommendation': '立即停止交易'}
            
        return {'status': 'OK', 'recommendation': 'Normal'}

# ==========================================
# 统一的回测引擎（修复版：防崩坏）
# ==========================================

class UnifiedBacktestEngine:
    """
    统一回测引擎 (修复增强版 V24.1)
    功能：
    1. 支持静默模式 (verbose=False)，用于优化器高频调用时不输出日志
    2. 修复了资金归零时的崩溃 BUG
    3. 统一了手动回测和滚动回测的执行逻辑
    """
    
   # 🔥 【替换】 __init__ 方法，增加 skip_indicator_calc 参数
    def __init__(self, config: Dict[str, Any], data_cache: Dict[str, Any] = None, 
                 verbose: bool = True, skip_indicator_calc: bool = False,
                 inherited_positions: List[Any] = None): # 🔥 修改点：新增继承持仓参数
        self.config = config
        self.verbose = verbose 
        # 新增：如果为 True，引擎将直接信任传入的 data_cache 已包含指标，不再重新计算
        self.skip_indicator_calc = skip_indicator_calc 
        
        # 初始化核心组件
        self.signal_detector = SmartMoneySignalDetector(config)
        self.token_screener = SmartMoneyTokenScreener(config)
        self.position_calculator = SmartPositionManager(config)
        
        # 使用传入的数据缓存
        self.data_cache = data_cache if data_cache else {}
        
        # 如果启用了SMC逻辑，设置BTC数据 (保留原逻辑)
        if config.get('use_smc_logic', False) and 'BTC/USDT' in self.data_cache:
            btc_data = {}
            if '1h' in self.data_cache['BTC/USDT']:
                btc_data['1h'] = self.data_cache['BTC/USDT']['1h']
            if '4h' in self.data_cache['BTC/USDT']:
                btc_data['4h'] = self.data_cache['BTC/USDT']['4h']
            self.signal_detector.set_btc_data(btc_data)
        
        # ✅ [修复] 严格初始化回测状态，防止多次调用时的状态污染
        self.initial_capital = config.get('initial_capital', 10000)
        self.total_capital = self.initial_capital
        
        # 🔥 修改点：处理跨月继承持仓逻辑
        self.positions: List[Trade] = []      
        self.used_margin = 0 
        
        if inherited_positions:
            self.positions = inherited_positions
            # 重新计算被老单子占用的保证金
            for trade in self.positions:
                self.used_margin += trade.margin_used
                if self.verbose:
                    print(f"🔄 [接力] 继承持仓: {trade.symbol} | 入场: {trade.entry_price} | 浮盈: {trade.pnl:.2f}U")
        
        # 计算可用资金 = 总权益 - 已用保证金
        self.available_capital = self.total_capital - self.used_margin
        
        self.closed_trades: List[Trade] = []  # 已平仓历史：必须显式清空
        self.equity_curve = [self.total_capital] # 资金曲线：使用当前权益（含浮盈）
        self.trade_counter = len(self.positions) # 计数器顺延
        self.risk_history = []
        self.anomaly_report = [] # 异常报告清空

        # 初始化风控管理器 (确保风控计数器重置)
        self.risk_manager = RealTimeRiskManager(self.initial_capital)

        # 初始化高级风控模块
        self.enable_annual_adjustment = config.get('enable_annual_adjustment', False) 
        self.enable_drift_detection = config.get('enable_drift_detection', True) 
        
        self.risk_budget_manager = DynamicRiskBudget(self.initial_capital)
        self.drift_detector = ParameterDriftDetector(window_size=50)

    def _calculate_and_cache_indicators(self):
        """
        强制指标预计算与清洗
        """
        # 如果标记为跳过计算（说明数据来自 MasterDataManager），直接返回
        if self.skip_indicator_calc:
            if self.verbose:
                # 仅在调试时打印，避免刷屏
                pass 
            return

        # 否则执行原有的计算逻辑 (保留原有逻辑处理手动模式下的未计算数据)
        if not hasattr(self, '_indicators_cached') or not self._indicators_cached:
            processed_cache = {}
            for symbol, timeframes in self.data_cache.items():
                processed_cache[symbol] = {}
                for timeframe, df in timeframes.items():
                    if timeframe in ['1h', '4h']:
                        df_copy = df.copy()
                        try:
                            df_calculated = self.signal_detector.calculate_indicators(df_copy)
                            processed_cache[symbol][timeframe] = df_calculated
                        except Exception as e:
                            if self.verbose:
                                print(f"❌ [Engine] 指标计算失败 {symbol} {timeframe}: {e}")
                            processed_cache[symbol][timeframe] = df 
                    else:
                        processed_cache[symbol][timeframe] = df
            
            self.data_cache = processed_cache
            self._indicators_cached = True
    def _execute_close(self, trade, price, time, reason):
        """
        执行平仓 (V31.0 最终修复版：MFE修复 + 硬止损适配 + R值计算)
        """
        # ==========================
        # 1. 计算滑点 (Slippage)
        # ==========================
        base_slippage = self.config.get('slippage', 0.001)
        
        # 针对不同平仓原因调整滑点
        # 爆仓或硬止损通常意味着行情剧烈或紧急离场，滑点会更大
        eff_slippage = base_slippage
        reason_str = str(reason or "")
        
        if "爆仓" in reason_str:
            eff_slippage = base_slippage * 5.0
        elif "硬止损" in reason_str or "HardStop" in reason_str:
            eff_slippage = base_slippage * 2.0  # 硬止损给 2倍滑点，模拟追单磨损
        
        # 计算真实成交价
        if trade.direction == TradeDirection.LONG:
            real_exit_price = price * (1 - eff_slippage)
            pnl_gross = (real_exit_price - trade.entry_price) * trade.position_size
        else:
            real_exit_price = price * (1 + eff_slippage)
            pnl_gross = (trade.entry_price - real_exit_price) * trade.position_size
            
        # 扣除费用
        fee_rate = self.config.get('fee_rate', 0.0004)
        exit_fee = real_exit_price * trade.position_size * fee_rate
        pnl_net = pnl_gross - exit_fee - trade.funding_fees
        
        # ==========================
        # 2. 资金结算
        # ==========================
        self.available_capital += (trade.margin_used + pnl_net)
        self.used_margin -= trade.margin_used
        self.total_capital = self.available_capital + self.used_margin
        
        # ==========================
        # 3. 记录交易状态
        # ==========================
        trade.exit_time = time
        trade.exit_price = real_exit_price
        trade.exit_reason = reason
        trade.pnl += pnl_net 
        
        # 计算初始保证金 (防止除零)
        if trade.remaining_ratio < 1.0:
             initial_margin = (trade.position_value / trade.remaining_ratio) / trade.leverage
        else:
             initial_margin = trade.position_value / trade.leverage
        
        if initial_margin > 0: 
            trade.pnl_percent = (trade.pnl / initial_margin) * 100 
        else: 
            trade.pnl_percent = 0
        
        self.closed_trades.append(trade)
        if trade in self.positions:
            self.positions.remove(trade)
        
        # 更新风控模块状态
        self.risk_manager.update_after_trade(trade.pnl)
        if hasattr(self, 'drift_detector'):
            self.drift_detector.update(trade.pnl_percent/100.0)
        
        # ==========================
        # 4. 🔥 [修复] 极值与R值计算
        # ==========================
        if self.verbose:
            # A. 持仓时间
            duration = time - trade.entry_time
            hours = max(0.0, duration.total_seconds() / 3600)
            
            # B. MFE/MAE 计算 (修复了做空取值错误的 Bug)
            h = getattr(trade, 'highest_price', trade.entry_price)
            l = getattr(trade, 'lowest_price', trade.entry_price)
            
            current_highest = max(h, real_exit_price)
            current_lowest = min(l, real_exit_price)

            if trade.direction == TradeDirection.LONG:
                mfe_pct = (current_highest - trade.entry_price) / trade.entry_price * 100
                mae_pct = (current_lowest - trade.entry_price) / trade.entry_price * 100
            else:
                # 做空：价格越低(lowest)收益越高(MFE)，价格越高(highest)亏损越大(MAE)
                mfe_pct = (trade.entry_price - current_lowest) / trade.entry_price * 100
                mae_pct = (trade.entry_price - current_highest) / trade.entry_price * 100
            
            # C. 利润回吐计算 (Retracement)
            retracement = 0
            if mfe_pct > 0:
                current_pnl_pct_raw = (pnl_net / initial_margin) * 100
                retracement = (mfe_pct * trade.leverage) - current_pnl_pct_raw
                if retracement < 0: retracement = 0
            
            # D. 实现盈亏比 (Realized R-Multiple)
            if hasattr(trade, 'initial_stop_loss') and trade.initial_stop_loss > 0:
                # 确保风险额度为正数 (abs)
                risk_amt = abs(trade.entry_price - trade.initial_stop_loss) * trade.position_size
            else:
                risk_amt = initial_margin * 0.05
            
            r_multiple = pnl_net / risk_amt if risk_amt > 0 else 0

            # E. 打印日志
            pnl_icon = "🟢" if trade.pnl > 0 else "🔴"
            # 如果是硬止损，加个特殊的标记
            reason_display = f"🚨 {reason}" if "硬止损" in reason_str or "HardStop" in reason_str else reason

            log_msg = (
                f"{pnl_icon} [平仓] {time} | {trade.symbol} | {reason_display}\n"
                f"   💰 盈亏: ${trade.pnl:+.2f} ({trade.pnl_percent:+.2f}%) | ⚖️ R值: {r_multiple:+.1f}R\n"
                f"   ⏱️ 持仓: {hours:.1f}h | 🌊 MFE(最高): {mfe_pct*trade.leverage:.1f}% | 🩸 MAE(最痛): {mae_pct*trade.leverage:.1f}%\n"
                f"   ↩️ 利润回吐: {retracement:.1f}%"
            )
            ui_log(log_msg)
    def _diagnose_btc_environment(self, current_time):
        """
        诊断当前的 BTC 市场环境
        返回: (是否安全(bool), 环境描述(str))
        """
        # 尝试获取 BTC 数据
        btc_df = self.data_cache.get('BTC/USDT')
        if btc_df is None or btc_df.empty:
            return True, "No_Data" # 没数据默认放行，但标记

        # 找到当前时间对应的数据行
        # 注意：这里需要确保时间索引对齐，简单起见用 asof 或直接查找
        try:
            # 假设 btc_df 索引是 datetime
            if current_time not in btc_df.index:
                # 尝试找最近的一个过去时间点 (防止对不齐)
                idx_loc = btc_df.index.get_indexer([current_time], method='pad')[0]
                if idx_loc == -1: return True, "Data_Miss"
                row = btc_df.iloc[idx_loc]
            else:
                row = btc_df.loc[current_time]
        except:
            return True, "Data_Err"

        # --- 诊断逻辑 ---
        price = row['close']
        ema_slow = row.get('ema_slow', row['close']) # 假设你有算 EMA
        adx = row.get('adx', 0)
        
        # 判定
        if price < ema_slow:
            # 价格在慢线之下：熊市或暴跌
            if adx > 30: 
                return False, "Crash" # 暴跌中 (动能强 + 线下) -> 🔴 禁止做多
            else:
                return True, "Bear_Chop" # 阴跌震荡 -> 🟡 勉强放行
        else:
            # 价格在慢线之上：牛市
            if adx > 25:
                return True, "Bull_Run" # 主升浪 -> 🟢 强烈推荐
            else:
                return True, "Bull_Rest" # 牛市回调 -> 🟢 安全

    def _calculate_fusion_stop_loss(self, df, idx, direction, atr_stop_price):
        """
        计算融合止损：取 ATR 和 结构止损 中 更宽(更安全) 的那个
        """
        # 1. 计算结构止损 (Swing Stop)
        lookback = 15 # 回溯 15 根 K 线找前低
        start = max(0, idx - lookback)
        window = df.iloc[start:idx+1]
        
        swing_stop = 0.0
        source = "ATR" # 默认来源
        final_sl = atr_stop_price
        
        if direction == TradeDirection.LONG:
            swing_low = window['low'].min()
            swing_stop = swing_low * 0.998 # 留一点缓冲
            
            # 融合逻辑：做多止损，谁更低(离价格更远)选谁
            if swing_stop < atr_stop_price:
                final_sl = swing_stop
                source = "Structure" # 结构止损生效
            else:
                source = "ATR" # ATR 更宽，用 ATR (防插针)
                
        else: # SHORT
            swing_high = window['high'].max()
            swing_stop = swing_high * 1.002
            
            # 融合逻辑：做空止损，谁更高(离价格更远)选谁
            if swing_stop > atr_stop_price:
                final_sl = swing_stop
                source = "Structure"
            else:
                source = "ATR"
                
        return final_sl, source            
    
    def _check_and_open_new_positions(self, check_time):
        """
        检查并执行开新仓逻辑 (V27.2 - 修复重复开单 + 补全日志信息)
        """
        # ==========================
        # 1. 基础环境检查
        # ==========================
        # 检查总持仓数量限制
        if len(self.positions) >= self.config.get('max_positions', 1):
            return

        # 漂移检测 (保留原有逻辑)
        drift_multiplier = 1.0
        if self.enable_drift_detection and len(self.closed_trades) > 10:
            drift_status = self.drift_detector.check_performance()
            if drift_status['status'] == 'CRITICAL': drift_multiplier = 0.0 
            elif drift_status['status'] == 'WARNING': drift_multiplier = 0.5
        
        if drift_multiplier <= 0: return 

        # ==========================
        # 2. 筛选最佳币种
        # ==========================
        best_token = self.token_screener.screen_tokens(
            self.data_cache, check_time, self.signal_detector
        )
        
        if not best_token: return

        # ==========================
        # 3. 数据提取与准备
        # ==========================
        signal = best_token['original_signal']
        symbol = best_token['symbol']
        direction = signal['direction']
        
        # 🔥🔥🔥【修复 1】防止同币种重复开单 (Anti-Stacking) 🔥🔥🔥
        # 检查当前持仓中是否已经有了这个币
        current_holdings = [p.symbol for p in self.positions]
        if symbol in current_holdings:
            return # 已经持有该币种，不再重复开仓，防止风险集中
        
        # 提取入场基因
        reasons_list = signal.get('reasons', [])
        reasons_str = " + ".join(reasons_list)
        score = signal.get('score', 0)
        
        if symbol not in self.data_cache or '1h' not in self.data_cache[symbol]: return
        df_1h = self.data_cache[symbol]['1h']
        
        mask = df_1h.index <= check_time
        if mask.sum() == 0: return
        idx = mask.sum() - 1
        
        current_adx = df_1h['adx'].iloc[idx] if 'adx' in df_1h.columns else 25
        current_atr = df_1h['atr'].iloc[idx] if 'atr' in df_1h.columns else 0
        prev_adx = df_1h['adx'].iloc[idx-1] if idx > 0 and 'adx' in df_1h.columns else current_adx
        
        # ==========================
        # 4. 核心过滤逻辑 (ADX/BTC防崩/4H共振)
        # ==========================
        if self.config.get('enable_adx_meltdown', False):
            if current_adx > self.config.get('adx_meltdown_threshold', 60): return 

        if self.config.get('enable_4h_resonance', False):
            if symbol in self.data_cache and '4h' in self.data_cache[symbol]:
                df_4h = self.data_cache[symbol]['4h']
                mask_4h = df_4h.index <= check_time
                if mask_4h.sum() > 0:
                    row_4h = df_4h.iloc[mask_4h.sum() - 1]
                    trend_ema = row_4h.get('ema_trend', row_4h.get('ema_slow', row_4h['close']))
                    is_bullish = row_4h['close'] > trend_ema
                    if direction == TradeDirection.LONG and not is_bullish: return
                    if direction == TradeDirection.SHORT and is_bullish: return

        btc_status = "N/A"
        if self.config.get('use_btc_protection', False) and symbol != 'BTC/USDT':
            is_safe, btc_status = self._diagnose_btc_environment(check_time)
            if not is_safe: return 

        # ==========================
        # 5. 执行开仓计算
        # ==========================
        if idx + 1 >= len(df_1h): return 
        
        # 获取下一根K线开盘价作为入场价
        next_bar_open = df_1h['open'].iloc[idx + 1]
        entry_timestamp = df_1h.index[idx + 1] # 入场时间

        if next_bar_open <= 0: return

        slippage = self.config.get('slippage', 0.001)
        if direction == TradeDirection.LONG:
            entry_price = next_bar_open * (1 + slippage)
            action_str = "开多 📈" 
        else:
            entry_price = next_bar_open * (1 - slippage)
            action_str = "开空 📉" 

        atr_val = current_atr if current_atr > 0 else entry_price * 0.01
        
        # ------------------------------------------------------
        # 止损计算 (ATR + 融合 + 硬止损)
        # ------------------------------------------------------
        
        # 1. 基础 ATR 止损
        atr_mult = self.config.get('stop_loss_atr', 2.0)
        if direction == TradeDirection.LONG:
            raw_sl = entry_price - (atr_val * atr_mult)
        else:
            raw_sl = entry_price + (atr_val * atr_mult)
        
        final_sl = raw_sl
        sl_source = "ATR"
        
        # 2. 结构止损融合 (如果开启)
        if self.config.get('use_fusion_stop_loss', True):
            final_sl, sl_source = self._calculate_fusion_stop_loss(df_1h, idx, direction, raw_sl)

        # ======================================================
        # 🔥🔥🔥 3. 硬损熔断机制 (The Hard Shield) 🔥🔥🔥
        # [关键] 必须放在最外层，不能缩进在 if use_fusion 里面！
        # ======================================================
        max_loss_pct = 0.08  # 允许最大单笔亏损 8%
        
        if direction == TradeDirection.LONG:
            hard_sl = entry_price * (1 - max_loss_pct)
            # 多头：止损价取较高者 (离进场价更近，亏损更少)
            if hard_sl > final_sl: 
                final_sl = hard_sl
                sl_source += "+HardCap"
        else:
            hard_sl = entry_price * (1 + max_loss_pct)
            # 空头：止损价取较低者 (离进场价更近，亏损更少)
            if hard_sl < final_sl: 
                final_sl = hard_sl
                sl_source += "+HardCap"
        # ======================================================

        # 止盈计算 (基于最终确定的 final_sl 计算风险距离)
        risk_dist = abs(entry_price - final_sl)
        # 防止除零错误
        if risk_dist == 0: risk_dist = entry_price * 0.01

        rr_ratio = self.config.get('min_rr_ratio', 2.5) 
        if direction == TradeDirection.LONG:
            take_profit = entry_price + (risk_dist * rr_ratio)
        else:
            take_profit = entry_price - (risk_dist * rr_ratio)
        
        # 仓位调整
        annual_multiplier = 1.0
        if self.enable_annual_adjustment:
            annual_multiplier = self.risk_budget_manager.adjust_for_year_performance(check_time)
            
        original_target = self.position_calculator.target_position_value
        final_multiplier = annual_multiplier * drift_multiplier
        if final_multiplier != 1.0:
            self.position_calculator.target_position_value = original_target * final_multiplier

        # 计算仓位 (传入经过硬止损修正后的 final_sl)
        position_data = self.position_calculator.calculate_position(
            entry_price=entry_price, 
            stop_loss=final_sl,  # <--- 关键：使用硬止损后的价格
            direction=direction, 
            current_capital=self.total_capital, 
            adx_value=current_adx,
            prev_adx_value=prev_adx, 
            atr_value=current_atr
        )
        self.position_calculator.target_position_value = original_target
        
        # ==========================
        # 6. 开仓执行与日志 (优化版：含乖离率战术面板)
        # ==========================
        if position_data.get('can_trade', False):
            margin_needed = position_data['margin_used']
            risk_amt = position_data.get('risk_amount_value', 0)
            
            can_trade = self.risk_manager.can_open_position(risk_amt, self.positions, check_time)
            
            if can_trade and margin_needed <= self.available_capital:
                trade = Trade(
                    id=self.trade_counter,
                    symbol=symbol,
                    direction=direction,
                    entry_time=entry_timestamp, 
                    entry_price=entry_price,
                    stop_loss=final_sl,
                    initial_stop_loss=final_sl, # 记录初始止损
                    take_profit=take_profit,
                    position_size=position_data['position_size'],
                    leverage=position_data['actual_leverage'],
                    margin_used=margin_needed,
                    liquidation_price=position_data['liquidation_price'],
                    position_value=position_data['position_value'],
                    signal_score=signal.get('score', 0),
                    signal_reasons=signal.get('reasons', []),
                    token_rank=best_token.get('rank', 0),
                    screening_score=best_token.get('composite_score', 0),
                    smc_info=signal.get('smc_info', {}),
                    position_data=position_data,
                    tp1_hit=False, 
                    remaining_ratio=1.0, 
                    is_breakeven=False,
                    sl_source=sl_source, 
                    btc_env=btc_status
                )
                
                trade.entry_reasons = reasons_str

                # 🔥🔥🔥 [新增] 计算乖离率用于日志显示 🔥🔥🔥
                # 乖离率 = (价格 - EMA) / EMA
                # 作用：直观显示当前价格是否偏离均线过远（追高/抄底风险）
                ema_val = df_1h['ema_trend'].iloc[idx] if 'ema_trend' in df_1h.columns else entry_price
                # 防止除零错误
                if ema_val == 0: ema_val = entry_price
                bias_pct = (entry_price - ema_val) / ema_val * 100
                
                # 定义乖离率状态颜色 (偏离超过 5% 标红显示，提醒注意风险)
                bias_color = "red" if abs(bias_pct) > 5 else "green"

                # 日志打印 (增强版战术面板)
                if self.verbose:
                    # 提取信号中的指标数据 (防止字典里没有key报错，给默认值)
                    s_adx = signal.get('adx', 0)
                    s_rsi = signal.get('rsi', 0)
                    s_vol = signal.get('vol_ratio', 0)
                    s_score = signal.get('score', 0)
                    
                    # 状态图标显示
                    is_bull = signal.get('is_bull', False)
                    status_icon = "🐮牛市" if is_bull else "🐻熊市"
                    
                    ui_log(
                        f"➕ [{action_str}] {entry_timestamp} | {symbol} | 价格: ${entry_price:.2f} <br>"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;🧬 <b>入场基因:</b> {reasons_str} <br>"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;📊 <b>战术面板:</b> <span style='color:orange'>分:{s_score}</span> | "
                        f"ADX:<b>{s_adx:.1f}</b> | Vol:<b>{s_vol:.1f}x</b> | RSI:{s_rsi:.0f} | "
                        f"Bias:<span style='color:{bias_color}'><b>{bias_pct:+.2f}%</b></span> | {status_icon} <br>"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;🛡️ 止损源: {sl_source} | 🌍 大盘: {btc_status} <br>"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;💰 仓位: ${position_data['position_value']:.0f} (Lev:{position_data['actual_leverage']:.1f}x)"
                    )
                
                self.positions.append(trade)
                open_fee = position_data.get('open_fee', 0)
                self.available_capital -= (margin_needed + open_fee)
                self.total_capital -= open_fee 
                self.used_margin += margin_needed
                self.trade_counter += 1
    def run_backtest(self) -> Dict[str, Any]:
        """
        运行回测主循环 (V26.0 旗舰版)
        🔥 新增特性：
        1. 动态时间止损 (亏损快跑，盈利多拿)
        2. 自适应 MFE 利润保护 (大赚后收紧止损，防止回撤)
        3. 解决贪婪冲突与负持仓时间
        """
        if not self.data_cache:
            self.anomaly_report.append({"time": "INIT", "type": "NO_DATA", "msg": "数据缓存为空"})
            return {}
        
        # 1. 预计算指标
        self._calculate_and_cache_indicators()
        
        # 2. 解析时间轴
        try:
            start = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
            end = datetime.strptime(self.config['end_date'], '%Y-%m-%d')
            end = end + timedelta(hours=23, minutes=59)
            check_interval = self.config.get('check_interval_hours', 1)
            check_times = pd.date_range(start, end, freq=f"{check_interval}H")
        except Exception as e:
            self.anomaly_report.append({"time": "INIT", "type": "DATE_ERROR", "msg": str(e)})
            return {}
            
        if len(check_times) == 0: return {}
        
        # 3. 状态变量
        peak_capital = self.total_capital
        max_drawdown = 0
        last_check_date = None
        current_year = None
        year_start_capital = self.total_capital

        # ==================== 回测主循环 ====================
        for check_time in check_times:
            
            # --- A. 资金熔断检测 ---
            if self.total_capital <= 0:
                msg = f"💀 账户已破产 (余额: ${self.total_capital:.2f})"
                if self.verbose: ui_log(msg)
                self.anomaly_report.append({"time": check_time, "type": "BANKRUPTCY", "msg": msg})
                # 强平
                for trade in self.positions:
                    trade.exit_time = check_time
                    trade.exit_reason = "账户破产强平"
                    trade.pnl = -trade.margin_used 
                    self.closed_trades.append(trade)
                self.positions = []
                self.equity_curve.append(0) 
                break 

            # 每日重置与年度结算
            current_date = check_time.date()
            if last_check_date != current_date:
                self.risk_manager.reset_daily()
                last_check_date = current_date

            if current_year is None: current_year = check_time.year
            if check_time.year > current_year:
                last_year_pnl_ratio = (self.total_capital - year_start_capital) / year_start_capital if year_start_capital > 0 else 0
                self.risk_budget_manager.record_year_performance(current_year, last_year_pnl_ratio)
                current_year = check_time.year
                year_start_capital = self.total_capital
            
            annual_multiplier = 1.0
            if self.enable_annual_adjustment:
                annual_multiplier = self.risk_budget_manager.adjust_for_year_performance(check_time)
            
            # 资金费率 (每8小时)
            if check_time.hour % 8 == 0 and check_time.minute == 0:
                funding_rate_per_interval = 0.0001 
                for trade in self.positions:
                    if trade.direction == TradeDirection.LONG:
                        cost = trade.position_value * funding_rate_per_interval
                        self.total_capital -= cost
                        self.available_capital -= cost
                        trade.funding_fees += cost
                if self.total_capital <= 0: continue

            # ==========================================================
            # 🔥 B. 检查持仓出场 (R/R 拯救计划：保本+趋势追踪+动态时间止损)
            # ==========================================================
            
            # 获取当前回测的时间周期 (默认为 4小时)
            check_interval = self.config.get('check_interval_hours', 4)
            is_small_timeframe = check_interval <= 1
            
            # 动态计算耐心阈值 (K线根数)
            # 4H 周期：耐心为 4根 (16小时)
            # 1H 周期：耐心为 18根 (18小时)
            patience_bars_1 = 18 if is_small_timeframe else 6  # 给 4H 策略 24小时耐心
            patience_bars_2 = 36 if is_small_timeframe else 8
            patience_bars_3 = 72 if is_small_timeframe else 12 # 24H -> 48H

            for trade in self.positions.copy():
                # [修复] 防止时间穿越
                if check_time <= trade.entry_time: continue

                symbol = trade.symbol
                if symbol not in self.data_cache or '1h' not in self.data_cache[symbol]: continue
                
                df_1h = self.data_cache[symbol]['1h']
                mask = df_1h.index <= check_time
                if mask.sum() == 0: continue
                idx = mask.sum() - 1
                curr_row = df_1h.iloc[idx]
                
                high, low, close = curr_row['high'], curr_row['low'], curr_row['close']
                current_atr = curr_row['atr'] if 'atr' in curr_row else close * 0.01

                # 更新 MFE/MAE (最大浮盈/最大浮亏)
                if not hasattr(trade, 'mfe') or trade.mfe == 0: trade.mfe = trade.entry_price
                if not hasattr(trade, 'mae') or trade.mae == 0: trade.mae = trade.entry_price
                if not hasattr(trade, 'highest_price'): trade.highest_price = trade.entry_price
                if not hasattr(trade, 'lowest_price') or trade.lowest_price == 0: trade.lowest_price = trade.entry_price

                if trade.direction == TradeDirection.LONG: 
                    trade.mfe = max(trade.mfe, high); trade.mae = min(trade.mae, low)
                    trade.highest_price = max(trade.highest_price, high)
                else:
                    trade.mfe = min(trade.mfe, low); trade.mae = max(trade.mae, high)
                    trade.lowest_price = min(trade.lowest_price, low)

                exit_price = None; exit_reason = None

                # 计算风险单元 (1R)
                # 优先使用 initial_stop_loss 以保持 R 值的一致性
                ref_sl = getattr(trade, 'initial_stop_loss', trade.stop_loss)
                risk_per_unit = abs(trade.entry_price - ref_sl)
                
                # 【防爆盾】强制设置最小风险距离
                min_risk_buffer = trade.entry_price * 0.005 
                risk_per_unit = max(risk_per_unit, min_risk_buffer)
                
                # 计算当前最高浮盈 R 倍数 (Max R Reached)
                if trade.direction == TradeDirection.LONG:
                    max_r_reached = (trade.highest_price - trade.entry_price) / risk_per_unit
                    curr_pnl_price = (close - trade.entry_price)
                else:
                    max_r_reached = (trade.entry_price - trade.lowest_price) / risk_per_unit
                    curr_pnl_price = (trade.entry_price - close)
                
                current_r = curr_pnl_price / risk_per_unit

                # ------------------------------------------------------
                # ✅ 策略修改 1: 动态保本策略 (Smart Breakeven)
                # ------------------------------------------------------
                # 4H 周期更宽容(2.0R)，1H 周期更敏捷(1.5R)
                breakeven_trigger = 1.5 if is_small_timeframe else 2.0
                
                if max_r_reached >= breakeven_trigger and not trade.is_breakeven:
                    trade.is_breakeven = True
                    # 移动止损到 开仓价 + 一点点保护垫
                    if trade.direction == TradeDirection.LONG:
                        trade.stop_loss = max(trade.stop_loss, trade.entry_price * 1.001)
                    else:
                        trade.stop_loss = min(trade.stop_loss, trade.entry_price * 0.999)
                    
                    if self.verbose:
                        ui_log(f"🛡️ [保本] {trade.symbol} 浮盈 > {breakeven_trigger}R，止损移至开仓位")

                # ------------------------------------------------------
                # ✅ 策略修改 2: 趋势追踪止损 (Let Profits Run)
                # ------------------------------------------------------
                # 只有当盈利非常丰厚 (>3.5R) 时，才开始收紧止损
                trail_activation_r = 3.5
                
                if max_r_reached >= trail_activation_r:
                    # 使用 2.5倍 ATR 作为安全垫 (收紧一点，因为已经大赚了)
                    atr_buffer = current_atr * 2.5
                    if trade.direction == TradeDirection.LONG:
                        new_sl = trade.highest_price - atr_buffer
                        if new_sl > trade.stop_loss:
                            trade.stop_loss = new_sl
                            trade.trailing_stop = new_sl
                    else:
                        new_sl = trade.lowest_price + atr_buffer
                        if new_sl < trade.stop_loss:
                            trade.stop_loss = new_sl
                            trade.trailing_stop = new_sl

                # ------------------------------------------------------
                # ✅ 策略修改 3: 动态 K 线计数止损 (Bar Counting)
                # ------------------------------------------------------
                holding_hours = (check_time - trade.entry_time).total_seconds() / 3600
                # 计算持有了多少根 K 线 (如果 check_interval=4，那么 bars_held = hours / 4)
                # 为了兼容性，这里我们直接用 patience_bars 换算回小时数进行比较
                
                threshold_hours_1 = patience_bars_1 * check_interval # 16h (4H) / 18h (1H)
                threshold_hours_2 = patience_bars_2 * check_interval # 32h (4H) / 36h (1H)
                threshold_hours_3 = patience_bars_3 * check_interval # 48h (4H) / 72h (1H)

                # 规则 A: 短期无力 (给了 N 根K线还没跑出利润)
                if holding_hours > threshold_hours_1 and current_r < 0.2:
                     exit_price = close
                     exit_reason = f"⏰ 动能不足 ({patience_bars_1}Bars)"

                # 规则 B: 中期僵尸 (给了 2N 根K线还在 0.5R 以下)
                elif holding_hours > threshold_hours_2 and current_r < 0.5:
                     exit_price = close
                     exit_reason = f"🧟 僵尸单 ({patience_bars_2}Bars)"

                # 规则 C: 长期超时 (强制换手)
                elif holding_hours > threshold_hours_3 and current_r < 1.0:
                     exit_price = close
                     exit_reason = f"⌛ 长期超时 ({patience_bars_3}Bars)"

                # ------------------------------------------------------
                # 4. 常规 止盈/止损/爆仓 检查 (执行离场)
                # ------------------------------------------------------
                if not exit_price:
                    if trade.direction == TradeDirection.LONG:
                        if low <= trade.liquidation_price: 
                            exit_price = trade.liquidation_price; exit_reason = "💥 爆仓"
                        elif low <= trade.stop_loss: 
                            exit_price = trade.stop_loss
                            if trade.is_breakeven and trade.stop_loss >= trade.entry_price:
                                exit_reason = "🛡️ 保本离场"
                            else:
                                exit_reason = "🔴 止损"
                        # 检查硬止盈 (>5R 才会考虑硬止盈，否则趋势优先)
                        elif high >= trade.take_profit: 
                             if (trade.take_profit - trade.entry_price) / risk_per_unit > 5.0:
                                 exit_price = trade.take_profit; exit_reason = "🎯 完美止盈 (>5R)"

                    else: # SHORT
                        if high >= trade.liquidation_price: 
                            exit_price = trade.liquidation_price; exit_reason = "💥 爆仓"
                        elif high >= trade.stop_loss: 
                            exit_price = trade.stop_loss
                            if trade.is_breakeven and trade.stop_loss <= trade.entry_price:
                                exit_reason = "🛡️ 保本离场"
                            else:
                                exit_reason = "🔴 止损"
                        elif low <= trade.take_profit:
                             if (trade.entry_price - trade.take_profit) / risk_per_unit > 5.0:
                                 exit_price = trade.take_profit; exit_reason = "🎯 完美止盈 (>5R)"

                # 执行平仓
                if exit_price:
                    self._execute_close(trade, exit_price, check_time, exit_reason)

            # --- E. 开新仓逻辑 ---
            if len(self.positions) < self.config.get('max_positions', 1):
                self._check_and_open_new_positions(check_time)

            # --- F. 记录净值曲线 ---
            self.equity_curve.append(self.total_capital) 
            if self.total_capital > peak_capital: peak_capital = self.total_capital
            if peak_capital > 0:
                dd = (peak_capital - self.total_capital) / peak_capital
                if dd > max_drawdown: max_drawdown = dd
        
        # ==================== 循环结束 ====================
        final_time = check_times[-1] if len(check_times) > 0 else datetime.now()
        
        for trade in self.positions:
            symbol = trade.symbol
            current_price = trade.entry_price 
            if symbol in self.data_cache and '1h' in self.data_cache[symbol]:
                df_1h = self.data_cache[symbol]['1h']
                mask = df_1h.index <= final_time
                if mask.sum() > 0: current_price = df_1h.iloc[mask.sum()-1]['close']
            
            if trade.direction == TradeDirection.LONG: trade.pnl = (current_price - trade.entry_price) * trade.position_size
            else: trade.pnl = (trade.entry_price - current_price) * trade.position_size
            
            trade.pnl -= trade.funding_fees
            margin_base = trade.position_value / trade.leverage if trade.leverage > 0 else 0
            if margin_base > 0: trade.pnl_percent = (trade.pnl / margin_base) * 100
            else: trade.pnl_percent = 0

            if self.verbose: ui_log(f"🔄 [接力] {trade.symbol} 持仓过夜 | 浮盈: ${trade.pnl:.2f} ({trade.pnl_percent:.2f}%)")

        stats = self.calculate_statistics(max_drawdown)
        stats['anomaly_report'] = self.anomaly_report
        stats['active_positions'] = self.positions
        floating_pnl_sum = sum(t.pnl for t in self.positions)
        stats['final_capital'] = self.total_capital + floating_pnl_sum
        stats['trades_history'] = getattr(self, 'history', []) 
        if not stats['trades_history']:
             stats['trades_history'] = getattr(self, 'closed_trades', [])

        return stats
    def calculate_statistics(self, max_drawdown: float) -> Dict[str, Any]:
        """计算统计指标 (修复版)"""
        # ✅ 优先使用 self.initial_capital，而不是去 config 里找默认值
        init_cap = self.initial_capital 
        
        if not self.closed_trades:
             return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': 0,
                'max_drawdown': max_drawdown * 100,
                'final_capital': self.total_capital,
                'sharpe': 0,
                'profit_factor': 0,
                'equity_curve': self.equity_curve,
                'trades': [],
                'initial_capital': init_cap,
                'annual_return': 0,
                'calmar': 0
            }
        
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for t in self.closed_trades if t.pnl > 0)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.closed_trades)
        total_return = (self.total_capital - init_cap) / init_cap * 100
        
        # 夏普比率
        sharpe = 0
        if len(self.equity_curve) > 1:
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            if returns.std() > 0:
                check_interval = self.config.get('check_interval_hours', 1)
                sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 24 / check_interval)
        
        # 盈利因子
        wins = [t.pnl for t in self.closed_trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in self.closed_trades if t.pnl <= 0]
        profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else 0
        
        # 计算年化收益率
        annual_return = 0
        try:
            start_str = self.config.get('start_date', '')
            end_str = self.config.get('end_date', '')
            if start_str and end_str:
                s_date = datetime.strptime(str(start_str)[:10], '%Y-%m-%d')
                e_date = datetime.strptime(str(end_str)[:10], '%Y-%m-%d')
                days = max((e_date - s_date).days, 1)
                years = days / 365.0
                if years > 0 and self.total_capital > 0:
                    annual_return = ((self.total_capital / init_cap) ** (1 / years) - 1) * 100
        except Exception:
            annual_return = 0
        
        calmar = annual_return / (max_drawdown * 100) if max_drawdown > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'max_drawdown': max_drawdown * 100,
            'final_capital': self.total_capital,
            'sharpe': sharpe,
            'profit_factor': profit_factor,
            'equity_curve': self.equity_curve,
            'trades': self.closed_trades,
            'initial_capital': init_cap,
            'annual_return': annual_return,
            'calmar': calmar
        }

# ==========================================
# 智能回测引擎 (负责数据调度与统一引擎调用)
# ==========================================

class SmartMoneyBacktestEngine:
    """聪明钱回测引擎 - 负责数据获取和调用统一回测逻辑"""
    
    def __init__(self, config: Dict, proxy_config: Dict = None, use_proxy: bool = True):
        self.config = config
        self.proxy_config = proxy_config or DEFAULT_PROXY
        self.use_proxy = use_proxy
        self.data_manager = DataManager()
        self.exchange = self._init_exchange()
        
        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f"backtest_results_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 数据缓存
        self.data_cache = {}
    
    def _init_exchange(self):
        """初始化交易所连接"""
        exchange_config = {
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
            'timeout': 30000
        }
        
        if self.use_proxy and self.proxy_config:
            exchange_config['proxies'] = self.proxy_config
        
        return ccxt.binance(exchange_config)
    
    def fetch_historical_data_with_cache(self, symbol: str, timeframe: str,
                                        start_date: str, end_date: str, 
                                        force_refresh: bool = False) -> pd.DataFrame:
        """获取历史数据（带缓存）"""
        # 检查缓存
        if not force_refresh:
            cached_data = self.data_manager.load_data(symbol, timeframe, start_date, end_date)
            if cached_data is not None:
                return cached_data
        
        st.info(f"正在下载 {symbol} {timeframe} 数据...")
        
        # 这里复用之前的 fetch_data_task 逻辑，但在单线程模式下运行
        # 为了简化，直接调用 ccxt
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        if end_dt > datetime.now():
            end_dt = datetime.now()
            end_date = end_dt.strftime('%Y-%m-%d')
        
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        
        all_ohlcv = []
        since = start_ts
        
        try:
            while since < end_ts:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                time.sleep(0.1)
                
                if len(all_ohlcv) > 200000: break # 防止死循环
                
        except Exception as e:
            st.error(f"下载数据出错: {e}")
            return pd.DataFrame()
            
        if not all_ohlcv:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).set_index('time').sort_index()
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        now = datetime.now()
        df = df[df['time'] <= now]
        
        # 保存缓存
        self.data_manager.save_data(symbol, timeframe, start_date, end_date, df)
        return df

    def load_all_timeframes(self, symbol: str, start_date: str, end_date: str, 
                           timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """加载所有时间周期的数据"""
        if timeframes is None:
            timeframes = self.config.get('timeframes', ['1h', '4h'])
        
        data_dict = {}
        
        for timeframe in timeframes:
            df = self.fetch_historical_data_with_cache(symbol, timeframe, start_date, end_date)
            if not df.empty:
                data_dict[timeframe] = df
        
        return data_dict

    def run(self, timeframes: List[str] = None):
        """运行回测主逻辑"""
        st.info("正在初始化回测引擎...")
        
        # 1. 确保数据已就绪
        # 如果 self.data_cache 是空的（没从 session_state 传进来），则尝试下载
        if not self.data_cache:
            for symbol in self.config['symbols']:
                data_dict = self.load_all_timeframes(
                    symbol, 
                    self.config['start_date'], 
                    self.config['end_date'],
                    timeframes or ['1h', '4h']
                )
                if data_dict:
                    self.data_cache[symbol] = data_dict
        
        if not self.data_cache:
            st.error("无可用数据，请检查网络或代理设置")
            return None, None

        # 2. 调用核心逻辑 (UnifiedBacktestEngine)
        # 这样确保了"优化"和"回测"用的是同一套数学逻辑
        core_engine = UnifiedBacktestEngine(self.config, self.data_cache)
        stats = core_engine.run_backtest()
        
        return stats, self.data_cache

class AdvancedParameterOptimizer:
    def __init__(self):
        # ==============================================================================
        # 🎮 中央控制室：在这里统一修改参数范围，下面代码会自动生效
        # 格式说明：
        #   数字范围: (最小值, 最大值)  -> 例如 (15, 35)
        #   开关选项: [True, False]     -> 例如 [False, True]
        # ==============================================================================
        self.bayesian_search_space = {
            
            # 【单笔风险】每笔交易允许亏损本金的百分之几？
            # 建议：(0.01, 0.03)。即 1%~3%。如果想固定2%，就写 (0.02, 0.02)
            'risk_per_trade': (0.01, 0.05),
            
            # 【杠杆倍数】
            # 建议：(1.0, 3.0)。现货写 (1.0, 1.0)。AI会自动测试高杠杆是否划算。
            'leverage_ratio': (1.0, 3.0),
            
            # 【最大持仓限制】同时最多拿几个币？
            # 建议：[1, 2]。资金小建议 1，资金大可以 2 或 3 分散风险。
            'max_positions': [1, 2],
            # ==============================================================================
            # 1. 趋势均线系统 (策略骨架 - 决定看多远的趋势)
            # ==============================================================================
            'ema_fast': (15, 35),       # 作用：最灵敏的均线。数值越小越快，但容易被震荡骗；数值越大越稳，但入场慢。
            'ema_medium': (40, 90),     # 作用：中期趋势确认。快线必须在此之上，防止短期逆势。
            'ema_slow': (100, 200),     # 作用：牛熊分界线。价格在此之上才考虑做多，过滤大部分熊市反弹。
            'ema_trend': (150, 300),    # 作用：4H级别指挥官。强制要求大周期共振，必须站稳此线才允许开单。

            # ==============================================================================
            # 2. 核心风控 (🔥 针对"尸检报告"的重点修复区)
            # ==============================================================================
            'stop_loss_atr': (2.0, 4.5), # 作用：止损宽度(防插针)。调大此值=止损变宽=不容易被洗=仓位自动变小。
            'min_adx': (25, 50),         # 作用：趋势强度门槛。调低(20)抓趋势刚启动；调高(40)只做主升浪(容易踏空)。
            'min_rr_ratio': (2.5, 6.0),  # 作用：盈亏比要求。预期赚不到 1.5倍 风险的钱，坚决不下注。

            # ==============================================================================
            # 3. 市场状态识别 (震荡 vs 趋势 - AI自动判断环境)
            # ==============================================================================
            'sideways_threshold': (15, 20), # 作用：震荡界定。ADX低于此值，视为垃圾时间，策略会变得极其挑剔。
            'sideways_rr': (2.0, 4.0),      # 作用：震荡市赔率。垃圾时间里，除非赔率极高(如3倍)，否则不开单。
            'trend_threshold': (25, 50),    # 作用：趋势界定。ADX高于此值，视为吃肉时间。
            'trend_rr': (1.5, 4.0),         # 作用：趋势市赔率。好行情里，赔率门槛稍微降低，先上车再说。

            # ==============================================================================
            # 4. 技术指标细节 (微调灵敏度)
            # ==============================================================================
            'rsi_period': (7, 25),      # 作用：RSI灵敏度。数值小(7)极灵敏适合超短线；数值大(25)平滑适合波段。
            'atr_period': (10, 30),     # 作用：波动率周期。计算止损距离的基础，数值越大对波动越不敏感。
            'bb_period': (15, 30),      # 作用：布林带周期。判断价格是否偏离均线过远。
            'bb_std': (1.5, 3.0),       # 作用：布林宽度。数值大(3.0)代表只有极端暴涨暴跌才触发回归逻辑。
            'adx_period': (10, 20),     # 作用：趋势反应速度。越小(10)对趋势变化越敏感，但也容易出假信号。
            'volume_ma': (10, 30),      # 作用：成交量均线。VPA放量确认的基准，用来识别机构进场。
            'min_signal_score': (60, 85), # 作用：入场及格分。调低(55)给"不完美但能赚钱"的单子机会；调高(80)只要极品。

            # ==============================================================================
            # 5. 高级逻辑开关 (布尔值/整数)
            # ==============================================================================
            'use_smc_logic': [False, True],     # 作用：SMC开关。True=叠加订单块逻辑(更严谨)，False=只用均线(更宽容)。
            'use_dynamic_risk': [False, True],  # 作用：动态风控。True=波动大时自动减仓保命(强烈推荐)。
            'fvg_lookback': (1, 5),             # 作用：缺口回溯。数值越大，寻找支撑压力的眼光越长远。
            'swing_lookback': (5, 15),          # 作用：波段结构。数值大(15)看大结构，数值小(5)看微观结构。
            'rs_period': (10, 30),              # 作用：相对强度。对比大盘(BTC)走势，只做比大盘强的币。
            
            # --- 6. 权重占位符 (勿动) ---
            'screening_weights': 'dirichlet' 
        }
    def calculate_smart_score_final(self, res):
        """
        评分函数 V5.0 (防误杀版)
        """
        def safe_get(key, default=0):
            val = res.get(key)
            return val if val is not None else default

        total_return = safe_get('total_return')
        max_dd = safe_get('max_drawdown')
        trades = safe_get('total_trades')
        win_rate = safe_get('win_rate')
        profit_factor = safe_get('profit_factor')
        
        # --- 1. 基础生存线 ---
        if trades == 0: return -100.0 
        if total_return <= 0: return -100.0 + total_return 

        # --- 2. 核心评分 ---
        score = 0.0
        score += (profit_factor - 1.0) * 20.0 
        dd_penalty = max(max_dd, 0.5) 
        calmar = total_return / dd_penalty
        score += calmar * 10.0

        # --- 3. 交易频率修正 ---
        if trades < 3:
            score *= 0.5 
        elif trades < 10:
            pass
        else:
            score += min(trades, 50) * 0.2

        # --- 4. 胜率修正 ---
        if win_rate < 30: score -= 10
        elif win_rate > 60: score += 5

        # --- 5. 盈亏质量 ---
        avg_pnl = total_return / trades
        if avg_pnl < 0.2: score -= 20 

        return score

    def select_best_params_ensemble(self, results: List[Dict], top_n: int = 5) -> Dict:
        """
        🔥 [增强版] 集成筛选逻辑 (保持不变)
        """
        if not results: return {}
        results.sort(key=lambda x: x.get('total_return', -999), reverse=True)
        top_results = results[:min(top_n, len(results))]
        if not top_results: return {}

        # print(f"🧩 [Ensemble] 正在集成前 {len(top_results)} 组最佳参数...")
        aggregated_params = {}
        param_keys = top_results[0]['params'].keys()
        from collections import Counter

        for key in param_keys:
            values = [r['params'][key] for r in top_results]
            
            if isinstance(values[0], (int, float)) and not isinstance(values[0], bool):
                avg_val = sum(values) / len(values)
                if isinstance(values[0], int):
                    aggregated_params[key] = int(round(avg_val))
                else:
                    aggregated_params[key] = round(avg_val, 4)
            elif isinstance(values[0], dict):
                try:
                    avg_dict = {}
                    sub_keys = values[0].keys()
                    for sub_k in sub_keys:
                        sub_vals = [d[sub_k] for d in values]
                        if isinstance(sub_vals[0], (int, float)):
                            avg_dict[sub_k] = float(sum(sub_vals) / len(sub_vals))
                        else:
                            avg_dict[sub_k] = sub_vals[0]
                    aggregated_params[key] = avg_dict
                except:
                    aggregated_params[key] = values[0]
            else:
                try:
                    # 投票
                    vote_vals = [str(v) for v in values] if isinstance(values[0], list) else values
                    vote_count = Counter(vote_vals)
                    most_common = vote_count.most_common(1)[0][0]
                    # 如果是bool字符串需要转回bool
                    if most_common == 'True': most_common = True
                    if most_common == 'False': most_common = False
                    aggregated_params[key] = most_common
                except:
                    aggregated_params[key] = values[0]

        # 逻辑修正
        if 'trend_threshold' in aggregated_params and 'sideways_threshold' in aggregated_params:
            if aggregated_params['trend_threshold'] > aggregated_params['sideways_threshold']:
                aggregated_params['trend_threshold'] = aggregated_params['sideways_threshold'] - 5
        return aggregated_params

  

    def bayesian_optimization(self, config: Dict, data_cache: Dict, 
                            n_trials: int = 30, timeout: int = 1800) -> List[Dict]:
        """
        全参数贝叶斯优化（集成 V3 评分与 Ensemble 筛选 - 联动配置版）
        """
        # ==========================================
        # 🕵️‍♂️ 侦探模式：调用来源追踪
        # ==========================================
        import traceback
        try:
            stack = traceback.extract_stack()
            caller = stack[-2]
            caller_name = caller.name
            line_no = caller.lineno
        except:
            caller_name = "Unknown"
            line_no = 0

        print(f"\n{'='*40}")
        print(f"🎯 贝叶斯优化启动 | 来源: {caller_name} (Line {line_no})")
        print(f"📅 优化区间: {config.get('start_date')} -> {config.get('end_date')}")
        print(f"🔄 试验次数: {n_trials} 次")
        
        if config.get('symbols'):
            sym = config['symbols'][0]
            if sym in data_cache:
                print(f"📊 数据就绪: {sym}")
        print(f"{'='*40}\n")
        # ==========================================

        st.info(f"🚀 开始全参数贝叶斯优化，共 {n_trials} 次试验...")
        
        # 提取搜索空间到局部变量，方便调用
        space = self.bayesian_search_space
        
        def objective(trial):
            # =======================================================
            # 🔥 关键修改：参数范围不再写死，而是从 self.bayesian_search_space 读取
            # 这样你只需要修改 __init__ 里的数字，这里就会自动变！
            # =======================================================
            params = {
                
                # --- 🔥 新增：资金管理参数联动 ---
                'risk_per_trade': trial.suggest_float('risk_per_trade', space['risk_per_trade'][0], space['risk_per_trade'][1], step=0.001),
                'leverage_ratio': trial.suggest_float('leverage_ratio', space['leverage_ratio'][0], space['leverage_ratio'][1], step=0.1),
                'max_positions': trial.suggest_categorical('max_positions', space['max_positions']),
                # --- 1. 趋势均线 ---
                'ema_fast': trial.suggest_int('ema_fast', space['ema_fast'][0], space['ema_fast'][1]),
                'ema_medium': trial.suggest_int('ema_medium', space['ema_medium'][0], space['ema_medium'][1]),
                'ema_slow': trial.suggest_int('ema_slow', space['ema_slow'][0], space['ema_slow'][1]),
                'ema_trend': trial.suggest_int('ema_trend', space['ema_trend'][0], space['ema_trend'][1], step=10),

                # --- 2. 风控参数 ---
                # 注意：float 类型通常带有 step (步长)，这里 step 保持硬编码以维持逻辑，但范围跟随配置
                'stop_loss_atr': trial.suggest_float('stop_loss_atr', space['stop_loss_atr'][0], space['stop_loss_atr'][1], step=0.1),
                'min_adx': trial.suggest_int('min_adx', space['min_adx'][0], space['min_adx'][1], step=1),
                'min_rr_ratio': trial.suggest_float('min_rr_ratio', space['min_rr_ratio'][0], space['min_rr_ratio'][1], step=0.1),

                # --- 3. 市场状态 ---
                'sideways_threshold': trial.suggest_int('sideways_threshold', space['sideways_threshold'][0], space['sideways_threshold'][1], step=1),
                'sideways_rr': trial.suggest_float('sideways_rr', space['sideways_rr'][0], space['sideways_rr'][1], step=0.1),
                'trend_threshold': trial.suggest_int('trend_threshold', space['trend_threshold'][0], space['trend_threshold'][1], step=5),
                'trend_rr': trial.suggest_float('trend_rr', space['trend_rr'][0], space['trend_rr'][1], step=0.1),

                # --- 4. 技术指标 ---
                'rsi_period': trial.suggest_int('rsi_period', space['rsi_period'][0], space['rsi_period'][1]),
                'atr_period': trial.suggest_int('atr_period', space['atr_period'][0], space['atr_period'][1]),
                'bb_period': trial.suggest_int('bb_period', space['bb_period'][0], space['bb_period'][1]),
                'bb_std': trial.suggest_float('bb_std', space['bb_std'][0], space['bb_std'][1], step=0.1),
                'adx_period': trial.suggest_int('adx_period', space['adx_period'][0], space['adx_period'][1]),
                'volume_ma': trial.suggest_int('volume_ma', space['volume_ma'][0], space['volume_ma'][1], step=5),
                'min_signal_score': trial.suggest_int('min_signal_score', space['min_signal_score'][0], space['min_signal_score'][1], step=5),

                # --- 5. 高级开关 ---
                'use_smc_logic': trial.suggest_categorical('use_smc_logic', space['use_smc_logic']),
                'use_dynamic_risk': trial.suggest_categorical('use_dynamic_risk', space['use_dynamic_risk']),
                'fvg_lookback': trial.suggest_int('fvg_lookback', space['fvg_lookback'][0], space['fvg_lookback'][1]),
                'swing_lookback': trial.suggest_int('swing_lookback', space['swing_lookback'][0], space['swing_lookback'][1]),
                'rs_period': trial.suggest_int('rs_period', space['rs_period'][0], space['rs_period'][1], step=5),
            }
            
            # 2. 逻辑一致性修正
            if params['trend_threshold'] > params['sideways_threshold']:
                params['trend_threshold'] = params['sideways_threshold'] - 5
            if params['trend_rr'] > params['sideways_rr']:
                params['trend_rr'] = params['sideways_rr'] - 0.5

            # 3. 权重处理 (AI 自动分配)
            logits = []
            weight_names = ['signal', 'trend', 'momentum', 'risk', 'vol']
            for name in weight_names:
                logits.append(trial.suggest_float(f'weight_logit_{name}', -2, 2))
            
            import numpy as np
            exp_logits = np.exp(logits)
            weights_array = exp_logits / np.sum(exp_logits)
            
            params['screening_weights'] = {
                'signal_score': float(weights_array[0]),
                'trend_strength': float(weights_array[1]),
                'momentum': float(weights_array[2]),
                'risk_reward': float(weights_array[3]),
                'volume_confirmation': float(weights_array[4])
            }
            # =======================================================
            # 🔥 关键修改：将 AI 找出的资金参数，强制注入到 config 中
            # =======================================================
            # 复制一份 config，以免修改原始配置
            trial_config = config.copy()
            
            # 覆盖资金管理参数
            trial_config['risk_per_trade'] = params['risk_per_trade']
            trial_config['leverage_ratio'] = params['leverage_ratio']
            trial_config['max_positions'] = params.get('max_positions', 1)
            # 4. 运行回测评估
            # 🔥🔥🔥 关键修复：优化必须强制重算指标，否则参数调整无效 🔥🔥🔥
            skip_calc = False
            
            result = self.evaluate_single_parameter_set(config, data_cache, params, skip_indicator_calc=skip_calc)
            
            if result:
                return self.calculate_smart_score_final(result)
            else:
                return -float('inf')
        
        # 创建Optuna研究
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner()
        )
        
        # 运行优化 (带进度条)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(n_trials):
            study.optimize(objective, n_trials=1)
            progress = (i + 1) / n_trials
            progress_bar.progress(progress)
            
            best_trial = study.best_trial
            try:
                current_val = study.trials[-1].value
                disp_val = max(current_val, -999) if current_val is not None else 0
                disp_best = max(best_trial.value, -999) if best_trial else 0
                status_text.text(f"优化进度: {i+1}/{n_trials} | 当前得分: {disp_val:.1f} | 最佳得分: {disp_best:.1f}")
            except:
                pass
        
        progress_bar.empty()
        status_text.empty()
        
        # 收集结果
        results = []
        for trial in study.trials:
            if trial.value is not None and trial.value > -1e8: 
                params = trial.params.copy()
                
                # 重建权重参数
                if 'weight_logit_signal' in params:
                    logits = [
                        params.get('weight_logit_signal', 0),
                        params.get('weight_logit_trend', 0),
                        params.get('weight_logit_momentum', 0),
                        params.get('weight_logit_risk', 0),
                        params.get('weight_logit_vol', 0)
                    ]
                    exp_logits = np.exp(logits)
                    weights_array = exp_logits / np.sum(exp_logits)
                    params['screening_weights'] = {
                        'signal_score': float(weights_array[0]),
                        'trend_strength': float(weights_array[1]),
                        'momentum': float(weights_array[2]),
                        'risk_reward': float(weights_array[3]),
                        'volume_confirmation': float(weights_array[4])
                    }
                    for key in list(params.keys()):
                        if key.startswith('weight_logit_'):
                            del params[key]

                # 重新回测获取完整数据
                result = self.evaluate_single_parameter_set(config, data_cache, params, skip_indicator_calc=False)
                if result:
                    results.append(result)
        
        if not results:
            print("❌ 警告：贝叶斯优化未找到任何有效参数组合！")
            return []

        # ==========================================
        # 🔥 [集成学习核心] 筛选最优参数
        # ==========================================
        
        # 1. 排序
        results.sort(key=lambda x: self.calculate_smart_score_final(x), reverse=True)
        
        # 2. 集成
        best_params_ensemble = self.select_best_params_ensemble(results, top_n=5)
        
        # 3. 构造结果
        ensemble_result = {
            'params': best_params_ensemble,
            'total_return': 0,      
            'max_drawdown': 0,      
            'sharpe_ratio': 0,      
            'is_ensemble': True,    
            'trades': []            
        }
        
        # 4. 置顶
        results.insert(0, ensemble_result)
        
        print("✅ [Bayesian] 集成优选完成，已将 Top5 平均参数置顶。")
        return results
    
    def optimize(self, config: Dict, data_cache: Dict, 
                method: str = 'grid', **kwargs) -> List[Dict]:
        """执行参数优化"""
        if method == 'grid':
            param_grid = kwargs.get('param_grid', self.default_param_grid)
            param_combinations = self.generate_param_combinations(param_grid)
            
            # 限制组合数量以避免过长时间运行
            if len(param_combinations) > 30:
                st.warning(f"参数组合过多({len(param_combinations)})，将随机选择30种进行优化")
                import random
                param_combinations = random.sample(param_combinations, 30)
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, params in enumerate(param_combinations):
                result = self.evaluate_single_parameter_set(config, data_cache, params)
                if result:
                    results.append(result)
                
                progress = (i + 1) / len(param_combinations)
                progress_bar.progress(progress)
                status_text.text(f"网格搜索进度: {i+1}/{len(param_combinations)}")
            
            progress_bar.empty()
            status_text.empty()
            
        elif method == 'bayesian':
            n_trials = kwargs.get('n_trials', 30)
            results = self.bayesian_optimization(config, data_cache, n_trials)
            
        elif method == 'genetic':
            population_size = kwargs.get('population_size', 30)
            generations = kwargs.get('generations', 15)
            results = self.genetic_algorithm_optimization(
                config, data_cache, population_size, generations)
        
        else:
            st.error(f"未知的优化方法: {method}")
            return []
        
        # 按总收益率排序
        if results:
            results.sort(key=lambda x: x.get('total_return', 0), reverse=True)
        
        return results
    
    def evaluate_single_parameter_set(self, config: Dict, data_cache: Dict, 
                                    params: Dict, skip_indicator_calc: bool = False) -> Optional[Dict]:
        """
        评估单个参数组合 (带 0 单自动诊断功能)
        """
        try:
            # 1. 合并配置参数
            test_config = config.copy()
            test_config.update(params)
            
            # 2. 第一次运行：静默模式 (追求速度)
            engine = UnifiedBacktestEngine(
                test_config, 
                data_cache, 
                verbose=False, 
                skip_indicator_calc=skip_indicator_calc
            )
            result = engine.run_backtest()
            
            # ======================================================
            # 🕵️‍♂️ 0单内窥镜：如果发现没开单，强制开启日志重跑一次
            # ======================================================
            if result and result.get('total_trades', 0) == 0:
                # 为了防止刷屏，我们只在第一次遇到 0 单时打印诊断信息
                if not hasattr(self, '_has_diagnosed_zero_trade'):
                    print(f"\n{'!'*40}")
                    print(f"⚠️ [诊断触发] 检测到 0 开单！正在以 Verbose=True 重跑一次以定位原因...")
                    print(f"🛠️ 当前调试参数: {params}")
                    print(f"{'!'*40}\n")
                    
                    # 强制开启日志
                    debug_engine = UnifiedBacktestEngine(
                        test_config, 
                        data_cache, 
                        verbose=True,  # <--- 开启啰嗦模式
                        skip_indicator_calc=skip_indicator_calc
                    )
                    debug_engine.run_backtest()
                    
                    print(f"\n{'!'*40}")
                    print(f"✅ [诊断结束] 请向上翻阅日志，查看 '❌' 或 '筛选失败' 的原因")
                    print(f"{'!'*40}\n")
                    
                    # 标记已诊断，避免后续 99 次都刷屏
                    self._has_diagnosed_zero_trade = True

            if result:
                result['params'] = params
                return result
            else:
                return None
        
        except Exception as e:
            # 捕获并打印详细报错
            import traceback
            print(f"❌ [优化器报错] 参数评估失败: {e}")
            print(traceback.format_exc()) # 打印完整堆栈，这很关键！
            return None
# ==========================================
# 2. 参数追踪器 (请放在 RollingWindowBacktester 类之前)
# ==========================================
import json
import hashlib
from datetime import datetime

class ParameterTracker:
    """参数使用审计追踪器"""
    def __init__(self):
        self.history = {}
        
    def track_usage(self, window_type, period_num, params, 
                    train_range, test_range, performance):
        """跟踪参数使用历史"""
        key = f"{window_type}_period{period_num}"
        self.history[key] = {
            'timestamp': datetime.now(),
            'params': params.copy(),
            'train_range': train_range,
            'test_range': test_range,
            'performance': performance,
            'param_hash': self._hash_params(params)
        }
    
    def _hash_params(self, params):
        """生成参数指纹"""
        try:
            # 过滤掉不可序列化的对象，只保留基本类型
            clean_params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
            # 排序确保一致性
            sorted_params = dict(sorted(clean_params.items()))
            param_str = json.dumps(sorted_params, sort_keys=True)
            return hashlib.md5(param_str.encode()).hexdigest()
        except Exception:
            return "hash_error"
    
    def compare_with_manual(self, manual_params):
        """对比滚动参数与手动参数"""
        if not manual_params:
            return []
            
        manual_hash = self._hash_params(manual_params)
        results = []
        
        for key, data in self.history.items():
            rolling_hash = data['param_hash']
            # 计算差异详情
            diff_details = []
            for k, v in manual_params.items():
                if k in data['params'] and data['params'][k] != v:
                    diff_details.append(f"{k}: 手动={v} vs 滚动={data['params'][k]}")
            
            is_same = (manual_hash == rolling_hash)
            
            results.append({
                'window': key,
                'is_same_as_manual': is_same,
                'diff_count': len(diff_details),
                'diff_details': "; ".join(diff_details[:5]) + ("..." if len(diff_details)>5 else ""), # 只显示前5个差异
                'train_range': data['train_range'],
                'test_range': data['test_range']
            })
        
        return results

# ==========================================
# 🔥 [新增类] 增强版亏损分析器 (插入在 RollingWindowBacktester 类定义之前)
# ==========================================
# ==========================================
# 🔥 [修复] 增强版亏损分析器 (完整代码)
# ==========================================
class AdvancedLossAnalyzer:
    """增强版亏损分析器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        # 动态阈值配置
        self.greedy_thresholds = {
            'low_volatility': 1.2,    # 低波动市场：1.2R
            'medium_volatility': 1.5,  # 中等波动：1.5R
            'high_volatility': 2.0     # 高波动：2.0R
        }
    
    def analyze_loss_reasons_enhanced(self, engine, symbol: str = None, 
                                      show_in_ui: bool = True) -> Dict:
        """
        🕵️‍♂️ 交易法医诊断报告 V4.0 (增强版)
        """
        from collections import Counter
        
        # 空结果结构
        empty_stats = {
            'total_loss': 0, 'good': 0, 'premature': 0, 'greedy': 0,
            'toxic': [], 'symbol': symbol or "ALL",
            'breakdown': {  # 新增细分统计
                'greedy_by_volatility': {'low': 0, 'medium': 0, 'high': 0},
                'premature_by_duration': {'very_short': 0, 'short': 0, 'normal': 0},
                'good_by_market_regime': {'trend': 0, 'ranging': 0, 'reversal': 0}
            }
        }

        if not engine.closed_trades:
            return empty_stats
        
        # 筛选亏损单
        loss_trades = self._filter_loss_trades(engine.closed_trades, symbol)
        if not loss_trades:
            return empty_stats
        
        # 初始化计数器
        premature_stop_count = 0
        good_stop_count = 0
        missed_profit_count = 0
        toxic_reasons = []
        
        # 获取数据缓存
        full_df = self._get_dataframe(engine.data_cache, symbol)
        
        # 逐单深度分析
        for trade in loss_trades:
            # A. 收集毒药指标
            self._collect_toxic_signals(trade, toxic_reasons)
            
            # B. 贪心检测（改进版）
            greedy_result = self._check_greediness_advanced(trade, full_df)
            if greedy_result['is_greedy']:
                missed_profit_count += 1
                # 记录细分：按波动率
                vol_level = self._get_volatility_level(greedy_result['atr_pct'])
                empty_stats['breakdown']['greedy_by_volatility'][vol_level] += 1
                continue
            
            # C. 止损质量检测（改进版）
            if trade.exit_reason == "止损" and full_df is not None:
                stop_quality = self._evaluate_stop_quality_advanced(trade, full_df)
                
                if stop_quality['is_premature']:
                    premature_stop_count += 1
                    # 记录细分：按持仓时间
                    empty_stats['breakdown']['premature_by_duration'][stop_quality['duration_category']] += 1
                else:
                    good_stop_count += 1
                    # 记录细分：按市场状态
                    empty_stats['breakdown']['good_by_market_regime'][stop_quality['market_regime']] += 1
            else:
                good_stop_count += 1
        
        # 统计结果
        total_loss = len(loss_trades)
        stats = {
            'symbol': symbol or "所有币种",
            'total_loss': total_loss,
            'good': good_stop_count,
            'premature': premature_stop_count,
            'greedy': missed_profit_count,
            'toxic': toxic_reasons,
            'breakdown': empty_stats['breakdown']
        }
        
        # UI显示 (如果需要)
        if show_in_ui:
            # 这里通常不需要 print，因为 visualize_window_diagnosis 会处理显示
            pass
        
        return stats
    
    def _check_greediness_advanced(self, trade, full_df: pd.DataFrame) -> Dict:
        """
        改进版贪心检测
        """
        # 兼容 Trade 对象属性
        mfe = getattr(trade, 'mfe', trade.entry_price)
        if mfe == 0: mfe = trade.entry_price
        
        entry_price = trade.entry_price
        stop_loss = trade.stop_loss
        direction = trade.direction
        
        # 基础计算
        risk = abs(entry_price - stop_loss)
        if risk == 0:
            return {'is_greedy': False, 'max_r': 0, 'reason': '无风险定义', 'atr_pct': 0}
        
        # 计算最大浮盈（R单位）
        # 兼容枚举和字符串
        is_long = direction == TradeDirection.LONG or str(direction) == 'LONG'
        
        if is_long:
            max_profit = mfe - entry_price
        else:
            max_profit = entry_price - mfe
        
        max_r = max_profit / risk
        
        # 1. 获取持仓时间
        hold_hours = getattr(trade, 'duration_hours', 0)
        time_adjusted_threshold = self._get_time_adjusted_threshold(hold_hours)
        
        # 2. 获取 ATR 波动率
        atr_pct = self._get_atr_percentage(trade, full_df) if full_df is not None else 0.02
        volatility_adjusted_threshold = self._get_volatility_adjusted_threshold(atr_pct)
        
        # 3. 综合阈值
        final_threshold = max(time_adjusted_threshold, volatility_adjusted_threshold)
        
        is_greedy = False
        if max_r > final_threshold:
            is_greedy = True
        
        return {
            'is_greedy': is_greedy,
            'max_r': max_r,
            'atr_pct': atr_pct
        }
    
    def _evaluate_stop_quality_advanced(self, trade, full_df: pd.DataFrame) -> Dict:
        """
        改进版止损质量评估
        """
        # 1. 找到止损时间点
        exit_time = trade.exit_time
        if exit_time not in full_df.index:
            try:
                exit_idx = full_df.index.searchsorted(exit_time)
                if exit_idx >= len(full_df):
                    return {'is_premature': False, 'duration_category': 'normal', 'market_regime': 'ranging'}
            except:
                return {'is_premature': False, 'duration_category': 'normal', 'market_regime': 'ranging'}
        else:
            exit_idx = full_df.index.get_loc(exit_time)
            # 如果索引重复，取第一个
            if isinstance(exit_idx, slice):
                exit_idx = exit_idx.start
            elif isinstance(exit_idx, np.ndarray):
                exit_idx = exit_idx[0]
        
        # 2. 智能回溯窗口
        atr_value = self._get_atr_at_time(full_df, exit_idx)
        entry_price = trade.entry_price
        
        look_ahead_bars = 48 # 观察未来48小时
        if exit_idx + look_ahead_bars >= len(full_df):
            look_ahead_bars = len(full_df) - exit_idx - 1
        
        if look_ahead_bars <= 0:
            return {'is_premature': False, 'duration_category': 'normal', 'market_regime': 'ranging'}
        
        # 3. 获取后续数据
        post_data = full_df.iloc[exit_idx+1 : exit_idx+look_ahead_bars+1]
        
        # 4. 判断逻辑
        is_premature = False
        is_long = trade.direction == TradeDirection.LONG or str(trade.direction) == 'LONG'
        
        if is_long:
            # 如果做多止损后，价格又涨回了入场价上方
            if post_data['high'].max() > entry_price + (atr_value * 0.5):
                is_premature = True
        else:
            # 如果做空止损后，价格又跌回了入场价下方
            if post_data['low'].min() < entry_price - (atr_value * 0.5):
                is_premature = True
        
        # 5. 辅助信息
        market_regime = self._detect_market_regime_at_time(full_df, exit_idx)
        
        hold_hours = getattr(trade, 'duration_hours', 0)
        if hold_hours < 4: duration_category = 'very_short'
        elif hold_hours < 24: duration_category = 'short'
        else: duration_category = 'normal'
        
        return {
            'is_premature': is_premature,
            'market_regime': market_regime,
            'duration_category': duration_category
        }

    # --- 辅助方法 ---
    def _filter_loss_trades(self, trades, symbol):
        """筛选亏损单"""
        # 兼容 Trade 对象和字典
        loss_trades = []
        for t in trades:
            pnl = getattr(t, 'pnl', 0)
            t_symbol = getattr(t, 'symbol', '')
            if pnl < 0:
                if symbol is None or (symbol and symbol in t_symbol):
                    loss_trades.append(t)
        return loss_trades
        
    def _get_dataframe(self, data_cache, symbol):
        """获取数据DataFrame"""
        if not data_cache: return None
        # 尝试获取
        target_sym = symbol
        if not target_sym:
            # 如果没有指定symbol，随便拿一个（通常是第一个）来做环境判断参考，或者返回None
            if list(data_cache.keys()):
                target_sym = list(data_cache.keys())[0]
            else:
                return None
        
        if target_sym in data_cache:
            if '1h' in data_cache[target_sym]: return data_cache[target_sym]['1h']
            if '4h' in data_cache[target_sym]: return data_cache[target_sym]['4h']
        return None
        
    def _collect_toxic_signals(self, trade, toxic_list):
        """收集致死信号"""
        # 兼容 list 和 str
        reasons = getattr(trade, 'signal_reasons', []) or getattr(trade, 'entry_reasons', [])
        if reasons:
            if isinstance(reasons, list):
                toxic_list.extend(reasons)
            elif isinstance(reasons, str):
                toxic_list.append(reasons)
            
    def _get_time_adjusted_threshold(self, hours):
        """时间动态阈值"""
        if hours < 6: return 1.2
        elif hours < 24: return 1.5
        return 2.0
        
    def _get_atr_percentage(self, trade, df):
        """获取 ATR 百分比"""
        # 尝试根据 entry_time 获取当时的 ATR
        entry_time = getattr(trade, 'entry_time', None)
        if entry_time and entry_time in df.index:
            row = df.loc[entry_time]
            if 'atr' in row and 'close' in row and row['close'] > 0:
                return row['atr'] / row['close']
        return 0.02 # 默认值
        
    def _get_volatility_level(self, atr_pct):
        """波动率分级"""
        if atr_pct > 0.03: return 'high'
        elif atr_pct > 0.015: return 'medium'
        return 'low'

    def _get_volatility_adjusted_threshold(self, atr_pct):
        """波动率动态阈值"""
        if atr_pct > 0.03: return 2.5 # 高波动
        elif atr_pct > 0.015: return 1.8 # 中波动
        return 1.3 # 低波动
        
    def _get_atr_at_time(self, df, idx):
        """获取特定时间的ATR"""
        if isinstance(idx, int) and idx < len(df):
            if 'atr' in df.columns:
                return df['atr'].iloc[idx]
            return df['close'].iloc[idx] * 0.02 # 兜底
        return 0
        
    def _detect_market_regime_at_time(self, df, idx):
        """判断市场状态"""
        if isinstance(idx, int) and idx < len(df):
            if 'adx' in df.columns:
                return 'trend' if df['adx'].iloc[idx] > 25 else 'ranging'
        return 'ranging'
# ==========================================
# 🔥 【新增】 6+1 滚动窗口回测引擎 (完整版)
# ==========================================

class RollingWindowBacktester:
    """
    6+1窗口滚动回测器 (Walk-Forward Analysis) - 终极调试增强版
    """
    def create_profit_analysis_tab(self, trades: List[Any], period_num: int):
        """
        🔥 [增强版] 盈利分析Tab：详细分析盈利单的离场方式
        兼容 Dictionary 和 Trade Object 两种数据格式
        """
        # --- 0. 安全访问辅助函数 ---
        def get_val(item, attr_name, dict_key, default=None):
            if isinstance(item, dict):
                return item.get(dict_key, default)
            return getattr(item, attr_name, default)

        def set_val(item, attr_name, dict_key, value):
            if isinstance(item, dict):
                item[dict_key] = value
            else:
                setattr(item, attr_name, value)

        # --- 1. 筛选盈利单 ---
        # 兼容 t.pnl 和 t['pnl']
        win_trades = [t for t in trades if get_val(t, 'pnl', 'pnl', 0) > 0]
        
        if not win_trades:
            st.info("🐻 本周期没有盈利单，无法分析利润结构。")
            return
        
        # --- 2. 收集/推断退出原因 ---
        exit_reasons = []
        for trade in win_trades:
            # 获取关键字段
            exit_reason = get_val(trade, 'exit_reason', 'exit_reason')
            exit_price = get_val(trade, 'exit_price', 'exit_price')
            entry_price = get_val(trade, 'entry_price', 'entry_price')
            take_profit = get_val(trade, 'take_profit', 'take_profit')
            direction = get_val(trade, 'direction', 'direction') # 可能是枚举或字符串

            # 智能推断逻辑
            if not exit_reason:
                if exit_price is not None and entry_price is not None:
                    # 处理枚举转字符串的情况
                    is_long = direction == 'long' or direction == 1 or str(direction) == 'TradeDirection.LONG'
                    
                    if is_long:
                        if take_profit and exit_price >= take_profit:
                            exit_reason = "🎯 主动止盈(固定TP)"
                        elif exit_price > entry_price:
                            exit_reason = "🛡️ 被动止盈(移动止损)"
                        else:
                            exit_reason = "🔴 亏损止损" # 理论上不该进这里，因为我们筛选了 wins
                    else: # Short
                        if take_profit and exit_price <= take_profit:
                            exit_reason = "🎯 主动止盈(固定TP)"
                        elif exit_price < entry_price:
                            exit_reason = "🛡️ 被动止盈(移动止损)"
                        else:
                            exit_reason = "🔴 亏损止损"
                else:
                    exit_reason = "❓ 未知原因"
                
                # 回写推断结果 (为了后续统计)
                set_val(trade, 'exit_reason', 'exit_reason', exit_reason)
            
            exit_reasons.append(exit_reason)
        
        # --- 3. 创建DataFrame ---
        df_wins = pd.DataFrame({
            'exit_reason': exit_reasons,
            'pnl': [get_val(t, 'pnl', 'pnl') for t in win_trades],
            'pnl_percent': [get_val(t, 'return_pct', 'return_pct', 0)*100 for t in win_trades], # 注意单位转换
            'symbol': [get_val(t, 'symbol', 'symbol') for t in win_trades]
        })
        
        # --- 4. 可视化绘图 ---
        reason_counts = df_wins['exit_reason'].value_counts()
        
        color_map = {
            "🎯 主动止盈(固定TP)": "#FF6B6B", 
            "📈 移动止盈(追踪止损)": "#4ECDC4", 
            "🛡️ 被动止盈(移动止损)": "#45B7D1", 
            "🛡️ 被动止盈(保本止损)": "#96CEB4",
            "⏰ 时间止损(持仓超时)": "#FFEAA7", 
            "🔴 亏损止损": "#D7263D",
            "❓ 未知原因": "#95a5a6"
        }

        # 图1：饼图
        fig_pie = px.pie(
            values=reason_counts.values,
            names=reason_counts.index,
            title="📊 盈利是靠什么落袋的？",
            color=reason_counts.index,
            color_discrete_map=color_map,
            hole=0.4
        )
        
        # 图2：条形图 (含金量分析)
        avg_pnl_by_reason = df_wins.groupby('exit_reason')['pnl_percent'].mean().sort_values()
        
        fig_bar = px.bar(
            x=avg_pnl_by_reason.values,
            y=avg_pnl_by_reason.index,
            orientation='h',
            title="💰 哪种离场方式赚得更多？(平均收益率%)",
            labels={'x': '平均收益率(%)', 'y': ''},
            color=avg_pnl_by_reason.index,
            color_discrete_map=color_map,
            text_auto='.2f'
        )
        fig_bar.update_layout(showlegend=False)

        # 布局显示
        c1, c2 = st.columns([1, 1.2]) # 右边宽一点给条形图
        with c1:
            st.plotly_chart(fig_pie, use_container_width=True, key=f"win_pie_{period_num}")
        with c2:
            st.plotly_chart(fig_bar, use_container_width=True, key=f"win_bar_{period_num}")

        # --- 5. 智能诊断 ---
        st.markdown("#### 🧠 盈利结构诊断")
        
        total_wins = len(win_trades)
        hard_tp_count = reason_counts.get("🎯 主动止盈(固定TP)", 0)
        trailing_count = reason_counts.get("🛡️ 被动止盈(移动止损)", 0) + reason_counts.get("📈 移动止盈(追踪止损)", 0)
        
        insights = []
        if total_wins > 0:
            tp_ratio = hard_tp_count / total_wins
            
            # 诊断 1: 截断利润风险
            if tp_ratio > 0.6:
                insights.append(f"⚠️ **严重截断利润**：{tp_ratio:.1%} 的单子都是止盈出局。建议**移除/调大固定止盈**，让利润奔跑！")
            elif tp_ratio < 0.2:
                insights.append(f"✅ **奔跑吧利润**：大部分订单没有被固定止盈限制住，这是大牛市策略的特征。")
                
            # 诊断 2: 移动止损效率
            if trailing_count > 0:
                avg_trail_pnl = df_wins[df_wins['exit_reason'].str.contains("移动|追踪", na=False)]['pnl_percent'].mean()
                avg_fix_pnl = df_wins[df_wins['exit_reason'].str.contains("固定", na=False)]['pnl_percent'].mean()
                
                if pd.notna(avg_fix_pnl) and avg_fix_pnl > 0:
                    ratio = avg_trail_pnl / avg_fix_pnl
                    if ratio > 1.2:
                        insights.append(f"💎 **移动止损真香**：移动止损单的平均利润是固定止盈单的 **{ratio:.1f}倍**。坚持用它！")
                    elif ratio < 0.8:
                        insights.append(f"🔧 **移动止损太紧**：移动止损虽然保住了命，但平均利润不如直接止盈。建议**放宽回调阈值**。")

        for i in insights: st.info(i)
    def __init__(self, config: Dict[str, Any], data_cache: Dict[str, Any]):
        self.config = config
        self.data_cache = data_cache
        self.optimizer = AdvancedParameterOptimizer()
        self.tracker = ParameterTracker()
        # 🔥 新增：初始化结果存储
        self.results = []  # 存储所有窗口结果
        self.cumulative_equity = []  # 存储资金曲线
        
        # 🔥 [新增] 初始化增强版分析器
        self.loss_analyzer = AdvancedLossAnalyzer(self.config)
    def run_6plus1_validation(self, 
                             start_date: str = "2023-01-01",
                             end_date: str = "2024-01-01",
                             train_months: int = 5,
                             test_months: int = 1,
                             roll_step_months: int = 1,
                             n_optimization_trials: int = 50,
                             debug_fixed_params: Dict = None):
        """
        全真模拟滚动回测 (Walk-Forward Analysis) - 跨月持仓接力版
        特性：
        1. 动态缓冲计算指标
        2. 支持跨月持仓接力 (不强平)
        """
        import pandas as pd
        from datetime import datetime, timedelta
        
        # ================= [DEBUG START] =================
        ui_log("🎯 [滚动回测] 方法开始执行 - 跨月持仓接力模式")
        ui_log(f"   范围: {start_date} -> {end_date}")
        ui_log(f"   训练月数: {train_months}, 测试月数: {test_months}")
        ui_log(f"   初始配置: {self.config.get('initial_capital')}U")
        # ================= [DEBUG END] =================

        # ==============================================================================
        # 1. 环境准备与UI (保持原样)
        # ==============================================================================
        st.subheader("🧐 模拟环境核对")
        if debug_fixed_params:
            st.warning("🔒 调试模式已开启：跳过贝叶斯优化，强制使用手动配置参数！结果应与手动回测高度一致。")

        curr_cfg = self.config
        is_fixed = curr_cfg.get('position_mode') == 'fixed'
        mode_label = "固定仓位" if is_fixed else "复合增长"
        mode_icon = "💰" if is_fixed else "🚀"
        pos_main = f"${curr_cfg.get('target_position_value', 0):,.0f}" if is_fixed else f"比例 {curr_cfg.get('compounding_ratio', 0):.1f}"
        pos_sub = "单仓价值" if is_fixed else "复利 (1.0=全仓)"

        try:
            s_dt = datetime.strptime(start_date, '%Y-%m-%d')
            e_dt = datetime.strptime(end_date, '%Y-%m-%d')
            total_span_days = (e_dt - s_dt).days
            span_display = f"{total_span_days} 天"
        except:
            s_dt, e_dt = None, None
            span_display = "N/A"

        st.caption("💰 **资金设定**")
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1: st.metric("资金模式", f"{mode_icon} {mode_label}", pos_sub)
        with m_c2: st.metric("初始本金", f"${curr_cfg.get('initial_capital'):,.0f}", f"杠杆: {curr_cfg.get('leverage')}x")
        with m_c3: st.metric("仓位规模", pos_main)
        with m_c4: 
            if debug_fixed_params:
                st.metric("优化模式", "⛔ 已禁用", "使用固定参数")
            else:
                st.metric("单月优化", f"{n_optimization_trials} 次", "贝叶斯尝试")

        st.markdown("---")
        
        # ==============================================================================
        # 2. 循环初始化
        # ==============================================================================
        results = []
        
        # 关键变量定义
        initial_cap_setting = self.config.get('initial_capital', 10000)
        running_capital = initial_cap_setting 
        cumulative_equity = [running_capital]
        
        # 🔥 [关键新增] 跨月接力棒：存储上个月留下的活跃持仓
        carried_over_positions = []
        
        current_date = s_dt
        final_date = e_dt
        period_num = 1
        
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        if not s_dt or not e_dt or total_span_days <= 0: return [], []

        # ==============================================================================
        # 3. 🔄 滚动循环
        # ==============================================================================
        while True:
            ui_log(f"\n🔄 第 {period_num} 轮滚动窗口开始 | 当前日期: {current_date.date()} | 本金接力: ${running_capital:.2f}")

            # --- A. 时间窗口计算 ---
            train_start = current_date
            train_end_raw = train_start + pd.DateOffset(months=train_months)
            test_start = train_end_raw
            test_end = test_start + pd.DateOffset(months=test_months)
            
            if test_start >= final_date: 
                ui_log("🛑 测试窗口超出结束日期，停止模拟")
                break
            if test_end > final_date: test_end = final_date
            
            # 进度条
            days_passed = (test_end - s_dt).days
            progress = min(days_passed / total_span_days, 1.0)
            progress_bar.progress(progress)
            
            # 破产检查
            if running_capital <= 100:
                ui_log("!" * 60)
                ui_log(f"💀 [严重警报] 账户已破产！模拟终止！")
                st.error("❌ 账户已破产，模拟终止！详情请看上方日志。")
                break
            
            # 格式化日期字符串
            train_start_str = train_start.strftime('%Y-%m-%d')
            human_train_end = (train_end_raw - timedelta(days=1)).strftime('%Y-%m-%d')
            test_start_str = test_start.strftime('%Y-%m-%d')
            test_end_str = test_end.strftime('%Y-%m-%d')

            status_container.markdown(f"""
            ### 🔄 第 {period_num} 轮滚动
            - **🧠 训练**: `{train_start_str}` ~ `{human_train_end}`
            - **⚔️ 实战**: `{test_start_str}` ~ `{test_end_str}`
            - **💰 本金**: `${running_capital:,.2f}`
            - **🤝 接力**: `{len(carried_over_positions)}` 单
            """)

            # ==========================================
            # 🔥 [核心修复] 动态缓冲切片函数
            # 每一轮都必须把数据往前多切 90 天，用于现场计算指标
            # ==========================================
            buffer_days = 90
            
            def get_buffered_slice(s_date_str, e_date_str):
                """获取带缓冲区的原始数据切片，用于现场计算指标"""
                slice_s_dt = pd.to_datetime(s_date_str) - timedelta(days=buffer_days)
                # 结束时间包含当天最后一秒
                slice_e_dt = pd.to_datetime(e_date_str) + timedelta(hours=23, minutes=59, seconds=59)
                
                cache_slice = {}
                has_data = False
                for sym, tfs in self.data_cache.items():
                    cache_slice[sym] = {}
                    for tf, df in tfs.items():
                        if df.empty: continue
                        # 物理切片：保留原始 OHLCV
                        mask = (df.index >= slice_s_dt) & (df.index <= slice_e_dt)
                        df_sub = df.loc[mask].copy()
                        if not df_sub.empty:
                            cache_slice[sym][tf] = df_sub
                            has_data = True
                return cache_slice, has_data

            # ==============================================================================
            # --- B. 备战阶段：优化 ---
            # ==============================================================================
            best_params = {}
            top_10_results = []

            if debug_fixed_params:
                # 调试模式
                best_params = debug_fixed_params.copy()
                self.tracker.track_usage(
                    window_type="debug_fixed",
                    period_num=period_num,
                    params=best_params,
                    train_range="SKIPPED",
                    test_range=f"{test_start_str}~{test_end_str}",
                    performance=0
                )
                import time
                time.sleep(0.05)
                
            else:
                # 正常模式：运行贝叶斯优化
                # ⚠️ 注意：训练阶段通常只关心参数对新行情的影响，暂不考虑带单入场
                # 如果非要带单训练会极其复杂，这里坚持“新环境选新参数”的原则
                
                # 1. 获取带缓冲的训练数据
                train_data_buffered, has_train = get_buffered_slice(train_start_str, human_train_end)
                
                if has_train:
                    # 2. 配置参数
                    current_target_val = self.config.get('target_position_value', 30000)
                    safe_running_cap = running_capital if running_capital > 0 else 10000 
                    current_leverage_ratio = current_target_val / safe_running_cap
                    
                    train_fixed_capital = 10000
                    scaled_target_position = train_fixed_capital * current_leverage_ratio
                    
                    train_config = self.config.copy()
                    train_config.update({
                        'start_date': train_start_str,   # 引擎逻辑开始时间
                        'end_date': human_train_end,
                        'initial_capital': train_fixed_capital,
                        'target_position_value': scaled_target_position
                    })

                    # 3. 运行优化
                    optimization_results = self.optimizer.bayesian_optimization(
                        train_config, train_data_buffered, n_trials=n_optimization_trials
                    )
                    
                    if optimization_results:
                        top_10_results = optimization_results[:10]
                        best_params = top_10_results[0]['params']
                        
                        self.tracker.track_usage(
                            window_type="rolling_opt",
                            period_num=period_num,
                            params=best_params,
                            train_range=f"{train_start_str}~{human_train_end}",
                            test_range=f"{test_start_str}~{test_end_str}",
                            performance=top_10_results[0].get('total_return', 0)
                        )
                    else:
                        ui_log(f"⚠️ [Window {period_num}] 优化未返回结果")
                else:
                    ui_log(f"⚠️ [Window {period_num}] 训练窗口数据不足")

            # ==============================================================================
            # --- C. 实战阶段：真实交易 (跨月接力核心) ---
            # ==============================================================================
            
            # 1. 获取带缓冲的测试数据
            test_data_buffered, has_test = get_buffered_slice(test_start_str, test_end_str)
            
            test_config = self.config.copy()
            if best_params:
                test_config.update(best_params) 
            
            # 必须接力资金
            test_config.update({
                'start_date': test_start_str, # 引擎会从这一天开始交易，自动跳过前面的缓冲
                'end_date': test_end_str,
                'initial_capital': running_capital
            })
            
            # 2. 运行回测引擎
            # 🔥 关键修改：
            # (1) skip_indicator_calc=False (现场算指标)
            # (2) inherited_positions=carried_over_positions (带老单入场)
            engine = UnifiedBacktestEngine(
                test_config, 
                test_data_buffered, 
                verbose=True,
                skip_indicator_calc=False,
                inherited_positions=carried_over_positions # <--- 传入接力棒
            )
            test_stats = engine.run_backtest()

            # =========================================================
            # 🔥 [修复] 正确的诊断逻辑顺序：先获取变量，再打印，再循环
            # =========================================================
            
            # 1. 先定义 all_symbols
            all_symbols = list(engine.data_cache.keys())
            
            # 2. UI 显示日志和标题
            print(f"\n{'='*20} 🏥 启动多币种法医诊断 ({len(all_symbols)}个) {'='*20}")
            st.markdown("### 🏥 本轮交易法医诊断 (亏损分析)")
            
            # 3. 容器用于存储本轮所有币种的诊断结果
            current_window_diagnosis = {}
            
            # 4. 执行循环诊断 (收集数据)
            for symbol in all_symbols:
                diag_stats = self.analyze_loss_reasons(engine, symbol, show_in_ui=False) 
                if diag_stats and diag_stats.get('total_loss', 0) > 0:
                    current_window_diagnosis[symbol] = diag_stats
            
            # 5. 调用汇总仪表盘
            self.visualize_window_diagnosis(engine, period_num)
            
            # =========================================================

            # 侦探埋点
            if 'diff_detective' in st.session_state:
                st.session_state.diff_detective.capture_rolling(
                    period_num=period_num,
                    config=test_config,
                    data_cache=self.data_cache, 
                    stats=test_stats
                )
            
            # ==============================================================================
            # --- D. 结算与更新 (接力逻辑闭环) ---
            # ==============================================================================
            
            # 1. 提取本轮结束后的“幸存者”
            carried_over_positions = test_stats.get('active_positions', [])
            
            # 2. 获取最终净值 (已在引擎中修正，包含了 active_positions 的浮盈)
            # 优先使用引擎计算好的 final_capital，如果异常则回退
            final_cap = test_stats.get('final_capital', running_capital)
            
            # 异常纠错
            if final_cap <= 100 and running_capital > 1000:
                has_liquidation = any(t.exit_reason and "爆仓" in t.exit_reason for t in engine.closed_trades)
                # 只有真的爆仓才认亏，否则可能是计算错误，回滚本金
                if not has_liquidation:
                    final_cap = running_capital
                    ui_log("⚠️ [异常] 资金归零但未检测到爆仓，回滚本金")

            # 计算本轮盈亏 (净值增长)
            profit = final_cap - running_capital
            return_pct = (profit / running_capital * 100) if running_capital > 0 else 0
            
            # 记录结果
            window_result = {
                'period_num': period_num,
                'train_range': f"{train_start_str}~{human_train_end}",
                'test_range': f"{test_start_str}~{test_end_str}",
                'start_balance': running_capital,
                'end_balance': final_cap,
                'profit': profit,
                'return_pct': return_pct,
                'best_params': best_params,
                'optimization_top_list': top_10_results,
                'detailed_trades': test_stats.get('trades', []) if test_stats else [],
                'diagnosis_report': current_window_diagnosis
            }
            results.append(window_result)

            # 更新本金，准备下一轮
            running_capital = final_cap
            cumulative_equity.append(running_capital)
            
            ui_log(f"💰 窗口{period_num}结算: {return_pct:+.2f}% ({len(engine.closed_trades)}单已结) | 资金 ${final_cap:,.0f}")
            if carried_over_positions:
                ui_log(f"   -> 🤝 接力 {len(carried_over_positions)} 个持仓到下一轮")
            
            # 推进时间
            current_date = current_date + pd.DateOffset(months=roll_step_months)
            period_num += 1
            
            # 边界检查
            if period_num > 100: 
                ui_log("⚠️ 达到最大窗口限制")
                break
                
            next_train_end = current_date + pd.DateOffset(months=train_months)
            next_test_end = next_train_end + pd.DateOffset(months=test_months)
            
            if next_test_end > final_date + timedelta(days=20): 
                ui_log(f"🛑 模拟结束 (下一轮超出数据范围)")
                break
            
        progress_bar.progress(1.0)
        status_container.success(f"✅ 完成！最终资金: ${cumulative_equity[-1]:,.2f}")
        
        if 'rolling_tracker' not in st.session_state:
            st.session_state.rolling_tracker = self.tracker
        else:
            st.session_state.rolling_tracker = self.tracker
            
        return results, cumulative_equity



    def analyze_rolling_results(self, results: List[Dict], cumulative_equity: List[float]):
        """
        📊 分析滚动回测结果并绘图 (修复版：修复属性不存在错误)
        """
        import plotly.graph_objects as go
        import numpy as np
        
        # 🔥 关键修复：保存结果到实例属性
        self.results = results
        self.cumulative_equity = cumulative_equity
        
        if not results:
            st.warning("⚠️ 没有产生有效交易数据，无法分析。")
            return
        
        # --- 内部辅助函数：安全获取属性 ---
        def get_val(item, key, default=0):
            if isinstance(item, dict):
                return item.get(key, default)
            return getattr(item, key, default)
        
        # ==========================================
        # 1. 核心资金指标统计 (保持原样)
        # ==========================================
        total_profit = cumulative_equity[-1] - cumulative_equity[0]
        total_return = (total_profit / cumulative_equity[0]) * 100
        
        equity_series = pd.Series(cumulative_equity)
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        
        # 计算胜率
        total_trades = sum([len(r.get('detailed_trades', [])) for r in results])
        total_wins = sum([len([t for t in r.get('detailed_trades', []) if get_val(t, 'pnl', 0) > 0]) for r in results])
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        # 指标卡片 (保持原样)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 累计总收益", f"{total_return:.2f}%", f"${total_profit:,.0f}")
        col2.metric("📉 最大回撤 (实盘)", f"{max_dd:.2f}%", help="基于滚动实盘资金曲线计算的历史最大回撤")
        col3.metric("📅 平均月度收益", f"{np.mean([r['return_pct'] for r in results]):.2f}%")
        col4.metric("🎯 全局胜率", f"{win_rate:.1f}%", f"共 {total_trades} 单")
        
        # ==========================================
        # 2. 资金增长曲线图 (保持原样)
        # ==========================================
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=cumulative_equity, 
            mode='lines', 
            name='实盘模拟资金',
            line=dict(color='#00CC96', width=3),
            fill='tozeroy' 
        ))
        fig.update_layout(
            title='资金增长曲线 (Walk-Forward Equity Curve)',
            xaxis_title='滚动时间轴 (K线计数)',
            yaxis_title='账户资金 (USDT)',
            template='plotly_dark',
            height=450,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ==========================================
        # 3. 全局法医诊断 (双重报告) - 🔥 修复调用方式
        # ==========================================
        st.markdown("---")
        st.subheader("🧬 全局策略体检 (Forensic Diagnosis)")
        
        # 3.1 调用亏损归因报告
        if hasattr(self, 'visualize_global_forensic_report'):
            try:
                # 🔥 修复：传递 results 参数
                self.visualize_global_forensic_report(results)
            except Exception as e:
                st.error(f"亏损分析报告生成失败: {e}")
        else:
            st.info("ℹ️ 亏损分析模块未加载")

        st.divider()

        # 3.2 调用盈利归因报告
        if hasattr(self, 'visualize_global_profit_report'):
            try:
                # 🔥 关键修复：传递 results 参数
                self.visualize_global_profit_report(results)
            except Exception as e:
                st.error(f"盈利分析报告生成失败: {e}")
        else:
            st.info("ℹ️ 盈利分析模块未加载")
        
        # ==========================================
        # 4. 月度详细复盘 (保持原样)
        # ==========================================
        st.markdown("---")
        st.subheader("🔍 滚动窗口详细复盘 (Monthly Breakdown)")
        st.caption("点击下方窗口，查看每个月AI进化出的具体参数和交易明细。")
        
        for res in results:
            pnl_icon = "🟢" if res['profit'] >= 0 else "🔴"
            period_num = res.get('period_num', 'N/A')
            test_range = res.get('test_range', '未知时间')
            
            title = f"{pnl_icon} 窗口 {period_num} | 时间: {test_range} | 收益: {res['return_pct']:.2f}% (${res['profit']:.0f})"
            
            with st.expander(title, expanded=False):
                
                # 获取当月最佳参数
                bp = res.get('best_params', {})
                
                # --- 辅助格式化函数 ---
                def fmt(val):
                    if isinstance(val, float): return f"{val:.2f}"
                    if isinstance(val, bool): return "✅ 开启" if val else "⛔ 关闭"
                    return str(val)

                st.markdown("#### 🛠️ 本月实战完整参数配置 (基于上月训练集)")
                
                # === 第一组：动态风控与门槛 ===
                st.caption("🛡️ **动态风控与交易门槛**")
                r1, r2, r3, r4 = st.columns(4)
                with r1: st.metric("震荡信号门槛", fmt(bp.get('sideways_threshold', 'N/A')), "防御线")
                with r2: st.metric("趋势信号门槛", fmt(bp.get('trend_threshold', 'N/A')), "进攻线")
                with r3: st.metric("震荡盈亏比", fmt(bp.get('sideways_rr', 'N/A')))
                with r4: st.metric("趋势盈亏比", fmt(bp.get('trend_rr', 'N/A')))
                
                r5, r6, r7, r8 = st.columns(4)
                with r5: st.metric("基础信号分", fmt(bp.get('min_signal_score', 'N/A')))
                with r6: st.metric("基础盈亏比", fmt(bp.get('min_rr_ratio', 'N/A')))
                with r7: st.metric("最小ADX阈值", fmt(bp.get('min_adx', 'N/A')))
                with r8: st.metric("最大波动率", fmt(bp.get('max_volatility', 0.04)))

                st.markdown("---")

                # === 第二组：技术指标 ===
                st.caption("📈 **技术指标参数 (周期设置)**")
                t1, t2, t3, t4, t5 = st.columns(5)
                with t1: st.metric("EMA 快/中", f"{bp.get('ema_fast')}/{bp.get('ema_medium')}")
                with t2: st.metric("EMA 慢/大势", f"{bp.get('ema_slow')}/{bp.get('ema_trend')}")
                with t3: st.metric("RSI 周期", bp.get('rsi_period'))
                with t4: st.metric("ATR 周期", bp.get('atr_period'))
                with t5: st.metric("ADX 周期", bp.get('adx_period'))
                
                t6, t7, t8, t9, t10 = st.columns(5)
                with t6: st.metric("布林周期", bp.get('bb_period'))
                with t7: st.metric("布林宽度(Std)", fmt(bp.get('bb_std')))
                with t8: st.metric("成交量MA", bp.get('volume_ma'))
                with t9: st.metric("BB阈值", fmt(bp.get('bb_threshold', 0.0))) 
                with t10: st.metric("-", "-")

                st.markdown("---")

                # === 第三组：SMC与逻辑开关 ===
                st.caption("🧠 **SMC 聪明钱与逻辑开关**")
                s1, s2, s3, s4, s5 = st.columns(5)
                with s1: st.metric("SMC逻辑", fmt(bp.get('use_smc_logic', False)))
                with s2: st.metric("动态风控", fmt(bp.get('use_dynamic_risk', False)))
                with s3: st.metric("FVG回溯", bp.get('fvg_lookback'))
                with s4: st.metric("波段回溯", bp.get('swing_lookback'))
                with s5: st.metric("RS相对强弱", bp.get('rs_period'))

                st.markdown("---")

                # --- B. 显示参数排行榜 ---
                st.markdown(f"#### 🏆 训练期参数竞争榜 (Top Candidates)")
                st.caption(f"说明：实战使用的是排名 #1 的参数。AI 在 {res.get('train_range', '未知')} 期间训练得出。")
                
                if 'optimization_top_list' in res and res['optimization_top_list']:
                    if hasattr(SmartMoneyVisualizer, 'create_parameter_optimization_results'):
                        param_map = globals().get('PARAM_CN_MAP', {}) 
                        SmartMoneyVisualizer.create_parameter_optimization_results(
                            res['optimization_top_list'], 
                            param_map, 
                            key_suffix=f"roll_{period_num}"
                        )
                    else:
                        st.dataframe(pd.DataFrame(res['optimization_top_list']))
                else:
                    st.warning("⚠️ 无参数榜单数据")

                st.markdown("---")
                
                # --- C. 显示实战交易详情 ---
                detailed_trades = res.get('detailed_trades', [])
                st.markdown(f"#### ⚔️ 实战交易记录 ({len(detailed_trades)} 单)")
                
                if detailed_trades:
                    if hasattr(SmartMoneyVisualizer, 'create_trade_details_table'):
                        SmartMoneyVisualizer.create_trade_details_table(detailed_trades)
                    else:
                        st.dataframe(pd.DataFrame(detailed_trades))
                else:
                    st.info("本月无交易信号 (空仓避险)")

                # D. 失败原因分析提示
                if res['return_pct'] < -5:
                    st.error("⚠️ **本月亏损分析提示**: ")
                    st.markdown("""
                    * **参数过拟合**: 上个月表现好的参数，在这个月可能失效。
                    * **止损太窄**: 检查平仓原因是否多为“止损”且持仓极短。
                    """)
    def analyze_loss_reasons(self, engine, symbol: str = None, show_in_ui: bool = True) -> dict:
        """
        🕵️‍♂️ 交易法医诊断报告 V3.0 (支持数据返回)
        """
        from collections import Counter
        
        # 默认返回空结构，防止后续报错
        empty_stats = {
            'total_loss': 0, 'good': 0, 'premature': 0, 'greedy': 0, 
            'toxic': [], 'symbol': symbol or "ALL"
        }

        if not engine.closed_trades:
            return empty_stats
            
        # 筛选亏损单 (使用 .pnl)
        if symbol:
            loss_trades = [t for t in engine.closed_trades if t.pnl < 0 and t.symbol == symbol]
            target_name = symbol
        else:
            loss_trades = [t for t in engine.closed_trades if t.pnl < 0]
            target_name = "所有币种"

        if not loss_trades:
            return empty_stats

        # 计数器
        premature_stop_count = 0 
        good_stop_count = 0      
        missed_profit_count = 0  
        toxic_reasons = []  
        
        # 获取数据
        full_df = None
        if symbol and symbol in engine.data_cache:
            if '1h' in engine.data_cache[symbol]:
                full_df = engine.data_cache[symbol]['1h']
            elif '4h' in engine.data_cache[symbol]:
                full_df = engine.data_cache[symbol]['4h']
        
        # 逐单尸检
        for t in loss_trades:
            # A. 收集毒药指标
            if hasattr(t, 'signal_reasons') and t.signal_reasons:
                toxic_reasons.extend(t.signal_reasons)
            
            # B. 贪心检测
            risk = abs(t.entry_price - t.stop_loss)
            if not hasattr(t, 'mfe') or t.mfe == 0: t.mfe = t.entry_price
            
            # 计算最大浮盈
            if t.direction == TradeDirection.LONG: # 或 'long'
                max_profit = t.mfe - t.entry_price
            else:
                max_profit = t.entry_price - t.mfe
            
            max_r = max_profit / risk if risk > 0 else 0
            if max_r > 1.5: 
                missed_profit_count += 1
                continue 

            # C. 止损质量检测
            if t.exit_reason == "止损" and full_df is not None:
                # 寻找索引
                if t.exit_time in full_df.index:
                    exit_idx = full_df.index.get_loc(t.exit_time)
                else:
                    exit_idx = full_df.index.searchsorted(t.exit_time)

                # 自适应回溯窗口
                hold_bars = int(t.duration_hours) if hasattr(t, 'duration_hours') else 24
                look_ahead = min(max(24, int(hold_bars / 2)), 100)

                if isinstance(exit_idx, int) and exit_idx + look_ahead < len(full_df):
                    post_data = full_df.iloc[exit_idx+1 : exit_idx+look_ahead] 
                    
                    if t.direction == TradeDirection.LONG: # 或 'long'
                        if post_data['high'].max() > t.entry_price:
                            premature_stop_count += 1
                        else:
                            good_stop_count += 1
                    else:
                        if post_data['low'].min() < t.entry_price:
                            premature_stop_count += 1
                        else:
                            good_stop_count += 1
                else:
                    good_stop_count += 1
            else:
                good_stop_count += 1

        # 统计
        total_loss = len(loss_trades)
        washout_rate = (premature_stop_count / total_loss) * 100
        miss_rate = (missed_profit_count / total_loss) * 100
        good_rate = (good_stop_count / total_loss) * 100
        
        # UI 显示 (保持原样)
        if show_in_ui:
            # 简化的建议逻辑
            suggestion_color = "green"
            if washout_rate > 50: suggestion_color = "orange"
            elif miss_rate > 30: suggestion_color = "red"
            
            with st.expander(f"🕵️‍♂️ 法医诊断: {target_name} (亏损 {total_loss} 单)", expanded=False):
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("🛡️ 有效止损", f"{good_stop_count}", f"{good_rate:.0f}%")
                with c2: st.metric("🤕 被洗盘", f"{premature_stop_count}", f"{washout_rate:.0f}%", delta_color="inverse")
                with c3: st.metric("💸 利润回撤", f"{missed_profit_count}", f"{miss_rate:.0f}%", delta_color="inverse")
                
                # 只有 UI 显示时才展示毒药指标详情，防止后台刷屏
                top_toxic = Counter(toxic_reasons).most_common(3)
                if top_toxic:
                    st.caption("☠️ 主要致死诱因:")
                    for r, c in top_toxic:
                        st.text(f"- {r}: {c}单")

        # 🔥 返回结构化数据供汇总使用
        return {
            'symbol': target_name,
            'total_loss': total_loss,
            'good': good_stop_count,
            'premature': premature_stop_count,
            'greedy': missed_profit_count,
            'toxic': toxic_reasons # 返回原始列表供全局统计
        }
    # ==========================================
    # 🔥 [新增功能] 盈利解剖核心逻辑 (插入在 visualize_window_diagnosis 之前)
    # ==========================================
    def create_profit_analysis_tab(self, trades: List[Any], period_num: int):
        """
        盈利分析Tab：详细分析盈利单的离场方式
        兼容 Dictionary 和 Trade Object 两种数据格式，防止报错
        """
        # --- 内部辅助：安全访问函数 ---
        def get_val(item, attr_name, dict_key, default=None):
            if isinstance(item, dict):
                return item.get(dict_key, default)
            return getattr(item, attr_name, default)

        def set_val(item, attr_name, dict_key, value):
            if isinstance(item, dict):
                item[dict_key] = value
            else:
                setattr(item, attr_name, value)

        # 1. 筛选盈利单
        win_trades = [t for t in trades if get_val(t, 'pnl', 'pnl', 0) > 0]
        
        if not win_trades:
            st.info("🐻 本周期没有盈利单，无法分析利润结构。")
            return
        
        # 2. 收集/推断退出原因
        exit_reasons = []
        for trade in win_trades:
            # 获取关键字段
            exit_reason = get_val(trade, 'exit_reason', 'exit_reason')
            exit_price = get_val(trade, 'exit_price', 'exit_price')
            entry_price = get_val(trade, 'entry_price', 'entry_price')
            take_profit = get_val(trade, 'take_profit', 'take_profit')
            direction = get_val(trade, 'direction', 'direction') 

            # 智能推断逻辑 (如果没有记录原因，根据价格倒推)
            if not exit_reason:
                if exit_price is not None and entry_price is not None:
                    # 兼容枚举或字符串判断
                    is_long = str(direction).lower() in ['long', 'tradedirection.long', '1']
                    
                    if is_long:
                        if take_profit and exit_price >= take_profit:
                            exit_reason = "🎯 主动止盈(固定TP)"
                        elif exit_price > entry_price:
                            exit_reason = "🛡️ 被动止盈(移动止损)"
                        else:
                            exit_reason = "🔴 亏损止损" # 理论上不该进这里
                    else: # Short
                        if take_profit and exit_price <= take_profit:
                            exit_reason = "🎯 主动止盈(固定TP)"
                        elif exit_price < entry_price:
                            exit_reason = "🛡️ 被动止盈(移动止损)"
                        else:
                            exit_reason = "🔴 亏损止损"
                else:
                    exit_reason = "❓ 未知原因"
                
                # 回写推断结果
                set_val(trade, 'exit_reason', 'exit_reason', exit_reason)
            
            exit_reasons.append(exit_reason)
        
        # 3. 创建DataFrame
        df_wins = pd.DataFrame({
            'exit_reason': exit_reasons,
            'pnl': [get_val(t, 'pnl', 'pnl') for t in win_trades],
            'pnl_percent': [get_val(t, 'return_pct', 'return_pct', 0)*100 for t in win_trades],
            'symbol': [get_val(t, 'symbol', 'symbol') for t in win_trades]
        })
        
        # 4. 可视化绘图
        reason_counts = df_wins['exit_reason'].value_counts()
        
        color_map = {
            "🎯 主动止盈(固定TP)": "#FF6B6B", 
            "📈 移动止盈(追踪止损)": "#4ECDC4", 
            "🛡️ 被动止盈(移动止损)": "#45B7D1", 
            "🛡️ 被动止盈(保本止损)": "#96CEB4",
            "⏰ 时间止损(持仓超时)": "#FFEAA7", 
            "🔴 亏损止损": "#D7263D",
            "❓ 未知原因": "#95a5a6"
        }

        # 图1：饼图 (数量占比)
        fig_pie = px.pie(
            values=reason_counts.values,
            names=reason_counts.index,
            title="📊 盈利单离场方式分布 (数量)",
            color=reason_counts.index,
            color_discrete_map=color_map,
            hole=0.4
        )
        
        # 图2：条形图 (含金量分析)
        avg_pnl_by_reason = df_wins.groupby('exit_reason')['pnl_percent'].mean().sort_values()
        
        fig_bar = px.bar(
            x=avg_pnl_by_reason.values,
            y=avg_pnl_by_reason.index,
            orientation='h',
            title="💰 哪种离场方式赚得更多？(平均收益率%)",
            labels={'x': '平均收益率(%)', 'y': ''},
            color=avg_pnl_by_reason.index,
            color_discrete_map=color_map,
            text_auto='.2f'
        )
        fig_bar.update_layout(showlegend=False)

        # 布局显示
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.plotly_chart(fig_pie, use_container_width=True, key=f"win_pie_{period_num}")
        with c2:
            st.plotly_chart(fig_bar, use_container_width=True, key=f"win_bar_{period_num}")

        # 5. 智能诊断文案
        st.markdown("#### 🧠 盈利结构诊断")
        
        total_wins = len(win_trades)
        hard_tp_count = reason_counts.get("🎯 主动止盈(固定TP)", 0)
        trailing_count = reason_counts.get("🛡️ 被动止盈(移动止损)", 0) + reason_counts.get("📈 移动止盈(追踪止损)", 0)
        
        insights = []
        if total_wins > 0:
            tp_ratio = hard_tp_count / total_wins
            
            # 诊断 1: 截断利润风险
            if tp_ratio > 0.6:
                insights.append(f"⚠️ **严重截断利润**：{tp_ratio:.1%} 的单子都是固定止盈出局。建议在牛市中**移除或调大固定止盈**，让利润奔跑！")
            elif tp_ratio < 0.2:
                insights.append(f"✅ **奔跑吧利润**：大部分订单没有被固定止盈限制住，符合趋势策略特征。")
                
            # 诊断 2: 移动止损效率
            if trailing_count > 0:
                avg_trail_pnl = df_wins[df_wins['exit_reason'].str.contains("移动|追踪", na=False)]['pnl_percent'].mean()
                avg_fix_pnl = df_wins[df_wins['exit_reason'].str.contains("固定", na=False)]['pnl_percent'].mean()
                
                if pd.notna(avg_fix_pnl) and avg_fix_pnl > 0:
                    ratio = avg_trail_pnl / avg_fix_pnl
                    if ratio > 1.2:
                        insights.append(f"💎 **移动止损真香**：移动止损单的平均利润是固定止盈单的 **{ratio:.1f}倍**。坚持用它！")
                    elif ratio < 0.8:
                        insights.append(f"🔧 **移动止损太紧**：移动止损虽然保住了命，但平均利润不如直接止盈。建议**放宽回调阈值**。")

        for i in insights: st.info(i)

    def visualize_window_diagnosis(self, engine, period_num: int):
        """
        📊 窗口级深度复盘仪表盘 (最终增强版：集成法医分析与盈利解剖)
        """
        import plotly.express as px
        
        # 1. 获取基础数据
        trades = engine.closed_trades
        if not trades:
            st.warning(f"窗口 {period_num} 无交易数据")
            return

        # 2. 🔥【核心替换】调用高级法医分析器 (替代旧的手工统计循环)
        # 这一步会自动完成所有的亏损归因、贪心检测、止损质量评估
        loss_stats = self.loss_analyzer.analyze_loss_reasons_enhanced(engine, show_in_ui=False)
        
        # 3. 简单的极值统计 (保留用于显示"高光时刻")
        sorted_by_pnl = sorted(trades, key=lambda x: getattr(x, 'pnl', 0))
        worst_trade = sorted_by_pnl[0] if sorted_by_pnl else None
        best_trade = sorted_by_pnl[-1] if sorted_by_pnl else None
        
        # 4. 简单的币种统计 (保留用于 Tab 3)
        symbol_stats = {}
        for t in trades:
            s = t.symbol.split('/')[0]
            if s not in symbol_stats:
                symbol_stats[s] = {'pnl': 0.0, 'loss_count': 0, 'win_count': 0}
            symbol_stats[s]['pnl'] += t.pnl
            if t.pnl < 0: symbol_stats[s]['loss_count'] += 1
            else: symbol_stats[s]['win_count'] += 1

        # ==================== UI 渲染 ====================
        
        with st.expander(f"🧬 窗口 {period_num} 深度复盘报告 (点击展开详情)", expanded=False):
            
            # --- 第一行：高光与至暗时刻 (UI保持不变) ---
            c1, c2 = st.columns(2)
            with c1:
                if best_trade and best_trade.pnl > 0:
                    st.success(f"🏆 **盈利王**: {best_trade.symbol}")
                    st.caption(f"💰 +${best_trade.pnl:.0f} (+{best_trade.pnl_percent:.1f}%) | 持仓: {best_trade.duration_hours:.1f}h")
                    # 兼容不同格式的 reasons
                    reasons = getattr(best_trade, 'signal_reasons', []) or getattr(best_trade, 'entry_reasons', [])
                    if isinstance(reasons, list) and reasons:
                        st.caption(f"🚀 原因: {', '.join(reasons[:2])}")
            with c2:
                if worst_trade and worst_trade.pnl < 0:
                    st.error(f"💀 **亏损王**: {worst_trade.symbol}")
                    st.caption(f"💸 -${abs(worst_trade.pnl):.0f} ({worst_trade.pnl_percent:.1f}%) | 持仓: {worst_trade.duration_hours:.1f}h")
                    reasons = getattr(worst_trade, 'signal_reasons', []) or getattr(worst_trade, 'entry_reasons', [])
                    if isinstance(reasons, list) and reasons:
                        st.caption(f"🥀 诱因: {', '.join(reasons[:2])}")

            st.divider()

            # --- 定义三个标签页 ---
            t1, t2, t3 = st.tabs(["☠️ 亏损法医(Pro)", "💰 盈利解剖(Pro)", "📊 币种分布"])
            
            # --- Tab 1: 亏损诊断 (全面升级) ---
        with t1:
            if loss_stats['total_loss'] > 0:
                # 1. 显示核心指标卡片
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("总亏损单", loss_stats['total_loss'])
                c2.metric("🟢 有效止损", f"{loss_stats['good']} ({loss_stats['good']/loss_stats['total_loss']:.1%})")
                c3.metric("🟠 被洗盘(过窄)", f"{loss_stats['premature']} ({loss_stats['premature']/loss_stats['total_loss']:.1%})")
                c4.metric("🔴 贪心回撤", f"{loss_stats['greedy']} ({loss_stats['greedy']/loss_stats['total_loss']:.1%})")
                
                # 2. 调用图表绘制函数 (修复3)
                fig = self.create_stop_loss_analysis_chart(loss_stats, period_num)
                st.plotly_chart(fig, use_container_width=True, key=f"loss_adv_{period_num}")
                
                # 3. 显示智能建议 (修复4)
                st.markdown("### 🧠 首席法医建议")
                advice_list = self.generate_stop_loss_advice(loss_stats)
                for advice in advice_list:
                    st.info(advice)
                    
            else:
                st.success("🎉 本周期无亏损单，完美！")
            
            # --- Tab 2: 盈利解剖 (调用新方法) ---
            with t2:
                # 获取详细交易列表
                # 尝试从 results 中获取，如果 self.results 还没更新，就用当前的 trades
                # 为了稳妥，我们直接传当前的 trades (包含了所有 closed_trades)
                self.create_profit_analysis_tab(trades, period_num)
            
            # --- Tab 3: 币种分布 (保留原有逻辑) ---
            with t3:
                sym_data = []
                for s, stats in symbol_stats.items():
                    sym_data.append({'币种': s, '盈亏': stats['pnl'], '亏损单数': stats['loss_count']})
                
                if sym_data:
                    df_sym = pd.DataFrame(sym_data)
                    fig2 = px.bar(df_sym, x='币种', y='盈亏', color='盈亏', 
                                  color_continuous_scale='RdYlGn', title="各币种盈亏贡献")
                    st.plotly_chart(fig2, use_container_width=True, key=f"sym_pnl_{period_num}")
    def create_stop_loss_analysis_chart(self, loss_stats: Dict, period_num: int):
        """止损质量细分分析图表"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('止损类型分布', '被洗盘细分(按持仓)', 
                           '有效止损细分(按市场)', '贪心回撤细分(按波动)'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # 1. 饼图
        labels = ['有效止损', '被洗盘', '贪心回撤']
        values = [loss_stats['good'], loss_stats['premature'], loss_stats['greedy']]
        fig.add_trace(go.Pie(labels=labels, values=values, hole=0.3, 
                             marker_colors=['#2E8B57', '#FFA500', '#CD5C5C']), row=1, col=1)
        
        # 2. 被洗盘细分
        br = loss_stats['breakdown']['premature_by_duration']
        fig.add_trace(go.Bar(x=list(br.keys()), y=list(br.values()), name='被洗盘', marker_color='#FFA500'), row=1, col=2)
        
        # 3. 有效止损细分
        bg = loss_stats['breakdown']['good_by_market_regime']
        fig.add_trace(go.Bar(x=list(bg.keys()), y=list(bg.values()), name='有效止损', marker_color='#2E8B57'), row=2, col=1)
        
        # 4. 贪心细分
        bv = loss_stats['breakdown']['greedy_by_volatility']
        fig.add_trace(go.Bar(x=list(bv.keys()), y=list(bv.values()), name='贪心', marker_color='#CD5C5C'), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, title_text=f"窗口 {period_num} 止损深度分析")
        return fig

    def generate_stop_loss_advice(self, loss_stats: Dict) -> List[str]:
        """生成建议"""
        advice = []
        total = loss_stats['total_loss']
        if total == 0: return ["✅ 无亏损单"]
        
        if loss_stats['premature'] / total > 0.4:
            advice.append("⚠️ **止损过窄**：超过40%的亏损是被洗盘。建议放大ATR止损倍数。")
        if loss_stats['greedy'] / total > 0.3:
            advice.append("💸 **止盈太贪**：超过30%的单子是利润回撤。建议启用移动止盈。")
        if loss_stats['good'] / total > 0.6:
            advice.append("✅ **止损健康**：大部分亏损是正常的趋势反转。")
            
        return advice
    def visualize_global_forensic_report(self, all_results: list):
        """
        🌍 全局法医诊断总报告 (汇总所有窗口数据)
        """
        import plotly.express as px
        import plotly.graph_objects as go
        from collections import Counter, defaultdict
        import pandas as pd

        # 1. 数据聚合容器
        global_stats = {
            'total_loss_count': 0,
            'good_stop': 0,
            'premature': 0,
            'greedy': 0,
            'toxic_reasons': [],
            'symbol_loss_counts': defaultdict(int),
            'symbol_pnl': defaultdict(float) # 统计真实盈亏金额
        }

        has_data = False

        # 2. 遍历所有窗口结果
        for res in all_results:
            # A. 聚合法医诊断数据 (亏损原因)
            if 'diagnosis_report' in res and res['diagnosis_report']:
                has_data = True
                for sym, diag in res['diagnosis_report'].items():
                    global_stats['total_loss_count'] += diag.get('total_loss', 0)
                    global_stats['good_stop'] += diag.get('good', 0)
                    global_stats['premature'] += diag.get('premature', 0)
                    global_stats['greedy'] += diag.get('greedy', 0)
                    global_stats['toxic_reasons'].extend(diag.get('toxic', []))
                    global_stats['symbol_loss_counts'][sym] += diag.get('total_loss', 0)

            # B. 聚合交易盈亏数据 (从详细交易记录中提取)
            if 'detailed_trades' in res:
                for t in res['detailed_trades']:
                    # 这里 t 可能是对象也可能是字典，做个兼容
                    pnl = getattr(t, 'pnl', 0) if hasattr(t, 'pnl') else t.get('pnl', 0)
                    symbol = getattr(t, 'symbol', 'Unknown') if hasattr(t, 'symbol') else t.get('symbol', 'Unknown')
                    s_clean = symbol.split('/')[0]
                    global_stats['symbol_pnl'][s_clean] += pnl

        if not has_data:
            return

        # ==================== UI 渲染 ====================
        st.markdown("---")
        st.subheader("💀 全局法医验尸报告 (All-Time Forensic Report)")
        st.caption("基于所有滚动窗口的汇总统计，揭示策略的根本弱点。")

        # 1. 核心指标概览
        total = global_stats['total_loss_count']
        if total > 0:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📉 总亏损单数", total)
            
            p_good = global_stats['good_stop'] / total * 100
            c2.metric("🛡️ 有效止损率", f"{p_good:.1f}%", f"{global_stats['good_stop']}单", help="趋势反转，跑得对")
            
            p_premature = global_stats['premature'] / total * 100
            c3.metric("🤕 被洗盘率", f"{p_premature:.1f}%", f"{global_stats['premature']}单", delta_color="inverse", help="止损太窄")
            
            p_greedy = global_stats['greedy'] / total * 100
            c4.metric("💸 贪心回撤率", f"{p_greedy:.1f}%", f"{global_stats['greedy']}单", delta_color="inverse", help="曾经大赚没走")

            # 2. 亏损原因饼图
            fig_pie = go.Figure(data=[go.Pie(
                labels=['有效止损 (趋势不对)', '被洗盘 (止损太窄)', '利润回撤 (止盈太贪)'],
                values=[global_stats['good_stop'], global_stats['premature'], global_stats['greedy']],
                hole=.4,
                marker_colors=['#2E8B57', '#FFA500', '#CD5C5C']
            )])
            fig_pie.update_layout(title_text="🛑 到底是怎么亏的？(全局占比)", height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        # 3. 毒药指标总榜 (Top 10)
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.markdown("##### ☠️ 全局“毒药”指标 Top 10")
            st.caption("这些开仓信号在历史上导致亏损次数最多：")
            if global_stats['toxic_reasons']:
                top_toxic = Counter(global_stats['toxic_reasons']).most_common(10)
                df_toxic = pd.DataFrame(top_toxic, columns=['信号', '致死次数'])
                df_toxic = df_toxic.sort_values('致死次数', ascending=True)
                
                fig_bar = px.bar(df_toxic, x='致死次数', y='信号', orientation='h', text='致死次数')
                fig_bar.update_traces(marker_color='#D32F2F', textposition='outside')
                fig_bar.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("无信号记录")

        # 4. 币种红黑榜 (盈亏金额)
        with c_right:
            st.markdown("##### 💰 币种提款机 vs 碎钞机")
            st.caption("各币种在整个回测期间的累计净盈亏 (Net PnL)：")
            if global_stats['symbol_pnl']:
                # 转换数据
                pnl_data = [{'币种': k, '累计盈亏': v} for k, v in global_stats['symbol_pnl'].items()]
                df_pnl = pd.DataFrame(pnl_data).sort_values('累计盈亏', ascending=False)
                
                fig_pnl = px.bar(df_pnl, x='累计盈亏', y='币种', orientation='h', 
                                 color='累计盈亏', color_continuous_scale='RdYlGn', text='累计盈亏')
                fig_pnl.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                fig_pnl.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_pnl, use_container_width=True)
    def visualize_global_profit_report(self, results):
        """
        🌍 全局盈利归因报告 (Global Profit Forensic Report)
        汇总所有窗口的盈利单，分析整体获利结构
        """
        import plotly.express as px
        import pandas as pd
        
        # --- 1. 数据聚合 ---
        all_trades = []
        for res in results:  # 🔥 修复：使用传入的 results 参数
            if 'detailed_trades' in res:
                all_trades.extend(res['detailed_trades'])
        
        if not all_trades:
            st.warning("全局无交易数据")
            return

        # 辅助函数：安全获取属性
        def get_val(item, key, default=None):
            if isinstance(item, dict): return item.get(key, default)
            return getattr(item, key, default)

        # 筛选盈利单
        win_trades = [t for t in all_trades if get_val(t, 'pnl', 0) > 0]
        
        if not win_trades:
            st.info("全局无盈利单")
            return

        # --- 2. 核心逻辑：全局离场原因标准化 ---
        exit_reasons = []
        pnl_percents = []
        durations = []
        
        for t in win_trades:
            raw = get_val(t, 'exit_reason', '未知')
            entry_p = get_val(t, 'entry_price', 0)
            exit_p = get_val(t, 'exit_price', 0)
            direction = get_val(t, 'direction', 'long')
            pnl_pct = get_val(t, 'return_pct', 0) * 100
            
            # 强制翻译逻辑
            final_reason = raw
            if raw == "止盈":
                final_reason = "🎯 主动止盈(固定TP)"
            elif raw == "止损":
                final_reason = "🛡️ 被动止盈(移动止损)"
            elif raw == "回测结束平仓":
                final_reason = "⏰ 时间止损"
            elif not raw or raw == "未知":
                # 推断
                is_long = str(direction).lower() in ['long', '1', 'tradedirection.long']
                if exit_p and entry_p:
                    if is_long:
                        final_reason = "🎯 主动止盈(固定TP)" if exit_p > entry_p * 1.01 else "🛡️ 被动止盈(移动止损)"
                    else:
                        final_reason = "🎯 主动止盈(固定TP)" if exit_p < entry_p * 0.99 else "🛡️ 被动止盈(移动止损)"
            
            exit_reasons.append(final_reason)
            pnl_percents.append(pnl_pct)
            durations.append(get_val(t, 'duration_hours', 0))

        # --- 3. 构建分析 DataFrame ---
        df_global = pd.DataFrame({
            'Reason': exit_reasons,
            'PnL_Pct': pnl_percents,
            'Duration': durations
        })

        # --- 4. UI 渲染 ---
        with st.expander("💰 全局盈利归因报告 (Global Profit Forensic Report)", expanded=True):
            
            # A. 核心指标卡片
            k1, k2, k3, k4 = st.columns(4)
            total_wins = len(win_trades)
            avg_win = df_global['PnL_Pct'].mean()
            max_win = df_global['PnL_Pct'].max()
            
            # 计算移动止损占比
            trailing_count = len(df_global[df_global['Reason'].str.contains('移动|追踪')])
            trailing_ratio = trailing_count / total_wins if total_wins > 0 else 0
            
            k1.metric("全局盈利单数", total_wins)
            k2.metric("平均单笔盈利", f"{avg_win:.2f}%")
            k3.metric("最大单笔神单", f"{max_win:.2f}%")
            k4.metric("移动止损占比", f"{trailing_ratio:.1%}", help="越高说明越能吃到趋势")

            st.divider()

            # B. 图表分析区
            c1, c2 = st.columns([1, 1.3])
            
            # 颜色映射
            color_map = {
                "🎯 主动止盈(固定TP)": "#FF6B6B", 
                "📈 移动止盈(追踪止损)": "#4ECDC4", 
                "🛡️ 被动止盈(移动止损)": "#45B7D1", 
                "🛡️ 被动止盈(保本止损)": "#96CEB4",
                "⏰ 时间止损": "#FFEAA7", 
                "❓ 未知": "#95a5a6"
            }
            
            with c1:
                # 图1：全局分布饼图
                reason_counts = df_global['Reason'].value_counts()
                fig_pie = px.pie(
                    values=reason_counts.values,
                    names=reason_counts.index,
                    title="全局获利来源分布 (数量)",
                    color=reason_counts.index,
                    color_discrete_map=color_map,
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with c2:
                # 图2：全局含金量分析
                avg_pnl = df_global.groupby('Reason')['PnL_Pct'].mean().sort_values()
                fig_bar = px.bar(
                    x=avg_pnl.values,
                    y=avg_pnl.index,
                    orientation='h',
                    title="哪种方式在长跑中赚得更多？(平均收益%)",
                    labels={'x': '平均收益率(%)', 'y': ''},
                    color=avg_pnl.index,
                    color_discrete_map=color_map,
                    text_auto='.2f'
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # C. 深度洞察 (全局版)
            st.markdown("##### 🧠 全局策略定性")
            
            fix_pnl = df_global[df_global['Reason'].str.contains('固定')]['PnL_Pct'].mean()
            trail_pnl = df_global[df_global['Reason'].str.contains('移动|追踪')]['PnL_Pct'].mean()
            
            if pd.isna(fix_pnl): fix_pnl = 0
            if pd.isna(trail_pnl): trail_pnl = 0
            
            insight_cols = st.columns(1)
            with insight_cols[0]:
                if trail_pnl > fix_pnl * 1.3:
                    st.success(f"🚀 **趋势收割机**：在所有历史窗口中，移动止损单的平均盈利 ({trail_pnl:.1f}%) 远超固定止盈 ({fix_pnl:.1f}%)。说明策略在捕捉大趋势方面表现优异，建议继续保持或放大移动止损的权重。")
                elif fix_pnl > trail_pnl:
                    st.warning(f"📉 **短视策略风险**：固定止盈的平均盈利 ({fix_pnl:.1f}%) 超过了移动止损 ({trail_pnl:.1f}%)。这说明策略经常在大行情来临前就'吃一口跑了'，或者移动止损设置得太容易被洗盘。建议：**放宽移动止损间距**。")
                else:
                    st.info("⚖️ **平衡型策略**：固定止盈和移动止损带来的收益差距不大。")
class MonteCarloRollingValidator:
    """蒙特卡洛滚动验证器"""
    
    @staticmethod
    def run_monte_carlo_validation(config, data_cache, n_simulations=50):
        """
        随机抽取不同的起止时间，运行多次滚动回测，检验策略是否只在特定时间段有效
        """
        results_all = []
        progress_bar = st.progress(0)
        
        st.info(f"🎲 开始蒙特卡洛压力测试 (模拟 {n_simulations} 次不同起点的实盘)...")
        
        # 定义可能的起始月份池 (2023年全年)
        start_months = [f"2023-{m:02d}-01" for m in range(1, 10)]
        
        for sim in range(n_simulations):
            # 随机参数
            rand_start = np.random.choice(start_months)
            rand_train = np.random.choice([3, 6]) # 随机训练3个月或6个月
            
            tester = RollingWindowBacktester(config, data_cache)
            
            # 跑一轮短期的滚动
            res, eq = tester.run_6plus1_validation(
                start_date=rand_start,
                end_date="2024-06-01", # 统一结束时间
                train_months=rand_train,
                test_months=1,
                n_optimization_trials=15 # 蒙特卡洛为了速度稍微降低优化次数
            )
            
            if eq:
                tot_ret = (eq[-1] - eq[0]) / eq[0]
                results_all.append(tot_ret)
            
            progress_bar.progress((sim + 1) / n_simulations)
            
        return results_all

# ==========================================
    # 🧪 窗口优化对比测试模块
    # ==========================================
    
    def compare_optimization_windows(self, config: Dict, data_cache: Dict):
        """
        对比不同优化窗口的效果 - 主函数
        """
        import numpy as np
        
        results = {}
        
        # 定义要测试的窗口组合
        windows_to_test = [
            {'name': '1个月训练+1周测试', 'train_months': 1, 'test_weeks': 1},
            {'name': '2个月训练+2周测试', 'train_months': 2, 'test_weeks': 2},
            {'name': '3个月训练+1个月测试', 'train_months': 3, 'test_weeks': 4},
            {'name': '6个月训练+1个月测试', 'train_months': 6, 'test_weeks': 4},
            {'name': '1年训练+3个月测试', 'train_months': 12, 'test_weeks': 13},
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, window in enumerate(windows_to_test):
            progress = idx / len(windows_to_test)
            progress_bar.progress(progress)
            status_text.text(f"正在测试: {window['name']} ({idx+1}/{len(windows_to_test)})")
            
            # 运行滚动优化
            window_results = self.rolling_window_optimization_v2(
                config=config,
                data_cache=data_cache,
                train_months=window['train_months'],
                test_weeks=window['test_weeks']
            )
            
            # 分析结果
            if window_results:
                stats = self.analyze_window_results(window_results)
                results[window['name']] = {
                    'stats': stats,
                    'raw_results': window_results
                }
            else:
                st.warning(f"窗口 {window['name']} 没有获得有效结果")
        
        progress_bar.progress(1.0)
        status_text.text("✅ 所有窗口测试完成！")
        
        # 绘制对比图
        if results:
            self.plot_window_comparison(results)
        
        return results
    
    def rolling_window_optimization_v2(self, config: Dict, data_cache: Dict,
                                      train_months: int, test_weeks: int):
        """
        通用滚动优化函数，支持任意训练和测试窗口
        """
        import pandas as pd
        from datetime import datetime, timedelta
        import numpy as np
        
        # 获取数据时间范围
        all_dates = []
        for symbol_data in data_cache.values():
            for timeframe, df in symbol_data.items():
                if timeframe in ['1h', '4h'] and not df.empty:
                    all_dates.append(df.index.min())
                    all_dates.append(df.index.max())
        
        if not all_dates:
            st.error("❌ 数据缓存为空")
            return []
        
        global_start = min(all_dates)
        global_end = max(all_dates)
        
        st.write(f"📊 数据范围: {global_start.strftime('%Y-%m-%d')} 到 {global_end.strftime('%Y-%m-%d')}")
        
        all_results = []
        current_train_start = global_start
        
        # 计算需要多少内存
        total_days = (global_end - global_start).days
        num_windows = max(1, total_days // (test_weeks * 7))
        
        st.write(f"🔢 将进行约 {num_windows} 次滚动优化")
        
        window_count = 0
        while True:
            # 计算训练结束时间
            train_end = current_train_start + timedelta(days=train_months * 30)
            
            # 计算测试结束时间
            test_end = train_end + timedelta(days=test_weeks * 7)
            
            # 如果测试期超出数据范围，停止
            if test_end > global_end:
                break
            
            # 🚀 快速优化版本（减少计算时间）
            # 训练集
            train_config = config.copy()
            train_config['start_date'] = current_train_start.strftime('%Y-%m-%d')
            train_config['end_date'] = train_end.strftime('%Y-%m-%d')
            
            # 🔧 简化优化：只做10次贝叶斯试验（为了速度）
            try:
                train_results = self.bayesian_optimization(
                    train_config, data_cache, n_trials=10  # 注意：减少试验次数以加速
                )
            except Exception as e:
                st.warning(f"训练期优化失败: {str(e)}")
                current_train_start = current_train_start + timedelta(days=test_weeks * 7)
                continue
            
            if not train_results:
                # 无数据，跳过这个窗口
                current_train_start = current_train_start + timedelta(days=test_weeks * 7)
                continue
            
            # 获取最佳参数
            best_params = train_results[0]['params']
            
            # 测试集
            test_config = config.copy()
            test_config.update(best_params)
            test_config['start_date'] = train_end.strftime('%Y-%m-%d')
            test_config['end_date'] = test_end.strftime('%Y-%m-%d')
            
            # 运行回测
            try:
                engine = UnifiedBacktestEngine(test_config, data_cache)
                test_stats = engine.run_backtest()
                
                # 记录结果
                result = {
                    'period': f"{train_end.strftime('%Y-%m')}",
                    'train_days': (train_end - current_train_start).days,
                    'test_days': (test_end - train_end).days,
                    'params': best_params,
                    'stats': test_stats,
                    'return_pct': test_stats.get('total_return', 0) if test_stats else 0,
                    'sharpe': test_stats.get('sharpe', 0) if test_stats else 0,
                    'max_dd': test_stats.get('max_drawdown', 0) if test_stats else 0,
                    'win_rate': test_stats.get('win_rate', 0) if test_stats else 0,
                }
                all_results.append(result)
                
                window_count += 1
                if window_count % 3 == 0:
                    st.write(f"  已完成 {window_count} 个窗口...")
                    
            except Exception as e:
                st.warning(f"测试期回测失败: {str(e)}")
            
            # 移动到下一个窗口
            current_train_start = current_train_start + timedelta(days=test_weeks * 7)
        
        st.write(f"✅ 完成 {len(all_results)} 次滚动优化")
        return all_results
    
    def analyze_window_results(self, window_results: List[Dict]):
        """
        分析单个窗口优化的结果
        """
        import numpy as np
        
        if not window_results:
            return {}
        
        returns = [r['return_pct'] for r in window_results if r['return_pct'] is not None]
        sharpes = [r['sharpe'] for r in window_results if r['sharpe'] is not None]
        drawdowns = [r['max_dd'] for r in window_results if r['max_dd'] is not None]
        win_rates = [r['win_rate'] for r in window_results if r['win_rate'] is not None]
        
        if not returns:
            return {}
        
        # 计算统计指标
        stats = {
            'avg_return': np.mean(returns) if returns else 0,
            'std_return': np.std(returns) if len(returns) > 1 else 0,
            'min_return': np.min(returns) if returns else 0,
            'max_return': np.max(returns) if returns else 0,
            'winning_periods': sum(1 for r in returns if r > 0),
            'total_periods': len(returns),
            'win_rate_periods': sum(1 for r in returns if r > 0) / len(returns) * 100 if returns else 0,
            'avg_sharpe': np.mean(sharpes) if sharpes else 0,
            'avg_drawdown': np.mean(drawdowns) if drawdowns else 0,
            'max_drawdown': np.max(drawdowns) if drawdowns else 0,
            'avg_win_rate': np.mean(win_rates) if win_rates else 0,
            'consistency_score': self.calculate_consistency(returns),
        }
        
        return stats
    
    def calculate_consistency(self, returns: List[float]) -> float:
        """
        计算一致性分数
        0-100分，越高表示收益越稳定
        """
        import numpy as np
        
        if not returns or len(returns) < 2:
            return 50.0  # 默认分数
        
        # 计算变异系数的倒数（越小越稳定）
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if mean_return == 0:
            cv = 1.0
        else:
            cv = std_return / abs(mean_return)
        
        # 转换为0-100分
        # cv越小越好，所以用指数衰减函数
        consistency = 100 * np.exp(-cv * 0.5)
        
        # 考虑正收益比例
        positive_ratio = sum(1 for r in returns if r > 0) / len(returns)
        consistency = consistency * (0.3 + 0.7 * positive_ratio)
        
        return min(100, max(0, consistency))
    
    def plot_window_comparison(self, results: Dict):
        """
        绘制不同窗口的对比图
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
        
        # 准备数据
        window_names = list(results.keys())
        
        # 提取统计数据
        avg_returns = [results[name]['stats']['avg_return'] for name in window_names]
        win_rates = [results[name]['stats']['win_rate_periods'] for name in window_names]
        sharpes = [results[name]['stats']['avg_sharpe'] for name in window_names]
        max_dds = [results[name]['stats']['max_drawdown'] for name in window_names]
        consistency = [results[name]['stats']['consistency_score'] for name in window_names]
        
        # 计算综合评分
        composite_scores = []
        for name in window_names:
            stats = results[name]['stats']
            # 加权综合评分
            score = (
                stats['avg_return'] * 0.3 +
                stats['win_rate_periods'] * 0.2 +
                stats['avg_sharpe'] * 0.3 -
                stats['max_drawdown'] * 0.2 +
                stats['consistency_score'] * 0.1
            )
            composite_scores.append(score)
        # 保存结果到session_state
        if 'streamlit' in str(type(st)):  # 确保在Streamlit环境中
            st.session_state.last_window_results = results
        # 创建对比表格
        st.subheader("📊 窗口对比结果表格")
        
        comparison_data = []
        for i, name in enumerate(window_names):
            stats = results[name]['stats']
            comparison_data.append({
                '窗口设置': name,
                '平均收益率': f"{stats['avg_return']:.2f}%",
                '正收益期占比': f"{stats['win_rate_periods']:.1f}%",
                '平均夏普': f"{stats['avg_sharpe']:.2f}",
                '最大回撤': f"{stats['max_drawdown']:.2f}%",
                '一致性分数': f"{stats['consistency_score']:.0f}",
                '综合评分': f"{composite_scores[i]:.2f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('平均收益率', '正收益期占比', '平均夏普比率', 
                           '最大回撤', '一致性分数', '综合评分'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # 颜色方案
        colors = px.colors.qualitative.Set3
        
        # 绘制每个指标
        fig.add_trace(
            go.Bar(x=window_names, y=avg_returns, name='平均收益率', 
                  marker_color=colors[0],
                  hovertemplate='%{x}<br>收益率: %{y:.2f}%<extra></extra>'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=window_names, y=win_rates, name='正收益期占比', 
                  marker_color=colors[1],
                  hovertemplate='%{x}<br>正收益期: %{y:.1f}%<extra></extra>'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=window_names, y=sharpes, name='夏普比率', 
                  marker_color=colors[2],
                  hovertemplate='%{x}<br>夏普: %{y:.2f}<extra></extra>'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=window_names, y=max_dds, name='最大回撤', 
                  marker_color=colors[3],
                  hovertemplate='%{x}<br>最大回撤: %{y:.2f}%<extra></extra>'),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(x=window_names, y=consistency, name='一致性分数', 
                  marker_color=colors[4],
                  hovertemplate='%{x}<br>一致性: %{y:.0f}<extra></extra>'),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Bar(x=window_names, y=composite_scores, name='综合评分', 
                  marker_color='gold',
                  hovertemplate='%{x}<br>综合评分: %{y:.2f}<extra></extra>'),
            row=3, col=2
        )
        
        fig.update_layout(
            title='🔬 不同优化窗口效果对比',
            height=1000,
            showlegend=False,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 找出最佳窗口
        best_idx = np.argmax(composite_scores)
        best_window = window_names[best_idx]
        best_score = composite_scores[best_idx]
        
        st.success(f"🎯 **最佳优化窗口**: {best_window} (综合评分: {best_score:.2f})")
        
        # 显示最佳窗口的详细统计
        st.subheader(f"📈 {best_window} 详细表现")
        best_stats = results[best_window]['stats']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("平均收益率", f"{best_stats['avg_return']:.2f}%")
        with col2:
            st.metric("正收益期占比", f"{best_stats['win_rate_periods']:.1f}%")
        with col3:
            st.metric("平均夏普", f"{best_stats['avg_sharpe']:.2f}")
        with col4:
            st.metric("最大回撤", f"{best_stats['max_drawdown']:.2f}%")
# ==========================================
# 新增：蒙特卡洛压力测试器 (修复变量名错误版)
# ==========================================
class MonteCarloAnalyzer:
    """蒙特卡洛模拟分析器：验证运气成分"""
    
    @staticmethod
    def run_simulation(trades: List[Trade], initial_capital: float, simulations: int = 1000):
        if len(trades) < 10:
            st.warning("交易笔数过少 (<10)，无法进行有效的蒙特卡洛模拟")
            return

        # 提取每笔交易的盈亏额 (PnL)
        pnl_sequence = [t.pnl for t in trades]
        
        results = []
        max_drawdowns = []
        final_capitals = []
        
        # 运行模拟
        with st.spinner(f"正在进行 {simulations} 次蒙特卡洛模拟..."):
            for _ in range(simulations):
                # 1. 随机打乱交易顺序 (Shuffle)
                shuffled_pnl = np.random.permutation(pnl_sequence)
                
                # 2. 计算资金曲线
                equity = [initial_capital]
                peak = initial_capital
                
                # 🔥 修正点：变量名定义为 max_dd
                max_dd = 0
                
                for pnl in shuffled_pnl:
                    current_cap = equity[-1] + pnl
                    equity.append(current_cap)
                    
                    # 计算回撤
                    if current_cap > peak:
                        peak = current_cap
                    
                    # 防止除以0错误
                    if peak > 0:
                        dd = (peak - current_cap) / peak
                    else:
                        dd = 0
                    
                    # 🔥 修正点：这里原来写成了 max_drawdown，导致报错
                    # 现在统一改为 max_dd
                    if dd > max_dd:
                        max_dd = dd
                
                final_capitals.append(equity[-1])
                max_drawdowns.append(max_dd)
                
                # 保存前100条曲线用于绘图（省内存）
                if len(results) < 100:
                    results.append(equity)

        # --- 统计分析 ---
        if not max_drawdowns:
            st.error("模拟失败，未能生成数据")
            return

        avg_dd = np.mean(max_drawdowns)
        worst_dd = np.percentile(max_drawdowns, 95) # 95%置信度下的最差回撤
        best_dd = np.min(max_drawdowns)
        
        # 破产概率
        bankruptcy_count = sum(1 for c in final_capitals if c <= 0)
        bankruptcy_rate = (bankruptcy_count / simulations) * 100
        
        st.subheader(f"🎲 蒙特卡洛模拟结果 ({simulations}次)")
        
        # 指标卡片
        col1, col2, col3 = st.columns(3)
        col1.metric("平均最大回撤", f"{avg_dd*100:.2f}%")
        col2.metric("95%置信度最差回撤", f"{worst_dd*100:.2f}%", help="只有5%的概率回撤会比这个更惨")
        col3.metric("破产概率 (归零)", f"{bankruptcy_rate:.1f}%", help="模拟中资金归零的概率")

        # 绘制“面条图” (Spaghetti Plot)
        fig = go.Figure()
        
        # 添加模拟曲线 (灰色，细线)
        for curve in results:
            fig.add_trace(go.Scatter(
                y=curve, 
                mode='lines', 
                line=dict(color='rgba(150, 150, 150, 0.1)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
            
        # 添加原始曲线 (红色，粗线)
        original_equity = [initial_capital]
        current = initial_capital
        for pnl in pnl_sequence:
            current += pnl
            original_equity.append(current)
            
        fig.add_trace(go.Scatter(
            y=original_equity,
            mode='lines',
            name='原始资金曲线',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title='蒙特卡洛路径模拟 (随机重排交易顺序)',
            xaxis_title='交易笔数',
            yaxis_title='资金 (U)',
            template='plotly_white',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# Streamlit可视化组件（修复增强版）
# ==========================================

class SmartMoneyVisualizer:
    """聪明钱可视化组件 (修复版：支持唯一Key)"""
    
    @staticmethod
    def create_equity_curve(stats: Dict, key_suffix: str = ""):
        """创建资金曲线图"""
        equity_curve = stats.get('equity_curve', [])
        if len(equity_curve) < 2:
            st.warning("资金曲线数据不足")
            return
        
        # 使用 stats.get() 安全访问 initial_capital
        initial_capital = stats.get('initial_capital', equity_curve[0] if equity_curve else 10000)
        
        fig = go.Figure()
        
        # 资金曲线
        fig.add_trace(go.Scatter(
            x=list(range(len(equity_curve))),
            y=equity_curve,
            mode='lines',
            name='资金曲线',
            line=dict(color='blue', width=2),
            hovertemplate='时间步: %{x}<br>资金: $%{y:,.2f} U<extra></extra>'
        ))
        
        # 初始资金线
        fig.add_trace(go.Scatter(
            x=[0, len(equity_curve)-1],
            y=[initial_capital, initial_capital],
            mode='lines',
            name='初始资金',
            line=dict(color='red', width=1, dash='dash'),
            hovertemplate=f'初始资金: ${initial_capital:,.0f} U<extra></extra>'
        ))
        
        fig.update_layout(
            title='资金曲线与回撤',
            xaxis_title='时间步数',
            yaxis_title='资金 (U)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        # 🔥【关键】这里接收了 key_suffix 参数
        st.plotly_chart(fig, use_container_width=True, key=f"equity_chart_{key_suffix}")
        
        # 关键指标卡片
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            final_capital = stats.get('final_capital', 0)
            total_return = stats.get('total_return', 0)
            st.metric("最终资金", f"${final_capital:,.0f} U", 
                     f"{total_return:+.2f}%")
        
        with col2:
            max_drawdown = stats.get('max_drawdown', 0)
            st.metric("最大回撤", f"{max_drawdown:.2f}%", 
                     "风险指标")
        
        with col3:
            annual_return = stats.get('annual_return', 0)
            sharpe = stats.get('sharpe', 0)
            st.metric("年化收益", f"{annual_return:.2f}%",
                     f"Sharpe: {sharpe:.2f}")
        
        with col4:
            total_trades = stats.get('total_trades', 0)
            win_rate = stats.get('win_rate', 0)
            st.metric("交易次数", f"{total_trades}",
                     f"胜率: {win_rate:.1f}%")
        
        with col5:
            profit_factor = stats.get('profit_factor', 0)
            calmar = stats.get('calmar', 0)
            st.metric("盈利因子", f"{profit_factor:.2f}",
                     f"Calmar: {calmar:.2f}")
    
    @staticmethod
    def create_trade_performance_chart(trades: List[Trade], key_suffix: str = ""):
        """创建交易表现图表"""
        if not trades:
            st.warning("没有交易数据")
            return
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('每笔交易盈亏', '累计盈亏曲线', '胜率分布',
                          '各代币表现', '交易持续时间', '策略类型对比'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}, {'type': 'pie'}],
                   [{'type': 'bar'}, {'type': 'histogram'}, {'type': 'bar'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. 每笔交易盈亏柱状图
        trade_ids = [f"#{t.id}" for t in trades]
        pnls = [t.pnl for t in trades]
        
        colors = []
        for trade in trades:
            if trade.smc_info:
                colors.append('purple' if trade.pnl > 0 else 'orange')
            else:
                colors.append('green' if trade.pnl > 0 else 'red')
        
        fig.add_trace(
            go.Bar(x=trade_ids, y=pnls, name='盈亏',
                  marker_color=colors, opacity=0.7,
                  hovertemplate='交易: %{x}<br>盈亏: $%{y:.2f} U<extra></extra>'),
            row=1, col=1
        )
        
        # 2. 累计盈亏曲线
        cumulative_pnl = np.cumsum(pnls)
        fig.add_trace(
            go.Scatter(x=trade_ids, y=cumulative_pnl,
                      mode='lines+markers', name='累计盈亏',
                      line=dict(color='green', width=2),
                      hovertemplate='交易: %{x}<br>累计盈亏: $%{y:.2f} U<extra></extra>'),
            row=1, col=2
        )
        
        # 3. 胜率饼图
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        fig.add_trace(
            go.Pie(labels=['盈利交易', '亏损交易'],
                  values=[len(winning_trades), len(losing_trades)],
                  marker_colors=['#4CAF50', '#F44336'],
                  hole=0.3,
                  hovertemplate='%{label}: %{value}笔 (%{percent})<extra></extra>'),
            row=1, col=3
        )
        
        # 4. 各代币表现
        symbol_data = {}
        for trade in trades:
            symbol = trade.symbol
            if symbol not in symbol_data:
                symbol_data[symbol] = {'pnl': 0, 'count': 0, 'wins': 0}
            symbol_data[symbol]['pnl'] += trade.pnl
            symbol_data[symbol]['count'] += 1
            if trade.pnl > 0:
                symbol_data[symbol]['wins'] += 1
        
        symbols = list(symbol_data.keys())
        symbol_pnls = [symbol_data[s]['pnl'] for s in symbols]
        symbol_counts = [symbol_data[s]['count'] for s in symbols]
        
        fig.add_trace(
            go.Bar(x=[s.replace('/USDT', '') for s in symbols],
                  y=symbol_pnls, name='总盈亏',
                  marker_color='skyblue',
                  hovertemplate='代币: %{x}<br>总盈亏: $%{y:.2f} U<br>交易次数: %{customdata}<extra></extra>',
                  customdata=symbol_counts),
            row=2, col=1
        )
        
        # 5. 交易持续时间分布
        durations = [t.duration_hours for t in trades if hasattr(t, 'duration_hours')]
        if durations:
            fig.add_trace(
                go.Histogram(x=durations, nbinsx=15,
                            name='持续时间',
                            marker_color='purple', opacity=0.7,
                            hovertemplate='持续时间: %{x:.1f}小时<br>交易次数: %{y}<extra></extra>'),
                row=2, col=2
            )
        
        # 6. 策略类型对比
        smc_trades = [t for t in trades if t.smc_info]
        regular_trades = [t for t in trades if not t.smc_info]
        
        if smc_trades or regular_trades:
            strategy_types = ['聪明钱策略', '常规策略']
            strategy_counts = [len(smc_trades), len(regular_trades)]
            strategy_wins = [
                sum(1 for t in smc_trades if t.pnl > 0),
                sum(1 for t in regular_trades if t.pnl > 0)
            ]
            
            fig.add_trace(
                go.Bar(x=strategy_types,
                      y=strategy_counts,
                      name='交易数量',
                      marker_color=['purple', 'blue'],
                      hovertemplate='策略: %{x}<br>交易次数: %{y}<br>盈利次数: %{customdata}<extra></extra>',
                      customdata=strategy_wins),
                row=2, col=3
            )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="交易表现分析",
            title_font_size=20,
            template='plotly_white'
        )
        
        # 🔥【关键】这里接收了 key_suffix 参数
        st.plotly_chart(fig, use_container_width=True, key=f"perf_chart_{key_suffix}")
    
    @staticmethod
    def create_smc_analysis(trades: List[Trade], key_suffix: str = ""):
        """创建聪明钱分析图表"""
        smc_trades = [t for t in trades if t.smc_info]
        if not smc_trades:
            return
        
        st.subheader("🧠 聪明钱信号分析")
        
        # FVG信号统计
        fvg_signals = {
            'bullish_1h': 0, 'bearish_1h': 0,
            'bullish_4h': 0, 'bearish_4h': 0
        }
        
        for trade in smc_trades:
            smc_info = trade.smc_info
            if 'fvg_direction_1h' in smc_info:
                if smc_info['fvg_direction_1h'] == 'bullish': fvg_signals['bullish_1h'] += 1
                elif smc_info['fvg_direction_1h'] == 'bearish': fvg_signals['bearish_1h'] += 1
            
            if 'fvg_direction_4h' in smc_info:
                if smc_info['fvg_direction_4h'] == 'bullish': fvg_signals['bullish_4h'] += 1
                elif smc_info['fvg_direction_4h'] == 'bearish': fvg_signals['bearish_4h'] += 1
        
        # 相对强弱统计
        rs_signals = {
            'strong_1h': 0, 'weak_1h': 0,
            'strong_4h': 0, 'weak_4h': 0
        }
        
        for trade in smc_trades:
            smc_info = trade.smc_info
            if 'rs_trend_1h' in smc_info:
                if smc_info['rs_trend_1h'] == 'strong': rs_signals['strong_1h'] += 1
                elif smc_info['rs_trend_1h'] == 'weak': rs_signals['weak_1h'] += 1
            
            if 'rs_trend_4h' in smc_info:
                if smc_info['rs_trend_4h'] == 'strong': rs_signals['strong_4h'] += 1
                elif smc_info['rs_trend_4h'] == 'weak': rs_signals['weak_4h'] += 1
        
        # 创建图表
        col1, col2 = st.columns(2)
        
        with col1:
            fig_fvg = go.Figure(data=[
                go.Bar(
                    x=['1H看涨FVG', '1H看跌FVG', '4H看涨FVG', '4H看跌FVG'],
                    y=[fvg_signals['bullish_1h'], fvg_signals['bearish_1h'], 
                       fvg_signals['bullish_4h'], fvg_signals['bearish_4h']],
                    marker_color=['green', 'red', 'lightgreen', 'lightcoral']
                )
            ])
            fig_fvg.update_layout(title='FVG信号分布', height=400, template='plotly_white')
            # 🔥【关键】这里接收了 key_suffix 参数
            st.plotly_chart(fig_fvg, use_container_width=True, key=f"fvg_chart_{key_suffix}")
        
        with col2:
            fig_rs = go.Figure(data=[
                go.Bar(
                    x=['1H强势', '1H弱势', '4H强势', '4H弱势'],
                    y=[rs_signals['strong_1h'], rs_signals['weak_1h'], 
                       rs_signals['strong_4h'], rs_signals['weak_4h']],
                    marker_color=['darkgreen', 'darkred', 'lightgreen', 'lightcoral']
                )
            ])
            fig_rs.update_layout(title='相对强弱信号分布', height=400, template='plotly_white')
            # 🔥【关键】这里接收了 key_suffix 参数
            st.plotly_chart(fig_rs, use_container_width=True, key=f"rs_chart_{key_suffix}")
        
        smc_win_rate = sum(1 for t in smc_trades if t.pnl > 0) / len(smc_trades) * 100 if smc_trades else 0
        st.info(f"**聪明钱交易胜率**: {smc_win_rate:.1f}% ({len(smc_trades)}笔交易)")
    
    @staticmethod
    def create_trade_details_table(trades: List[Trade]):
        """创建交易详情表格 (不需要Key，st.dataframe自动处理)"""
        if not trades:
            st.warning("没有交易数据")
            return
        
        trade_data = []
        for trade in trades:
            trade_type = "聪明钱" if trade.smc_info else "常规"
            direction_symbol = "📈" if trade.direction == TradeDirection.LONG else "📉"
            pnl_symbol = "💰" if trade.pnl > 0 else "💸"
            entry_time_str = trade.entry_time.strftime('%Y-%m-%d %H:%M')
            exit_time_str = trade.exit_time.strftime('%Y-%m-%d %H:%M') if trade.exit_time else "持仓中"
            
            trade_data.append({
                '交易ID': trade.id,
                '代币': trade.symbol.replace('/USDT', ''),
                '类型': f"{trade_type}",
                '方向': f'{direction_symbol} {trade.direction.value}',
                '入场时间': entry_time_str,
                '出场时间': exit_time_str,
                '入场价格': f"${trade.entry_price:.2f}",
                '出场价格': f"${trade.exit_price:.2f}" if trade.exit_price else "持仓中",
                '盈亏(U)': f"{pnl_symbol} ${trade.pnl:+.2f}",
                '盈亏(%)': f"{trade.pnl_percent:+.1f}%",
                '信号分数': trade.signal_score,
                '筛选分数': f"{trade.screening_score:.1f}" if trade.screening_score > 0 else "N/A",
                '排名': f"{trade.token_rank}" if trade.token_rank > 0 else "N/A",
                '持续时间': f"{trade.duration_hours:.1f}小时",
                '风险回报比': f"{trade.risk_reward_ratio:.2f}",
                '杠杆': f"{trade.leverage_used:.1f}倍",
                '出场原因': trade.exit_reason or "持仓中"
            })
        
        df_trades = pd.DataFrame(trade_data)
        st.dataframe(df_trades, use_container_width=True, height=500)
    
    @staticmethod
    def create_parameter_optimization_results(results: List[Dict], param_chinese_names: Dict, key_suffix: str = "default"):
        """创建参数优化结果 (最终版：核心优先 + 数值两位小数格式化)"""
        if not results:
            st.warning("没有参数优化结果")
            return
        
        st.subheader("🎯 参数优化结果")
        
       # === 1. 定义优先显示的参数 (🔥 核心VIP席位) ===
        priority_keys = [
            # --- 动态参数 (最重要，想看它有没有生效) ---
            'sideways_threshold', # 震荡门槛 (防御)
            'trend_threshold',    # 趋势门槛 (进攻)

             # --- 进阶参数 ---
            'sideways_rr',        # 震荡盈亏比
            'trend_rr',           # 趋势盈亏比
            
            # --- 基准参数 ---
            'min_signal_score',   # 基础分
            'min_rr_ratio',       # 基础盈亏比
            
           
            
            # --- 兼容旧版参数 (可选保留) ---
            'risk_reward_ratio' 
        ]
        
        # === 2. 辅助函数：数值格式化 (解决小数点过长问题) ===
        def format_value(v):
            if isinstance(v, float):
                return f"{v:.2f}"   # 强制保留2位小数
            elif isinstance(v, bool):
                return '是' if v else '否'
            return str(v)

        # === 3. 构建表格数据 ===
        result_data = []
        # 只显示前20名
        for i, result in enumerate(results[:20]):
            params = result.get('params', {})
            
            # --- 参数排序与拼接逻辑 ---
            priority_params = []
            other_params = []
            
            # A. 处理核心参数
            for key in priority_keys:
                if key in params:
                    chinese_name = param_chinese_names.get(key, key)
                    val_str = format_value(params[key]) 
                    # 加火苗图标强调
                    priority_params.append(f"🔥{chinese_name}: {val_str}")
            
            # B. 处理其他参数
            for key, value in params.items():
                if key not in priority_keys:
                    chinese_name = param_chinese_names.get(key, key)
                    val_str = format_value(value)
                    other_params.append(f"{chinese_name}: {val_str}")
            
            # C. 拼接：核心参数 || 其他参数
            if priority_params:
                full_params_str = "  ||  ".join(priority_params) + "  |  " + " | ".join(other_params)
            else:
                full_params_str = " | ".join(other_params)
            
            result_data.append({
                '排名': i + 1,
                '总收益率': f"{result.get('total_return', 0):.2f}%",
                '胜率': f"{result.get('win_rate', 0):.1f}%",
                '交易次数': result.get('total_trades', 0),
                '最大回撤': f"{result.get('max_drawdown', 0):.2f}%",
                '夏普比率': f"{result.get('sharpe', 0):.2f}",
                '最终资金': f"${result.get('final_capital', 0):,.0f}",
                '参数设置': full_params_str
            })
        
        df = pd.DataFrame(result_data)
        
        # === 4. 表格样式美化 ===
        def color_return(val):
            try:
                return_val = float(val.replace('%', ''))
                if return_val > 0: return 'color: green'
                elif return_val < 0: return 'color: red'
                else: return ''
            except: return ''
        
        styled_df = df.style.applymap(color_return, subset=['总收益率'])
        
       # === 显示表格 ===
        st.dataframe(
            styled_df, 
            use_container_width=True, 
            height=400,
            column_config={
                # 1. 【核心技巧】不仅设为 small，还把 label 改成两个字
                #    表头字数少了，列宽自然就缩进去了
                "排名": st.column_config.NumberColumn(
                    label="#",           # 把 "排名" 显示为 "#"
                    format="%d", 
                    width="small"
                ),
                "总收益率": st.column_config.TextColumn(
                    label="收益",        # 把 "总收益率" 显示为 "收益"
                    width="small"
                ),
                "胜率": st.column_config.TextColumn(
                    label="胜率",        # 保持不变，本身就很短
                    width="small"
                ),
                "交易次数": st.column_config.NumberColumn(
                    label="次数",        # 把 "交易次数" 显示为 "次数"
                    width="small"
                ),
                "最大回撤": st.column_config.TextColumn(
                    label="回撤",        # 把 "最大回撤" 显示为 "回撤"
                    width="small"
                ),
                "夏普比率": st.column_config.TextColumn(
                    label="夏普",        # 把 "夏普比率" 显示为 "夏普"
                    width="small"
                ),
                "最终资金": st.column_config.TextColumn(
                    label="资金",        # 把 "最终资金" 显示为 "资金"
                    width="small"
                ),
                
                # 2. 参数列：保持 large，现在前面的列让出了空间，它会更宽
                "参数设置": st.column_config.TextColumn(
                    label="核心参数 🔥 | 其他参数",
                    width="large", 
                    help="🔥表示核心影响参数，数值已保留2位小数"
                )
            }
        )
        
        # === 5. 绘制散点图 (风险 vs 收益) ===
        if results:
            fig = go.Figure()
            valid_results = [r for r in results if 'max_drawdown' in r and 'total_return' in r]
            
            if valid_results:
                # 所有点
                fig.add_trace(go.Scatter(
                    x=[r['max_drawdown'] for r in valid_results],
                    y=[r['total_return'] for r in valid_results],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=[r.get('sharpe', 0) for r in valid_results],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="夏普比率")
                    ),
                    text=[f"排名 #{i+1}" for i in range(len(valid_results))],
                    hovertemplate='排名: %{text}<br>收益: %{y:.2f}%<br>回撤: %{x:.2f}%<extra></extra>'
                ))
                
                # 标记最佳点 (第一名)
                best_r = valid_results[0]
                fig.add_trace(go.Scatter(
                    x=[best_r['max_drawdown']],
                    y=[best_r['total_return']],
                    mode='markers',
                    marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
                    name='最佳参数',
                    hoverinfo='skip'
                ))
                
                fig.update_layout(
                    title='参数分布图：收益率 vs 风险 (颜色代表夏普比率)',
                    xaxis_title='最大回撤 (%)',
                    yaxis_title='总收益率 (%)',
                    height=500,
                    template='plotly_white'
                )
                # 防止Key冲突
                st.plotly_chart(fig, use_container_width=True, key=f"opt_chart_scatter_{key_suffix}")

def show_backtest_history_tab():
    """显示回测历史记录选项卡"""
    st.header("📜 回测历史记录")
    
    if 'backtest_history' not in st.session_state or not st.session_state.backtest_history:
        st.info("暂无回测记录。请先运行一次回测。")
        return
        
    # 转换为 DataFrame 用于展示
    data_for_df = []
    # 倒序展示，最新的在前面
    for r in reversed(st.session_state.backtest_history):
        row = r.copy()
        
        # 处理嵌套的 params 字典，提取核心参数展示
        params = row.pop('params')
        # 构建一个紧凑的参数字符串
        core_params = (f"EMA:{params.get('ema_fast')}/{params.get('ema_slow')} "
                      f"| RR:{params.get('min_rr_ratio'):.1f} "
                      f"| Score:{params.get('min_signal_score')} "
                      f"| SMC:{'开' if params.get('use_smc_logic') else '关'}")
        
        row['核心参数'] = core_params
        
        # 格式化数值列
        row['总收益'] = f"{row.pop('total_return'):.2f}%"
        row['胜率'] = f"{row.pop('win_rate'):.1f}%"
        row['最大回撤'] = f"{row.pop('max_drawdown'):.2f}%"
        row['夏普'] = f"{row.pop('sharpe'):.2f}"
        row['盈亏比'] = f"{row.pop('profit_factor'):.2f}"
        
        # 重命名列以匹配展示需求
        row['回测时间'] = row.pop('timestamp')
        row['数据日期'] = row.pop('date_range')
        row['代币池'] = row.pop('symbols')
        row['最佳代币'] = row.pop('best_token')
        row['最差代币'] = row.pop('worst_token')
        
        data_for_df.append(row)
        
    df = pd.DataFrame(data_for_df)
    
    # 定义列顺序
    cols = ['回测时间', '总收益', '胜率', '最大回撤', '夏普', '最佳代币', '最差代币', '核心参数', '数据日期', '代币池']
    # 确保只包含存在的列
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    st.dataframe(df, use_container_width=True)
    
    # 导出功能
    col1, col2 = st.columns([1, 4])
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 导出历史记录为 CSV",
            csv,
            f"backtest_history_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            key='download-history-csv'
        )
    with col2:
        if st.button("🗑️ 清空历史记录", type="secondary"):
            st.session_state.backtest_history = []
            st.rerun()

# ==========================================
# 辅助函数 (请确保这段代码在 def main(): 之前)
# ==========================================

def fetch_data_task(symbol: str, timeframe: str, start_date: str, end_date: str, 
                   use_proxy: bool, proxy_config: Dict) -> Optional[Dict]:
    """获取数据的任务函数 (用于线程池并行执行)"""
    try:
        # 初始化交易所 (每个线程独立初始化，避免冲突)
        exchange_config = {
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
            'timeout': 30000
        }
        
        if use_proxy and proxy_config:
            exchange_config['proxies'] = proxy_config
        
        exchange = ccxt.binance(exchange_config)
        
        # 获取数据
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 确保结束时间不晚于当前时间
        if end_dt > datetime.now():
            end_dt = datetime.now()
            # 如果结束日期被修正，不需要更新传入的字符串，只用于时间戳计算
        
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        
        all_ohlcv = []
        since = start_ts
        
        # 简单的重试机制
        max_retries = 3
        
        while since < end_ts:
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    since = ohlcv[-1][0] + 1
                    success = True
                    time.sleep(0.1) # 避免频率限制
                    
                except Exception as e:
                    retry_count += 1
                    time.sleep(1)
            
            if not success:
                break
                
            # 防止无限循环 (数据量过大保护)
            if len(all_ohlcv) > 200000: 
                break
        
        if not all_ohlcv:
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'status': 'failed',
                'error': 'No data returned'
            }
        
        # 处理数据
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).set_index('time').sort_index()
        
        # 再次按日期过滤确保精确
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if df.empty:
             return {
                'symbol': symbol,
                'timeframe': timeframe,
                'status': 'failed',
                'error': 'Empty dataframe after filtering'
            }

        # 保存到缓存 (这一步需要实例化 DataManager)
        # 注意：多线程写入文件可能存在冲突，但在Streamlit这种简单场景下通常没事
        # 或者可以选择只返回数据，由主线程统一保存。这里为了逻辑简单直接保存。
        try:
            data_manager = DataManager()
            data_manager.save_data(symbol, timeframe, start_date, end_date, df)
        except Exception as e:
            print(f"缓存保存失败: {e}")

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'data': df,
            'status': 'success'
        }
    
    except Exception as e:
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'error': str(e),
            'status': 'failed'
        }

# ==========================================
# 运行主函数
# ==========================================

def main():
    render_trading_memo()
    
    st.set_page_config(
        page_title="领哥加密货币量化回测系统V24.0",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # 👇👇👇 【插入这段监控代码】 👇👇👇
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕵️‍♂️ 验证器内存监控")
    if 'global_validator' in st.session_state:
        val = st.session_state.global_validator
        
        # 检查手动数据
        has_ind = bool(val.independent and val.independent.get('config'))
        st.sidebar.markdown(f"**手动数据 (Tab3):** {'✅ 已存档' if has_ind else '❌ 空 (未运行)'}")
        
        # 检查滚动数据
        has_roll = bool(val.rolling and val.rolling.get('config'))
        st.sidebar.markdown(f"**滚动数据 (Tab7):** {'✅ 已存档' if has_roll else '❌ 空 (未运行)'}")
    else:
        st.sidebar.warning("验证器未初始化")
    st.sidebar.markdown("---")
    # 👆👆👆
    # 初始化会话状态
    if 'config' not in st.session_state:
        st.session_state.config = DEFAULT_CONFIG.copy()
    
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}
    
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = []
    
    if 'selected_symbol_kline' not in st.session_state:
        st.session_state.selected_symbol_kline = None
    
    if 'last_stats' not in st.session_state:
        st.session_state.last_stats = None
    
    if 'last_data_cache' not in st.session_state:
        st.session_state.last_data_cache = None
    
    if 'show_last_results' not in st.session_state:
        st.session_state.show_last_results = False
    
    if 'applied_optimization_params' not in st.session_state:
        st.session_state.applied_optimization_params = None
        
    if 'backtest_history' not in st.session_state:
        st.session_state.backtest_history = []
    
    # 应用标题
    st.title("🎯 领哥-量化回测系统V24.0映射2")
    st.markdown("""
    #### 多时间框架趋势跟踪 + 代币筛选策略 + 贝叶斯优化，# 
    
    """)
    
    # 侧边栏配置
    st.sidebar.title("⚙️ 配置设置")
    
    # 代理设置
    st.sidebar.subheader("🌐 网络代理设置")
    
    use_proxy = st.sidebar.checkbox("启用代理", value=True)
    
    if use_proxy:
        proxy_type = st.sidebar.selectbox(
            "代理类型",
            ["HTTP", "SOCKS5", "自定义"],
            index=0
        )
        
        if proxy_type == "HTTP":
            proxy_host = st.sidebar.text_input("代理主机", "127.0.0.1")
            proxy_port = st.sidebar.number_input("代理端口", 1080, 65535, 10808)
            proxy_config = {
                'http': f'http://{proxy_host}:{proxy_port}',
                'https': f'http://{proxy_host}:{proxy_port}'
            }
        elif proxy_type == "SOCKS5":
            proxy_host = st.sidebar.text_input("代理主机", "127.0.0.1")
            proxy_port = st.sidebar.number_input("代理端口", 1080, 65535, 10808)
            proxy_config = {
                'http': f'socks5://{proxy_host}:{proxy_port}',
                'https': f'socks5://{proxy_host}:{proxy_port}'
            }
        else:
            http_proxy = st.sidebar.text_input("HTTP代理", "http://127.0.0.1:10808")
            https_proxy = st.sidebar.text_input("HTTPS代理", "http://127.0.0.1:10808")
            proxy_config = {
                'http': http_proxy,
                'https': https_proxy
            }
    else:
        proxy_config = None
    
    

    # 创建选项卡
    # 修改这一行，增加一个 Tab
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 数据获取", "⚙️ 策略配置", "🔧 参数优化", 
        "🚀 回测执行", "📜 历史记录", "🎲 压力测试", "🔄 滚动回测", 
        "🕵️ 显微镜对比(Debug)"
    ])
        
    with tab1:
        st.header("📊 智能资产与数据管理 (Smart Asset Management)")
        
        # 1. 初始化智能管理器 (确保 SmartTokenManager 类已在前面定义)
        try:
            # 🔥 关键修改：传入全局代理配置
            # use_proxy 和 proxy_config 是侧边栏定义的全局变量
            current_proxies = proxy_config if use_proxy else None
            token_manager = SmartTokenManager(proxies=current_proxies)
        except NameError:
            st.error("❌ 错误：未找到 `SmartTokenManager` 类。请先将该类代码添加到 `DataManager` 类之前。")
            st.stop()

        col_select, col_time = st.columns([1.6, 1])
        
        # ========================================================
        # 👈 左侧：智能代币选择系统 (核心升级)
        # ========================================================
        with col_select:
            st.subheader("🎯 资产池选择 (Asset Selection)")
            
            # --- A. 选币模式切换 ---
            selection_mode = st.radio(
                "选择选币模式", 
                ["🔥 捕捉当下热点 (实时)", "📅 历史情景回放 (回测)", "✨ 手动自选"],
                horizontal=True,
                key="tab1_selection_mode"
            )
            
            # 初始化 session_state 中的代币池缓存，防止刷新丢失
            pool_cache_key = 'tab1_selected_pool_cache'
            if pool_cache_key not in st.session_state:
                # 默认初始值
                st.session_state[pool_cache_key] = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']

            # --- B. 根据模式显示不同控件 ---
            
            # [模式 1] 实时热点
            if "当下热点" in selection_mode:
                st.info("💡 策略逻辑：基于 Binance 24h 成交量(流动性)与波动率(机会)的加权算法，自动捕捉资金最关注的品种。")
                col_btn, col_spin = st.columns([1, 1.5])
                with col_btn:
                    if st.button("🚀 立即扫描全市场", type="primary", use_container_width=True):
                        with st.spinner("正在连接交易所接口，扫描量价数据..."):
                            # 调用 SmartTokenManager 获取热点
                            hot_tokens = token_manager.fetch_dynamic_hot_tokens(top_n=15)
                            st.session_state[pool_cache_key] = hot_tokens
                            st.success(f"✅ 已捕获 {len(hot_tokens)} 个热点资产！")
                            time.sleep(0.5)
                            st.rerun() # 强制刷新以更新下方多选框

            # [模式 2] 历史回放
            elif "历史情景" in selection_mode:
                st.info("💡 策略逻辑：加载特定年份的主流叙事代币，并自动锁定当年时间，还原真实市场环境。")
                col_year, col_load = st.columns([1, 1.5])
                with col_year:
                    hist_year = st.selectbox("选择回测年份", [2020, 2021, 2022, 2023, 2024, 2025], index=4)
                with col_load:
                    if st.button(f"📂 加载 {hist_year} 年核心资产", use_container_width=True):
                        # 1. 调用 SmartTokenManager 获取历史池
                        hist_tokens = token_manager.get_history_pool(hist_year)
                        st.session_state[pool_cache_key] = hist_tokens
                        
                        # 2. 🔥【新增功能】自动同步日期范围
                        # 设定为当年的 1月1日 到 12月31日
                        auto_start = datetime(hist_year, 1, 1).date()
                        auto_end = datetime(hist_year, 12, 31).date()
                        
                        # 如果是当前年份（如2025），结束日期不能超过今天
                        if hist_year == datetime.now().year:
                            auto_end = datetime.now().date()
                            
                        st.session_state['start_date_input'] = auto_start
                        st.session_state['end_date_input'] = auto_end
                        
                        st.success(f"✅ 已加载 {len(hist_tokens)} 个代币，并将时间锁定为 {hist_year} 全年")
                        time.sleep(0.5)
                        st.rerun()

            # [模式 3] 手动自选
            else:
                st.caption("请在下方直接选择或输入代币。")

            # --- C. 最终选择确认框 ---
            # 合并默认列表和缓存列表，确保下拉菜单里有这些选项
            default_options = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'DOGE/USDT', 'XRP/USDT', 'ADA/USDT', 'AVAX/USDT']
            all_options = list(set(st.session_state[pool_cache_key] + default_options))
            
            selected_symbols = st.multiselect(
                "✅ 确认交易列表 (支持手动增删)",
                options=sorted(all_options),
                default=st.session_state[pool_cache_key],
                
            )
            
            # 实时同步到全局配置
            if selected_symbols:
                st.session_state.config['symbols'] = selected_symbols
            
            # --- D. 代币板块透视 (Expander) ---
            if selected_symbols:
                with st.expander("🧐 查看代币板块与风险分布 (点击展开)", expanded=False):
                    # 使用 SmartTokenManager 进行分类
                    sectors = [token_manager.classify_token(s) for s in selected_symbols]
                    
                    # 1. 风险提示
                    meme_count = sectors.count('Meme')
                    if meme_count >= 3:
                        st.warning(f"⚠️ 高风险警告：组合中包含 {meme_count} 个 Meme 币！此类资产同涨同跌严重，请注意控制仓位。")
                    
                    # 2. 分布统计图
                    sec_counts = pd.Series(sectors).value_counts()
                    col_chart, col_data = st.columns([1, 1])
                    
                    with col_chart:
                        st.caption("板块权重分布:")
                        for sec, count in sec_counts.items():
                            ratio = count / len(selected_symbols)
                            st.progress(ratio, text=f"{sec}: {count}个 ({ratio:.1%})")
                    
                    with col_data:
                        st.caption("详细分类表:")
                        df_sec = pd.DataFrame({'代币': selected_symbols, '板块': sectors})
                        st.dataframe(df_sec, use_container_width=True, hide_index=True)

        # ========================================================
        # 👉 右侧：时间配置与执行
        # ========================================================
        with col_time:
            st.subheader("⏳ 时间与数据执行")
            
            # 1. 快速时间选择逻辑
            def set_quick_date(days):
                st.session_state['start_date_input'] = datetime.now().date() - timedelta(days=days)
                st.session_state['end_date_input'] = datetime.now().date()

            # 2. 快捷按钮组
            qc1, qc2, qc3, qc4 = st.columns(4)
            if qc1.button("1月", help="近30天", use_container_width=True): set_quick_date(30)
            if qc2.button("3月", help="近90天", use_container_width=True): set_quick_date(90)
            if qc3.button("半年", help="近180天", use_container_width=True): set_quick_date(180)
            if qc4.button("1年", help="近365天", use_container_width=True): set_quick_date(365)

            # 3. 日期选择器 (带 State 记忆)
            if 'start_date_input' not in st.session_state:
                st.session_state['start_date_input'] = datetime.now().date() - timedelta(days=180)
            if 'end_date_input' not in st.session_state:
                st.session_state['end_date_input'] = datetime.now().date()

            dc1, dc2 = st.columns(2)
            start_date_obj = dc1.date_input("开始日期", key="start_date_input")
            end_date_obj = dc2.date_input("结束日期", key="end_date_input")

            # 同步到 config
            st.session_state.config['start_date'] = start_date_obj.strftime('%Y-%m-%d')
            st.session_state.config['end_date'] = end_date_obj.strftime('%Y-%m-%d')
            
            # 4. K线周期选择
            timeframes = st.multiselect(
                "K线周期", ['15m', '30m', '1h', '4h', '1d'],
                default=st.session_state.config.get('timeframes', ['1h', '4h', '1d'])
            )
            st.session_state.config['timeframes'] = timeframes

            st.markdown("---")
            
            # 5. 执行按钮 (集成数据清洗)
            if st.button("📥 开始批量获取数据 (含智能清洗)", type="primary", use_container_width=True):
                if not selected_symbols or not timeframes:
                    st.error("❌ 请先选择【代币】和【周期】！")
                elif start_date_obj >= end_date_obj:
                    st.error("❌ 开始日期必须早于结束日期！")
                else:
                    # 初始化UI反馈
                    prog_bar = st.progress(0)
                    status_txt = st.empty()
                    
                    total_ops = len(selected_symbols) * len(timeframes)
                    done_ops = 0
                    
                    valid_results = []
                    rejected_results = [] # 记录被清洗掉的垃圾数据

                    # 开启线程池下载
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        futures = []
                        for s in selected_symbols:
                            for tf in timeframes:
                                # fetch_data_task 需在外部定义 (原代码已有)
                                futures.append(executor.submit(
                                    fetch_data_task, 
                                    s, tf, 
                                    start_date_obj.strftime('%Y-%m-%d'), 
                                    end_date_obj.strftime('%Y-%m-%d'),
                                    use_proxy, proxy_config
                                ))
                        
                        # 处理结果
                        for future in as_completed(futures):
                            res = future.result()
                            if res and res.get('status') == 'success':
                                # 🔥 核心改进：调用 SmartTokenManager 进行数据体检
                                # 只有体检通过的数据才会被存入缓存
                                is_ok, reason = token_manager.check_data_quality(res['data'], timeframe=res['timeframe'])
                                
                                if is_ok:
                                    valid_results.append(res)
                                    # 写入缓存
                                    if res['symbol'] not in st.session_state.data_cache:
                                        st.session_state.data_cache[res['symbol']] = {}
                                    st.session_state.data_cache[res['symbol']][res['timeframe']] = res['data']
                                else:
                                    # 记录不合格原因
                                    rejected_results.append(f"{res['symbol']}: {reason}")
                            
                            done_ops += 1
                            prog_bar.progress(done_ops / total_ops)
                            status_txt.text(f"🚀 数据获取中: {done_ops}/{total_ops} ...")
                    
                    prog_bar.empty()
                    status_txt.empty()
                    
                    # 最终报告
                    if valid_results:
                        st.success(f"✅ 成功获取 {len(valid_results)} 个有效数据片段！已存入缓存。")
                    
                    if rejected_results:
                        # 显示被剔除的数据，防止用户困惑为什么选了币却没数据
                        with st.expander(f"⚠️ 智能清洗系统已自动剔除 {len(rejected_results)} 个低质量数据", expanded=True):
                            st.write(rejected_results)
                            st.caption("🔍 剔除原因：数据长度不足(无法计算指标)、缺失值过多或长期无波动的僵尸币。")

        # ========================================================
        # 👇 底部：缓存管理工具
        # ========================================================
        st.markdown("---")
        bc1, bc2, bc3 = st.columns(3)
        
        with bc1:
            if st.button("🗑️ 清空所有缓存"):
                DataManager().clear_cache()
                st.session_state.data_cache = {}
                st.success("缓存已全部释放")
                time.sleep(1)
                st.rerun()
                
        with bc2:
            cache_count = sum(len(v) for v in st.session_state.data_cache.values())
            st.metric("已缓存数据片段", f"{cache_count} 个")
            
        with bc3:
            st.metric("当前代币池数量", len(st.session_state.config.get('symbols', [])))
    
    with tab2:
        st.header("⚙️ 策略参数配置")
        
        # 显示当前已应用的参数信息（如果有）
        if st.session_state.applied_optimization_params is not None:
            st.success("✅ 已应用优化后的参数！")
            st.info(f"已应用 {len(st.session_state.applied_optimization_params)} 个优化参数")
            
            if st.button("🔄 重置为默认参数", type="secondary"):
                st.session_state.config = DEFAULT_CONFIG.copy()
                st.session_state.applied_optimization_params = None
                st.success("已重置为默认参数")
                st.rerun()
        

        # 🔥 新增：动态参数配置
        st.subheader("🔄 动态参数调整配置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_dynamic_params = st.checkbox(
                "启用动态参数调整",
                value=st.session_state.config.get('enable_dynamic_params', True),
                help="根据市场状态自动调整交易参数"
            )
            
            st.markdown("**震荡市参数（防御模式）**")
            sideways_threshold = st.slider(
                "震荡市信号分门槛", 
                70, 85, 
                st.session_state.config.get('sideways_threshold', 75),
                step=5,
                help="震荡市要求更高的信号质量"
            )
            
            sideways_rr = st.slider(
                "震荡市最小盈亏比", 
                2.5, 4.0, 
                st.session_state.config.get('sideways_rr', 3.0),
                step=0.1,
                help="震荡市要求更高的盈亏比补偿"
            )
        
        with col2:
            st.markdown("**趋势市参数（进攻模式）**")
            trend_threshold = st.slider(
                "趋势市信号分门槛", 
                55, 70, 
                st.session_state.config.get('trend_threshold', 65),
                step=5,
                help="趋势市可以放宽信号质量要求"
            )
            
            trend_rr = st.slider(
                "趋势市最小盈亏比", 
                1.8, 2.5, 
                st.session_state.config.get('trend_rr', 2.0),
                step=0.1,
                help="趋势市可以降低盈亏比要求"
            )
        
        # 保存时添加到配置
        new_config = {
            # ... 原有配置 ...
            'enable_dynamic_params': enable_dynamic_params,
            'sideways_threshold': sideways_threshold,
            'sideways_rr': sideways_rr,
            'trend_threshold': trend_threshold,
            'trend_rr': trend_rr,
        }

        # 配置参数输入
        col1, col2 = st.columns(2)
        
        with col1:
            # 资金与基本设置
            st.subheader("💰 资金与基本设置")
            
            initial_capital = st.number_input(
                "初始本金 (U)",
                min_value=100, max_value=100000, 
                value=st.session_state.config.get('initial_capital', 10000), 
                step=1000,
                key="initial_capital_input"
            )
            
            # 注意：target_position_value 在后面根据模式动态显示
            
            fee_rate = st.number_input(
                "手续费率",
                min_value=0.0001, max_value=0.01,
                value=st.session_state.config.get('fee_rate', 0.0004),
                step=0.0001,
                format="%.4f",
                key="fee_rate_input"
            )
            
            slippage = st.number_input(
                "滑点",
                min_value=0.0001, max_value=0.01,
                value=st.session_state.config.get('slippage', 0.0010),
                step=0.0001,
                format="%.4f",
                key="slippage_input"
            )
            
            max_positions = st.slider(
                "最大同时持仓数",
                1, 5, 
                st.session_state.config.get('max_positions', 1),
                key="max_positions_slider"
            )
            
            check_interval_hours = st.slider(
                "检查间隔 (小时)",
                1, 24, 
                st.session_state.config.get('check_interval_hours', 1),
                key="check_interval_hours_slider"
            )
            
            st.markdown("---")
            
            # 仓位模式与风控性格选择
            st.subheader("💰 仓位与风控管理 (V3 引擎)")

            # 使用边框容器包裹，视觉上更紧凑
            with st.container(border=True):
                
                # 第一行：左边选模式，右边选性格 (利用横向空间)
                c1, c2 = st.columns([1, 1.5], gap="large")
                
                with c1:
                    st.markdown("###### 1. 仓位模式")
                    position_mode = st.radio(
                        "模式选择", 
                        options=['fixed', 'compounding'],
                        format_func=lambda x: '💰 固定仓位' if x == 'fixed' else '🚀 复合增长',
                        index=1 if st.session_state.config.get('position_mode') == 'compounding' else 0,
                        label_visibility="collapsed", 
                        key="pos_mode_radio",
                        help="固定仓位：每单固定金额。\n复合增长：根据余额比例开仓，利滚利。"
                    )
                    
                with c2:
                    st.markdown("###### 2. 鳄鱼风控性格")
                    risk_profile = st.select_slider(
                        "风控性格",
                        options=['Conservative', 'Balanced', 'Aggressive'],
                        value=st.session_state.config.get('risk_preference', 'Balanced'),
                        format_func=lambda x: {
                            'Conservative': '🛡️ 保守 (1x-3x)',
                            'Balanced': '⚖️ 平衡 (2x-5x)',
                            'Aggressive': '🚀 激进 (3x-10x)'
                        }[x],
                        label_visibility="collapsed",
                        key="risk_profile_slider"
                    )
                    # 动态说明文案
                    if risk_profile == 'Aggressive':
                        st.caption("🔥 **激进**: 基础3x/最高10x | 适合博翻身，注意高波动降权")
                    elif risk_profile == 'Conservative':
                        st.caption("🛡️ **保守**: 基础1x/最高3x | 适合大资金理财，极难爆仓")
                    else:
                        st.caption("⚖️ **平衡**: 基础2x/最高5x | 兼顾防守与进攻 (推荐)")

                st.divider() # 分割线

                # 第二行：具体的数值滑块 (并列显示)
                c3, c4 = st.columns(2)
                
                with c3:
                    if position_mode == 'compounding':
                        compounding_ratio = st.slider(
                            "资金投入比例", 0.1, 1.0, 
                            st.session_state.config.get('compounding_ratio', 0.5), 
                            0.1, format="%.1f",
                            key="compounding_ratio_input_v3"
                        )
                        # 实时计算示例
                        init_cap = st.session_state.config.get('initial_capital', 10000)
                        example_val = init_cap * compounding_ratio
                        st.caption(f"📝 示例: {init_cap}本金 × {compounding_ratio} = 投入 ${example_val:.0f}")
                        
                        # 这是一个为了保持变量名一致的dummy赋值
                        target_position_value = 30000 
                    else:
                        target_position_value = st.number_input(
                            "单仓价值 (U)", 1000, 1000000, 
                            st.session_state.config.get('target_position_value', 30000), 
                            1000,
                            key="target_pos_val_input_v3"
                        )
                        st.caption("📝 无论本金多少，每单固定开这个金额")
                        compounding_ratio = 1.0 # dummy

                with c4:
                    # 手动杠杆现在作为 "基础参考值"
                    leverage = st.slider(
                        "基础杠杆倍数", 1.0, 10.0, 
                        st.session_state.config.get('leverage', 3.0), 
                        0.5, format="%.1f倍",
                        help="这是鳄鱼策略计算的基准杠杆，实际杠杆会根据行情自动浮动。",
                        key="base_leverage_input_v3"
                    )
                    st.caption(f"🤖 AI 将在此基础上根据趋势自动浮动")

            # 保存配置逻辑 (必须更新到 config 字典中)
            st.session_state.config['risk_preference'] = risk_profile
            st.session_state.config['position_mode'] = position_mode
            st.session_state.config['compounding_ratio'] = compounding_ratio
            st.session_state.config['target_position_value'] = target_position_value
            st.session_state.config['leverage'] = leverage

            # 显示模式说明
            if position_mode == 'fixed':
                st.success("**固定仓位模式**：每笔交易使用固定保证金，风险稳定，适合保守策略")
            else:
                st.success("**复合增长模式**：根据当前资金动态调整仓位，盈利后仓位自动放大，适合进取策略")
        
        with col2:
            # 趋势指标
            st.subheader("📈 趋势指标")
            
            ema_fast = st.slider(
                "EMA快线周期", 
                3, 21, 
                st.session_state.config.get('ema_fast', 9),
                key="ema_fast_slider"
            )
            
            ema_medium = st.slider(
                "EMA中线周期",
                10, 50,
                st.session_state.config.get('ema_medium', 21),
                key="ema_medium_slider"
            )
            
            ema_slow = st.slider(
                "EMA慢线周期", 
                30, 100, 
                st.session_state.config.get('ema_slow', 50),
                key="ema_slow_slider"
            )
            
            ema_trend = st.slider(
                "EMA趋势线周期", 
                100, 300, 
                st.session_state.config.get('ema_trend', 200),
                key="ema_trend_slider"
            )
            
            adx_period = st.slider(
                "ADX周期", 
                10, 30, 
                st.session_state.config.get('adx_period', 14),
                key="adx_period_slider"
            )
        
        # 入场过滤与辅助指标
        st.subheader("🎯 入场过滤与辅助指标")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rsi_period = st.slider(
                "RSI周期", 
                7, 30, 
                st.session_state.config.get('rsi_period', 14),
                key="rsi_period_slider"
            )
            volume_ma = st.slider(
                "成交量均线周期", 
                10, 50, 
                st.session_state.config.get('volume_ma', 20),
                key="volume_ma_slider"
            )
            bb_period = st.slider(
                "布林带周期", 
                10, 30, 
                st.session_state.config.get('bb_period', 20),
                key="bb_period_slider"
            )
            
        with col2:
            atr_period = st.slider(
                "ATR周期", 
                7, 30, 
                st.session_state.config.get('atr_period', 14),
                key="atr_period_slider"
            )
            max_volatility = st.slider(
                "最大波动率限制", 
                0.01, 0.10, 
                st.session_state.config.get('max_volatility', 0.04), 
                0.01,
                key="max_volatility_slider"
            )
            bb_std = st.slider(
                "布林带宽度(Std)", 
                1.5, 3.5, 
                st.session_state.config.get('bb_std', 2.0), 
                0.1,
                key="bb_std_slider"
            )
            
        with col3:
            min_signal_score = st.slider(
                "最小信号得分", 
                50, 90, 
                st.session_state.config.get('min_signal_score', 70),
                key="min_signal_score_slider"
            )
            min_rr_ratio = st.slider(
                "最小盈亏比", 
                1.5, 5.0, 
                st.session_state.config.get('min_rr_ratio', 2.5), 
                0.1,
                key="min_rr_ratio_slider"
            )
            min_adx = st.slider(
                "最小ADX值", 
                10, 50, 
                st.session_state.config.get('min_adx', 25),
                key="min_adx_slider"
            )

        # 高级策略与筛选
        st.subheader("🧠 高级策略与筛选")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**功能开关**")
            use_smc_logic = st.checkbox(
                "启用聪明钱逻辑",
                value=st.session_state.config.get('use_smc_logic', False),
                key="use_smc_logic_checkbox"
            )
            use_dynamic_risk = st.checkbox(
                "启用动态风控",
                value=st.session_state.config.get('use_dynamic_risk', False),
                key="use_dynamic_risk_checkbox"
            )
            enable_token_screening = st.checkbox(
                "启用代币筛选",
                value=st.session_state.config.get('enable_token_screening', True),
                key="enable_token_screening_checkbox"
            )
            select_best_token_only = st.checkbox(
                "只做最佳代币",
                value=st.session_state.config.get('select_best_token_only', True),
                key="select_best_token_only_checkbox"
            )

        with col2:
            st.markdown("**SMC参数 (已解锁精确调节)**")
            
            # FVG 回溯：范围 1-10，步长 1
            fvg_lookback = st.slider(
                "FVG回溯周期", 
                1, 10, 
                st.session_state.config.get('fvg_lookback', 3),
                step=1,  # <--- ✅ 改为1，支持 1,2,3...
                key="fvg_lookback_slider"
            )
            
            # 相对强弱：范围 5-50，步长 1 (原为5)
            rs_period = st.slider(
                "相对强弱周期", 
                5, 50, 
                st.session_state.config.get('rs_period', 20), 
                step=1,  # <--- ✅ 改为1，现在可以选 13, 22 等精确值了
                key="rs_period_slider"
            )
            
            # 波段回溯：范围 5-30，步长 1 (原为5)
            swing_lookback = st.slider(
                "波段回溯周期", 
                5, 30, 
                st.session_state.config.get('swing_lookback', 10), 
                step=1,  # <--- ✅ 改为1，完美复现贝叶斯参数(如13)
                key="swing_lookback_slider"
            )
            
            # 筛选门槛：范围 50-90，步长 1
            min_signal_threshold = st.slider(
                "筛选入围分", 
                50, 90, 
                st.session_state.config.get('min_signal_threshold', 80),
                step=1,  # <--- ✅ 改为1
                key="min_signal_threshold_slider"
            )

        # 筛选权重
        st.subheader("⚖️ 筛选权重配置")
        st.caption("权重总和必须等于 1.0")
        
        # 获取当前权重
        weights = st.session_state.config.get('screening_weights', DEFAULT_CONFIG['screening_weights'])
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            signal_weight = st.slider(
                "信号分数权重", 
                0.0, 1.0, 
                weights.get('signal_score', 0.40), 
                0.05,
                key="signal_weight_slider"
            )
        with col2:
            trend_weight = st.slider(
                "趋势强度权重", 
                0.0, 1.0, 
                weights.get('trend_strength', 0.25), 
                0.05,
                key="trend_weight_slider"
            )
        with col3:
            momentum_weight = st.slider(
                "动量权重", 
                0.0, 1.0, 
                weights.get('momentum', 0.15), 
                0.05,
                key="momentum_weight_slider"
            )
        with col4:
            risk_weight = st.slider(
                "盈亏比权重", 
                0.0, 1.0, 
                weights.get('risk_reward', 0.10), 
                0.05,
                key="risk_weight_slider"
            )
        with col5:
            vol_weight = st.slider(
                "成交量权重", 
                0.0, 1.0, 
                weights.get('volume_confirmation', 0.10), 
                0.05,
                key="vol_weight_slider"
            )
        # ==============================================================================
        # 🛡️ [新增] V24.10 高级风控挂载 (默认关闭，不影响原核心)
        # ==============================================================================
        st.markdown("---")
        st.subheader("🛡️ V24.10 高级风控挂载 (实验性)")
        st.caption("说明：以下功能默认关闭。关闭状态下，回测逻辑与 V24.5 原版完全一致。")
        
        with st.container(border=True):
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                st.markdown("##### 🔌 熔断机制")
                ui_enable_melt = st.checkbox(
                    "开启 ADX 过热熔断",
                    value=st.session_state.config.get('enable_adx_meltdown', False), # 默认 False (关)
                    help="当 ADX 超过阈值时强制停止开仓。关闭此开关则完全回退到 V24.5 逻辑。"
                )
                ui_melt_limit = st.number_input(
                    "熔断阈值", 
                    min_value=50, max_value=90, 
                    value=st.session_state.config.get('adx_meltdown_threshold', 60),
                    disabled=not ui_enable_melt
                )

            with col_adv2:
                st.markdown("##### 🌊 趋势共振")
                ui_enable_4h = st.checkbox(
                    "开启 4H 趋势共振",
                    value=st.session_state.config.get('enable_4h_resonance', False), # 默认 False (关)
                    help="强制要求 1H 信号与 4H EMA 趋势一致。关闭此开关则只看 1H (V24.5原版逻辑)。"
                )
                
            # 动态选币 (可选，如果不想要可以注释掉)
            st.divider()
            ui_enable_dynamic = st.checkbox(
                "🌟 开启星探动态选币 (Dynamic Watchlist)",
                value=st.session_state.config.get('enable_dynamic_scan', False), # 默认 False
                help="从数据池中动态选择波动率最高的币种。"
            )
        
        
        
        # 保存按钮
        st.markdown("---")
        if st.button("💾 保存策略配置", type="primary", use_container_width=True, key="save_config_button"):
            # 验证权重总和
            total_weight = (signal_weight + trend_weight + momentum_weight + risk_weight + vol_weight)
            
            if abs(total_weight - 1.0) > 0.001:
                st.error(f"权重总和为 {total_weight:.3f}，必须等于 1.0！")
            else:
                # 构建新配置
                new_config = {
                    'symbols': st.session_state.config.get('symbols', []),
                    'start_date': st.session_state.config.get('start_date'),
                    'end_date': st.session_state.config.get('end_date'),
                    'initial_capital': initial_capital,
                    'target_position_value': target_position_value if position_mode == 'fixed' else 30000,
                    'fee_rate': fee_rate,
                    'slippage': slippage,
                    'max_positions': max_positions,
                    'check_interval_hours': check_interval_hours,
                    
                    'enable_adx_meltdown': ui_enable_melt,
                    'adx_meltdown_threshold': ui_melt_limit,
                    'enable_4h_resonance': ui_enable_4h,
                    'enable_dynamic_scan': ui_enable_dynamic,

                    'max_portfolio_risk': st.session_state.config.get('max_portfolio_risk', 0.1),
                    'margin_maintenance': st.session_state.config.get('margin_maintenance', 0.5),
                    'min_liquidity': st.session_state.config.get('min_liquidity', 1000000),
                    
                    'enable_token_screening': enable_token_screening,
                    'select_best_token_only': select_best_token_only,
                    'min_signal_threshold': min_signal_threshold,
                    'screening_weights': {
                        'signal_score': signal_weight,
                        'trend_strength': trend_weight,
                        'momentum': momentum_weight,
                        'risk_reward': risk_weight,
                        'volume_confirmation': vol_weight,
                    },
                    
                    # 核心指标参数
                    'ema_fast': ema_fast,
                    'ema_medium': ema_medium,
                    'ema_slow': ema_slow,
                    'ema_trend': ema_trend,
                    'rsi_period': rsi_period,
                    'atr_period': atr_period,
                    'volume_ma': volume_ma,
                    'bb_period': bb_period,
                    'bb_std': bb_std,
                    'adx_period': adx_period,
                    
                    # 交易门槛
                    'min_rr_ratio': min_rr_ratio,
                    'max_volatility': max_volatility,
                    'min_signal_score': min_signal_score,
                    'min_adx': min_adx,
                    
                    # 仓位管理参数
                    'leverage': leverage,
                    'position_mode': position_mode,
                    'compounding_ratio': compounding_ratio if position_mode == 'compounding' else 1.0,
                    'target_position_value': target_position_value if position_mode == 'fixed' else 30000,

                    # 功能开关
                    'use_smc_logic': use_smc_logic,
                    'use_dynamic_risk': use_dynamic_risk,
                    'fvg_lookback': fvg_lookback,
                    'rs_period': rs_period,
                    'swing_lookback': swing_lookback,
                    
                    'timeframes': st.session_state.config.get('timeframes', ['15m', '1h', '4h', '1d'])
                }
                
                # 保存配置
                st.session_state.config = new_config
                st.session_state.applied_optimization_params = None
                st.success("✅ 策略配置已保存！")
    
    with tab3:
        st.header("🔧 参数优化")
    
        # 检查是否有数据
        if not st.session_state.data_cache:
            st.warning("请先在数据获取选项卡中获取数据！")
        else:
            # ==========================================
            # 🧪 窗口优化对比测试（放在优化方法选择之前）
            # ==========================================
            st.subheader("🧪 窗口优化对比测试")
            
            col_window1, col_window2, col_window3 = st.columns(3)
            
            with col_window1:
                if st.button("🔬 快速窗口对比测试", type="primary", 
                            help="只测试2种窗口（3个月和6个月），快速了解趋势",
                            use_container_width=True):
                    
                    with st.spinner("正在运行快速窗口对比测试（约15-30分钟）..."):
                        # 初始化优化器
                        optimizer = AdvancedParameterOptimizer()
                        
                        # 只测试关键窗口
                        windows_to_test = [
                            {'name': '3个月训练+1个月测试', 'train_months': 3, 'test_weeks': 4},
                            {'name': '6个月训练+1个月测试', 'train_months': 6, 'test_weeks': 4},
                        ]
                        
                        # 创建临时函数
                        quick_results = {}
                        for window in windows_to_test:
                            st.write(f"正在测试: {window['name']}")
                            
                            # 运行滚动优化
                            window_results = optimizer.rolling_window_optimization_v2(
                                config=st.session_state.config,
                                data_cache=st.session_state.data_cache,
                                train_months=window['train_months'],
                                test_weeks=window['test_weeks']
                            )
                            
                            if window_results:
                                stats = optimizer.analyze_window_results(window_results)
                                quick_results[window['name']] = {
                                    'stats': stats,
                                    'raw_results': window_results
                                }
                        
                        if quick_results:
                            optimizer.plot_window_comparison(quick_results)
                            st.success("✅ 快速窗口对比测试完成！")
                        else:
                            st.error("快速测试失败，没有获得有效结果")
            
            with col_window2:
                if st.button("🔬 完整窗口对比测试", type="secondary",
                            help="测试5种窗口组合，获得全面分析（约1-2小时）",
                            use_container_width=True):
                    
                    with st.spinner("正在运行完整窗口对比测试（可能需要1-2小时）..."):
                        # 初始化优化器
                        optimizer = AdvancedParameterOptimizer()
                        
                        # 运行完整的对比测试
                        results = optimizer.compare_optimization_windows(
                            config=st.session_state.config,
                            data_cache=st.session_state.data_cache
                        )
                        
                        if results:
                            st.success("✅ 窗口对比测试完成！")
                            
                            # 显示最佳窗口
                            best_window_name = None
                            best_score = -999
                            for window_name, data in results.items():
                                stats = data['stats']
                                score = (
                                    stats['avg_return'] * 0.3 +
                                    stats['win_rate_periods'] * 0.2 +
                                    stats['avg_sharpe'] * 0.3 -
                                    stats['max_drawdown'] * 0.2 +
                                    stats['consistency_score'] * 0.1
                                )
                                if score > best_score:
                                    best_score = score
                                    best_window_name = window_name
                            
                            if best_window_name:
                                st.balloons()
                                st.markdown(f"""
                                ## 🎯 **推荐使用窗口: {best_window_name}**
                                
                                **为什么选择这个窗口:**
                                - 综合评分最高: {best_score:.2f}
                                - 更多详细分析请查看上方图表
                                """)
        
            with col_window3:
                if st.button("📊 查看上次对比结果", type="secondary",
                            help="查看上次运行的窗口对比测试结果",
                            use_container_width=True):
                    if hasattr(st.session_state, 'last_window_results') and st.session_state.last_window_results:
                        optimizer = AdvancedParameterOptimizer()
                        optimizer.plot_window_comparison(st.session_state.last_window_results)
                    else:
                        st.warning("没有找到上次的窗口对比结果，请先运行测试")
            
            st.markdown("---")
        
        # ==========================================
        # 🛠️ 优化方法选择（原有功能保持不变）
        # ==========================================
        st.subheader("🛠️ 优化方法选择")
        
        optimization_method = st.selectbox(
            "选择优化方法",
            options=['贝叶斯优化', '遗传算法'],
            index=0
        )
        
        # 优化参数配置
        st.subheader("📊 优化参数配置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if optimization_method == '贝叶斯优化':
                n_trials = st.slider("贝叶斯优化试验次数", 20, 400, 50,
                                    help="试验次数越多，找到最优参数的概率越高，但耗时也更长")
            else:  # 遗传算法
                population_size = st.slider("种群大小", 20, 100, 30)
                generations = st.slider("进化代数", 10, 50, 15)
        
        with col2:
            param_range_option = st.selectbox(
                "参数范围",
                options=['默认范围', '自定义范围'],
                index=0
            )
        
        # 自定义参数范围
        if param_range_option == '自定义范围':
            st.subheader("🎯 自定义参数范围")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ema_fast_range = st.slider("EMA快线周期范围", 3, 21, (5, 12))
                ema_slow_range = st.slider("EMA慢线周期范围", 30, 100, (50, 70))
            
            with col2:
                rsi_period_range = st.slider("RSI周期范围", 7, 30, (10, 20))
                min_rr_ratio_range = st.slider("最小风险回报比范围", 1.5, 5.0, (2.0, 3.0))
            
            with col3:
                min_signal_score_range = st.slider("最小信号分数范围", 50, 90, (60, 80))
                min_adx_range = st.slider("最小ADX范围", 10, 50, (20, 30))
        
        st.markdown("---")
        st.subheader("🧐 运行前环境检查 (关键参数)")
        
        # 1. 准备数据
        curr_cfg = st.session_state.config
        
        # 资金模式逻辑
        is_fixed = curr_cfg.get('position_mode') == 'fixed'
        mode_label = "固定仓位" if is_fixed else "复合增长"
        mode_icon = "💰" if is_fixed else "🚀"
        if is_fixed:
            pos_main = f"${curr_cfg.get('target_position_value', 0):,.0f}"
            pos_sub = "单仓价值"
        else:
            pos_main = f"比例 {curr_cfg.get('compounding_ratio', 0):.1f}"
            pos_sub = "复利 (1.0=全仓)"

        # 日期计算
        try:
            s_date = datetime.strptime(curr_cfg.get('start_date'), '%Y-%m-%d')
            e_date = datetime.strptime(curr_cfg.get('end_date'), '%Y-%m-%d')
            total_days = (e_date - s_date).days
            days_display = f"{total_days} 天"
        except:
            days_display = "N/A"

        # === 第一行：资金与风控 (Money & Risk) ===
        st.caption("💰 **资金设定**")
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4, gap="large")
        
        with r1_c1:
            st.metric("资金模式", f"{mode_icon} {mode_label}", pos_sub)
        with r1_c2:
            st.metric("杠杆倍数", f"⚡ {curr_cfg.get('leverage')}x", f"本金: ${curr_cfg.get('initial_capital'):,.0f}")
        with r1_c3:
            st.metric("仓位规模", pos_main, "核心风控")
        with r1_c4:
            # 预留位置，或者放其他参数
            st.metric("手续费率", f"{curr_cfg.get('fee_rate')*100:.2f}%")

        # === 第二行：时间与周期 (Time & Period) ===
        st.write("") #以此增加一点垂直间距
        st.caption("📅 **时间范围**")
        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4, gap="large")
        
        with r2_c1:
            st.metric("检查周期", f"⏰ {curr_cfg.get('check_interval_hours')} 小时", "K线频率")
        with r2_c2:
            st.metric("开始日期", curr_cfg.get('start_date'), "Start")
        with r2_c3:
            st.metric("结束日期", curr_cfg.get('end_date'), "End")
        with r2_c4:
            st.metric("数据跨度", days_display, "Total Days")

        # 醒目的警告条
        st.markdown(
            f"""
            <div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px; border: 1px solid #90caf9; color: #0d47a1; text-align: center; margin-top: 10px; margin-bottom: 20px;">
                🔎 <strong>请确认：</strong> 您正在使用 <strong>{curr_cfg.get('check_interval_hours')}小时级别</strong> 的数据，
                对 <strong>{curr_cfg.get('start_date')}</strong> 至 <strong>{curr_cfg.get('end_date')}</strong> 期间进行优化。
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        col_run1, col_run2, col_run3 = st.columns(3)
        
        with col_run1:
            run_optimization = st.button("🚀 开始参数优化", type="primary",
                                        use_container_width=True)
        
        with col_run2:
            quick_optimization = st.button("⚡ 快速参数优化",
                                          help="使用默认设置快速优化参数（20次试验）",
                                          use_container_width=True)
        
        with col_run3:
            if st.button("📋 查看上次优化结果1", type="secondary",
                        use_container_width=True):
                if 'optimization_results' in st.session_state and st.session_state.optimization_results:
                    results = st.session_state.optimization_results
                    SmartMoneyVisualizer.create_parameter_optimization_results(
                        results, PARAM_CHINESE_NAMES
                    )
                else:
                    st.warning("没有找到上次优化结果！")
        
        # 执行优化
        # 执行优化
        if run_optimization or quick_optimization:
            # ==========================================
            # 🔥🔥🔥 【核心修复】手动模式也必须物理切片 🔥🔥🔥
            # ==========================================
            st.warning("⚠️ [验证模式] 正在进行严格的数据切片，确保不读取未来数据...")

            # 1. 准备切片容器
            sliced_cache = {}
            # 获取配置中的结束日期，并设为当天最后一秒
            end_date_str = st.session_state.config['end_date']
            cut_off_time = pd.to_datetime(end_date_str) + timedelta(hours=23, minutes=59)
            
            # 2. 执行物理切割
            for sym, tfs in st.session_state.data_cache.items():
                sliced_cache[sym] = {}
                for tf, df in tfs.items():
                    if not df.empty:
                        # 只保留 <= 结束日期的数据
                        sliced_df = df[df.index <= cut_off_time].copy()
                        sliced_cache[sym][tf] = sliced_df
            
            # 3. 初始化优化器
            optimizer = AdvancedParameterOptimizer()
            
            # 确定方法
            if optimization_method == '贝叶斯优化':
                method = 'bayesian'
                kwargs = {'n_trials': 20} if quick_optimization else {'n_trials': n_trials}
            else:
                method = 'genetic'
                kwargs = {'population_size': 20, 'generations': 10}
            
            # 4. 运行优化 (传入 sliced_cache)
            with st.spinner(f"正在运行{optimization_method} (严格切片数据)..."):
                # 🔥 关键修改：这里传 sliced_cache，而不是 st.session_state.data_cache
                results = optimizer.optimize(
                    config=st.session_state.config,
                    data_cache=sliced_cache,  # <--- ✅ 使用切好的数据
                    method=method,
                    **kwargs
                )

            # ==========================================
            # 5. 验证器埋点 (存入切片后的证据)
            # ==========================================
            if 'global_validator' in st.session_state:
                st.session_state.global_validator.collect_independent(
                    config=st.session_state.config,
                    data_cache=sliced_cache, # <--- ✅ 存入切好的数据指纹
                    optimizer_results=results,
                    data_range_str=f"{st.session_state.config['start_date']}~{st.session_state.config['end_date']}"
                )
                st.toast("🕵️‍♂️ 验证器：手动优化数据(已切片)存档成功！")

            # 保存结果
            st.session_state.optimization_results = results
            
            # 显示结果
            if results:
                st.success(f"✅ 优化完成！共评估了 {len(results)} 种参数组合")
                SmartMoneyVisualizer.create_parameter_optimization_results(
                    results, PARAM_CHINESE_NAMES, key_suffix="manual_verify"
                )
            else:
                st.error("参数优化失败，没有有效的结果！")
    
    with tab4:
        st.header("🚀 回测执行")
        
        # 检查是否有配置
        if 'config' not in st.session_state:
            st.warning("请先在策略配置选项卡中设置并保存配置！")
        else:
            config = st.session_state.config
            
            # 运行按钮
            col1, col2, col3 = st.columns(3)
            
            with col1:
                run_backtest = st.button("🚀 开始回测", type="primary", use_container_width=True)
            
            with col2:
                run_optimization = st.button("🔧 快速参数优化", type="secondary", use_container_width=True,
                                            help="使用默认设置快速优化参数")
            
            with col3:
                if st.button("📊 查看上次回测结果", type="secondary", use_container_width=True):
                    if 'last_stats' in st.session_state and st.session_state.last_stats:
                        st.session_state.show_last_results = True
                    else:
                        st.warning("没有找到上次回测结果！")
            
            # 执行回测
            if run_backtest:
                if not config['symbols']:
                    st.error("请至少选择一个交易代币！")
                elif config['start_date'] >= config['end_date']:
                    st.error("开始日期必须早于结束日期！")
                else:
                    # 初始化回测引擎
                    engine = SmartMoneyBacktestEngine(config, proxy_config, use_proxy)
                    
                    # 如果有缓存数据，直接使用
                    if st.session_state.data_cache:
                        engine.data_cache = st.session_state.data_cache.copy()
                        st.info(f"使用缓存数据，共 {len(engine.data_cache)} 个代币")
                    
                    # 运行回测
                    with st.spinner("正在运行回测，请稍候..."):
                        try:
                            stats, data_cache = engine.run(timeframes=['1h', '4h'])
                            # 👇👇👇 【新增埋点】 👇👇👇
                            if stats:
                                st.session_state.diff_detective.capture_manual(
                                    config=st.session_state.config,
                                    data_cache=data_cache, # 注意这里传入的是引擎使用的 cache
                                    stats=stats
                                )
                                st.toast("🕵️ 侦探已记录手动回测现场数据")
                            # 👆👆👆
                            if stats and stats.get('trades'):
                                st.success(f"✅ 回测完成！共执行 {stats['total_trades']} 笔交易")
                                
                                # 更新会话状态
                                st.session_state.last_stats = stats
                                st.session_state.last_data_cache = data_cache
                                
                                # 记录到历史记录
                                if 'backtest_history' not in st.session_state:
                                    st.session_state.backtest_history = []
                                
                                # 构建历史记录项
                                history_entry = {
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                    'total_return': stats.get('total_return', 0),
                                    'win_rate': stats.get('win_rate', 0),
                                    'max_drawdown': stats.get('max_drawdown', 0),
                                    'sharpe': stats.get('sharpe', 0),
                                    'profit_factor': stats.get('profit_factor', 0),
                                    'total_trades': stats.get('total_trades', 0),
                                    'final_capital': stats.get('final_capital', 0),
                                    'params': config.copy(), # 保存当时的配置
                                    'symbols': ", ".join([s.split('/')[0] for s in config['symbols']]),
                                    'date_range': f"{config['start_date']} to {config['end_date']}",
                                    # 简单的最佳/最差代币分析
                                    'best_token': "N/A",
                                    'worst_token': "N/A"
                                }
                                
                                # 计算最佳/最差代币
                                symbol_pnl = defaultdict(float)
                                for t in stats['trades']:
                                    symbol_pnl[t.symbol] += t.pnl
                                
                                if symbol_pnl:
                                    best_sym = max(symbol_pnl.items(), key=lambda x: x[1])
                                    worst_sym = min(symbol_pnl.items(), key=lambda x: x[1])
                                    history_entry['best_token'] = f"{best_sym[0].split('/')[0]} (${best_sym[1]:.0f})"
                                    history_entry['worst_token'] = f"{worst_sym[0].split('/')[0]} (${worst_sym[1]:.0f})"
                                
                               
                                
                                st.session_state.backtest_history.append(history_entry)
                                
                                # ==========================================
                                # 🔥 修改：创建 4 个结果选项卡 (增加"深度复盘")
                                # ==========================================
                                result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
                                    "📊 概览", 
                                    "📈 资金曲线", 
                                    "💰 交易表现",
                                    "🔍 深度复盘"
                                ])
                                
                                with result_tab1:
                                    # 概览显示
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric(
                                            "总收益率",
                                            f"{stats.get('total_return', 0):.2f}%",
                                            f"${stats.get('total_pnl', 0):,.2f} U"
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            "年化收益率",
                                            f"{stats.get('annual_return', 0):.2f}%",
                                            f"Sharpe: {stats.get('sharpe', 0):.2f}"
                                        )
                                    
                                    with col3:
                                        st.metric(
                                            "胜率",
                                            f"{stats.get('win_rate', 0):.1f}%",
                                            f"{stats.get('winning_trades', 0)}/{stats.get('total_trades', 0)}"
                                        )
                                    
                                    with col4:
                                        st.metric(
                                            "最大回撤",
                                            f"{stats.get('max_drawdown', 0):.2f}%",
                                            f"Calmar: {stats.get('calmar', 0):.2f}"
                                        )
                                
                                with result_tab2:
                                    # 资金曲线
                                    SmartMoneyVisualizer.create_equity_curve(stats, key_suffix="current_run")
                                
                                with result_tab3:
                                    # 交易表现
                                    SmartMoneyVisualizer.create_trade_performance_chart(stats['trades'], key_suffix="current_run")
                                    
                                    # 聪明钱分析
                                    SmartMoneyVisualizer.create_smc_analysis(stats['trades'], key_suffix="current_run")
                                    
                                    # 交易详情
                                    SmartMoneyVisualizer.create_trade_details_table(stats['trades'])

                                # ==========================================
                                # 🔥 新增：深度复盘 Tab 内容
                                # ==========================================
                                with result_tab4:
                                    # 优先尝试获取 trades_history (包含更多细节)，如果没有则退化使用 trades
                                    trades_data = stats.get('trades_history', stats.get('trades', []))
                                    
                                    if trades_data and len(trades_data) > 0:
                                        # 调用刚才定义的 UI 函数 (请确保 display_trade_analysis_ui 已定义在文件头部)
                                        display_trade_analysis_ui(trades_data)
                                    else:
                                        st.info("📭 暂无交易记录，无法进行深度复盘")
                            else:
                                st.error("回测失败或没有交易数据！")
                                
                        except Exception as e:
                            st.error(f"回测过程中出现错误: {str(e)}")
                            # import traceback
                            # st.text(traceback.format_exc())
            
            # 执行快速参数优化
            if run_optimization:
                st.info("⚡ 正在启动快速参数优化 (网格搜索)...")
                
                if not st.session_state.data_cache:
                    st.error("请先获取数据！")
                else:
                    # ==========================================
                    # 1. 🔥 核心修复：强制全局指标预计算 (与滚动回测对齐)
                    # ==========================================
                    status_text = st.empty()
                    status_text.info("⚡ [一致性修正] 正在进行全局指标预热计算，消除预热偏差...")
                    
                    # 创建临时检测器
                    temp_detector = SmartMoneySignalDetector(st.session_state.config)
                    
                    # 强制计算所有缓存数据的指标
                    processed_cache = {}
                    for sym, timeframe_data in st.session_state.data_cache.items():
                        processed_cache[sym] = {}
                        for tf, df in timeframe_data.items():
                            if tf in ['1h', '4h'] and not df.empty:
                                # 基于全量历史数据计算指标
                                df_calculated = temp_detector.calculate_indicators(df.copy())
                                processed_cache[sym][tf] = df_calculated
                            else:
                                processed_cache[sym][tf] = df
                    
                    status_text.success(f"✅ 指标预热完成！快速优化将使用精准数据。")
                    
                    # ==========================================
                    # 2. 运行优化
                    # ==========================================
                    optimizer = AdvancedParameterOptimizer()
                    
                    with st.spinner("正在运行快速参数优化..."):
                        results = optimizer.optimize(
                            config=st.session_state.config,
                            data_cache=processed_cache,  # <--- 🔥 这里改成了 processed_cache
                            method='grid',
                            param_grid={
                                'ema_fast': [5, 9, 12],
                                'ema_slow': [50, 60, 70],
                                'rsi_period': [10, 14, 20],
                                'min_rr_ratio': [2.0, 2.5, 3.0],
                                'min_signal_score': [60, 70, 80],
                                'min_adx': [20, 25, 30],
                                'use_smc_logic': [False, True],
                                'use_dynamic_risk': [False, True]
                            }
                        )
                    
                    # ==========================================
                    # 3. 验证器埋点 (可选，用于对比验证)
                    # ==========================================
                    if 'global_validator' in st.session_state:
                        st.session_state.global_validator.collect_independent(
                            config=st.session_state.config,
                            data_cache=processed_cache, 
                            optimizer_results=results,
                            data_range_str=f"{st.session_state.config['start_date']}~{st.session_state.config['end_date']}"
                        )

                    # 保存结果
                    st.session_state.optimization_results = results
                    status_text.empty()
                    
                    # 显示结果
                    if results:
                        st.success(f"✅ 快速参数优化完成！共评估了 {len(results)} 种参数组合")
                        SmartMoneyVisualizer.create_parameter_optimization_results(
                            results, 
                            PARAM_CN_MAP, # 🔥 确保这里用的是全局定义的 PARAM_CN_MAP
                            key_suffix="manual_opt_grid" 
                        )
            
            # 显示上次回测结果
            if st.session_state.show_last_results:
                if 'last_stats' in st.session_state and st.session_state.last_stats:
                    stats = st.session_state.last_stats
                    
                    st.success("显示上次回测结果")
                    
                    # 创建结果选项卡
                    last_result_tab1, last_result_tab2, last_result_tab3 = st.tabs([
                        "📊 概览", 
                        "📈 资金曲线", 
                        "💰 交易表现"
                    ])
                    
                    with last_result_tab1:
                        # 概览显示
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "总收益率",
                                f"{stats.get('total_return', 0):.2f}%",
                                f"${stats.get('total_pnl', 0):,.2f} U"
                            )
                        
                        with col2:
                            st.metric(
                                "年化收益率",
                                f"{stats.get('annual_return', 0):.2f}%",
                                f"Sharpe: {stats.get('sharpe', 0):.2f}"
                            )
                        
                        with col3:
                            st.metric(
                                "胜率",
                                f"{stats.get('win_rate', 0):.1f}%",
                                f"{stats.get('winning_trades', 0)}/{stats.get('total_trades', 0)}"
                            )
                        
                        with col4:
                            st.metric(
                                "最大回撤",
                                f"{stats.get('max_drawdown', 0):.2f}%",
                                f"Calmar: {stats.get('calmar', 0):.2f}"
                            )
                    
                    with last_result_tab2:
                        # 资金曲线
                        SmartMoneyVisualizer.create_equity_curve(stats, key_suffix="last_run")
                    
                    with last_result_tab3:
                        # 交易表现
                        SmartMoneyVisualizer.create_trade_performance_chart(stats['trades'], key_suffix="last_run")
                        
                        # 聪明钱分析
                        SmartMoneyVisualizer.create_smc_analysis(stats['trades'], key_suffix="last_run")
                        
                        # 交易详情
                        SmartMoneyVisualizer.create_trade_details_table(stats['trades'])

    with tab5:
        show_backtest_history_tab()
    with tab6:
        st.header("🎲 蒙特卡洛压力测试")
        if 'last_stats' in st.session_state and st.session_state.last_stats:
            trades = st.session_state.last_stats['trades']
            init_cap = st.session_state.config['initial_capital']
            
            if st.button("开始模拟 (1000次)", type="primary"):
                MonteCarloAnalyzer.run_simulation(trades, init_cap)
        else:
            st.warning("请先运行一次回测以获取交易数据。")
    with tab7:
        st.header("🔄 6+1 滚动窗口回测 (Walk-Forward Analysis)")
        st.markdown("""
        > **这是检验策略是否过拟合的终极测试。**
        > 模拟真实世界：站在过去的时间点，优化参数，然后去交易未知的未来一个月。
        """)
        
        if not st.session_state.data_cache:
            st.error("请先在【数据获取】页面下载数据！建议下载 2023-01-01 至今的数据。")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # ==================================================
                # 1. 第一步：先定义所有输入控件 (Inputs)
                # ==================================================
                st.subheader("🛠️ 模拟设置")
                
                # [自动同步] 获取 Tab 1 的配置时间
                default_start = st.session_state.config.get('start_date', "2023-06-01")
                default_end = st.session_state.config.get('end_date', "2024-12-31")

                start_date_input = st.text_input("开始日期 (自动同步Tab1)", value=default_start)
                end_date_input = st.text_input("结束日期 (自动同步Tab1)", value=default_end)
                
                st.markdown("---")
                
                # [关键变量定义]
                train_m = st.number_input("训练窗口 (月) - 用于找参数", value=6, min_value=1)
                test_m = st.number_input("实战窗口 (月) - 用于验证", value=1, min_value=1)
                opt_trials = st.slider("每月优化尝试次数", 10, 300, 30)
                
                st.markdown("---")
                
                # 调试选项
                st.markdown("#### 🕵️‍♂️ 调试选项")
                lock_params_checkbox = st.checkbox(
                    "🔒 锁定参数 (强制使用Tab2配置)", 
                    value=False,
                    help="勾选后，滚动回测将不再优化参数，而是直接使用【策略配置】页面的参数。"
                )
                
                if lock_params_checkbox:
                    st.info("💡 已开启参数锁定。结果应与手动回测完全一致。")
                
                st.markdown("---")
                
                # 运行按钮
                run_btn = st.button(
                    "🚀 开始模拟实盘运维", 
                    type="primary", 
                    use_container_width=True, 
                    key="btn_start_rolling_sim_sidebar"
                )

            with col2:
                if run_btn:
                    # ================= [日志窗口初始化 V26.0] =================
                    if 'ui_log_queue' in st.session_state:
                        st.session_state.ui_log_queue = []
                    
                    st.markdown("### 📝 实时运行日志")
                    
                    with st.expander("查看详细运行过程 (实时滚动)", expanded=True):
                        st.session_state.log_placeholder = st.empty()
                        st.session_state.log_placeholder.markdown(
                            '<div style="padding:10px; color:gray; font-style:italic;">⏳ 日志系统就绪，等待交易信号...</div>', 
                            unsafe_allow_html=True
                        )

                    # 1. 实例化回测器
                    rolling_tester = RollingWindowBacktester(
                        st.session_state.config, 
                        st.session_state.data_cache
                    )
                    
                    # 2. 准备锁定参数
                    fixed_params = None
                    if lock_params_checkbox:
                        fixed_params = {k: v for k, v in st.session_state.config.items() 
                                      if k not in ['start_date', 'end_date', 'initial_capital', 'symbols']}
                    
                    status_container = st.empty()
                    status_container.info("⏳ 正在进行滚动回测...")
                    
                    # 3. 运行回测
                    try:
                        results, equity_curve = rolling_tester.run_6plus1_validation(
                            start_date=start_date_input,
                            end_date=end_date_input,
                            train_months=train_m,
                            test_months=test_m,
                            roll_step_months=1,
                            n_optimization_trials=opt_trials,
                            debug_fixed_params=fixed_params
                        )
                        
                        # 4. 显示结果
                        status_container.success("✅ 滚动回测完成！")
                        if results:
                            rolling_tester.analyze_rolling_results(results, equity_curve)
                            
                            # ==========================================
                            # 🔥 [修复] 新增：滚动回测-全周期深度复盘
                            # ==========================================
                            st.markdown("---")
                            st.header("🔍 滚动回测-全周期深度复盘")
                            
                            # 1. 收集所有窗口的交易
                            all_rolling_trades = []
                            for res in results:
                                # 🔥 核心修复：增加 detailed_trades 读取
                                trades = res.get('detailed_trades', res.get('trades_history', res.get('trades', [])))
                                if trades:
                                    all_rolling_trades.extend(trades)
                            
                            # 2. 调用复盘界面
                            if all_rolling_trades:
                                display_trade_analysis_ui(all_rolling_trades)
                            else:
                                st.warning("⚠️ 滚动回测期间未产生任何交易。")

                            # 验证器埋点
                            if 'global_validator' in st.session_state:
                                st.session_state.global_validator.collect_rolling(
                                    config=st.session_state.config,
                                    data_cache=st.session_state.data_cache,
                                    optimizer_results=[],
                                    data_range_str=f"{start_date_input}~{end_date_input}"
                                )
                    except Exception as e:
                        st.error(f"❌ 回测运行出错: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                
                # ==================================================
                # 2. 第二步：环境核对大屏显示 (Display)
                # ==================================================
                st.subheader("🧐 模拟环境核对")

                # 准备数据
                curr_cfg = st.session_state.config
                
                # 资金模式逻辑
                is_fixed = curr_cfg.get('position_mode') == 'fixed'
                mode_label = "固定仓位" if is_fixed else "复合增长"
                mode_icon = "💰" if is_fixed else "🚀"
                if is_fixed:
                    pos_main = f"${curr_cfg.get('target_position_value', 0):,.0f}"
                    pos_sub = "单仓价值"
                else:
                    pos_main = f"比例 {curr_cfg.get('compounding_ratio', 0):.1f}"
                    pos_sub = "复利 (1.0=全仓)"

                # 计算总跨度
                try:
                    s_date_roll = datetime.strptime(start_date_input, '%Y-%m-%d')
                    e_date_roll = datetime.strptime(end_date_input, '%Y-%m-%d')
                    total_days_roll = (e_date_roll - s_date_roll).days
                    days_display_roll = f"{total_days_roll} 天"
                except:
                    days_display_roll = "日期格式错误"

                # === 板块一：资金配置 ===
                st.caption("💰 **资金设定 (来自配置页)**")
                m_c1, m_c2 = st.columns(2)
                with m_c1: st.metric("资金模式", f"{mode_icon} {mode_label}", pos_sub)
                with m_c2: st.metric("初始本金", f"${curr_cfg.get('initial_capital'):,.0f}", f"杠杆: {curr_cfg.get('leverage')}x")
                
                m_c3, m_c4 = st.columns(2)
                with m_c3: st.metric("仓位规模", pos_main)
                with m_c4: st.metric("单次优化", f"{opt_trials} 次", "贝叶斯尝试数")

                # === 板块二：时间配置 ===
                st.markdown("---")
                st.caption("📅 **模拟时间轴 (Timeline)**")
                
                t_c1, t_c2 = st.columns(2)
                with t_c1: st.metric("模拟开始", start_date_input)
                with t_c2: st.metric("模拟结束", end_date_input)
                
                t_c3, t_c4 = st.columns(2)
                with t_c3: st.metric("K线周期", f"⏰ {curr_cfg.get('check_interval_hours')} 小时", "策略频率")
                with t_c4: st.metric("总跨度", days_display_roll, f"训练{train_m}月 + 实战{test_m}月")

                # 警告条
                st.markdown(
                    f"""
                    <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; border: 1px solid #ffeeba; color: #856404; text-align: center; margin-top: 10px; margin-bottom: 20px;">
                        🚧 <strong>高能预警：</strong> 即将进行 <strong>{days_display_roll}</strong> 的超长模拟。
                        请确认数据已下载覆盖 <strong>{start_date_input}</strong> 至 <strong>{end_date_input}</strong> 的完整区间！
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
            # 蒙特卡洛入口
            st.markdown("---")
            with st.expander("🎲 高级功能：蒙特卡洛随机起点验证"):
                if st.button("运行蒙特卡洛验证 (50次)"):
                    validator = MonteCarloRollingValidator()
                    res = validator.run_monte_carlo_validation(
                        st.session_state.config, 
                        st.session_state.data_cache
                    )
                    st.write(f"平均收益率分布: {res}")
    with tab8:
        st.header("🕵️ 显微镜：手动 vs 滚动 深度差异分析")
        st.markdown("""
        > **为什么同时间不同单？** 此工具将逐项核对两个环境的原子级差异。
        > 只有当 **参数、本金、指标数值** 完全一致时，交易结果才会一致。
        """)

        detective = st.session_state.diff_detective

        if not detective.manual_snapshot:
            st.warning("⚠️ 请先在【Tab 4】运行一次手动回测，作为对比基准。")
        elif not detective.rolling_snapshots:
            st.warning("⚠️ 请先在【Tab 7】运行滚动回测，作为对比目标。")
        else:
            # 1. 选择要对比的滚动窗口
            window_options = list(detective.rolling_snapshots.keys())
            selected_window = st.selectbox(
                "选择要对比的滚动窗口 (Rolling Window)", 
                window_options,
                format_func=lambda x: f"第 {x} 轮窗口 (Window {x})"
            )
            
            manual = detective.manual_snapshot
            rolling = detective.rolling_snapshots[selected_window]

            # ---------------------------------------
            # 第一部分：核心资金与环境对比
            # ---------------------------------------
            st.subheader("1. 🏦 资金与环境 (Environment Check)")
            
            env_data = [
                {
                    "项目": "回测类型",
                    "手动回测 (基准)": "Manual Run",
                    "滚动回测 (目标)": f"Window {selected_window}",
                    "状态": "ℹ️ Info"
                },
                {
                    "项目": "初始本金 (Initial Capital)",
                    "手动回测 (基准)": f"${manual['stats']['initial_capital']}",
                    "滚动回测 (目标)": f"${rolling['stats']['initial_capital']}",
                    "状态": "✅" if manual['stats']['initial_capital'] == rolling['stats']['initial_capital'] else "❌ 资金不同导致仓位不同"
                },
                {
                    "项目": "回测区间 (Range)",
                    "手动回测 (基准)": f"{manual['config']['start_date']} ~ {manual['config']['end_date']}",
                    "滚动回测 (目标)": f"{rolling['config']['start_date']} ~ {rolling['config']['end_date']}",
                    "状态": "⚠️ 必须重合才能对比"
                }
            ]
            st.dataframe(pd.DataFrame(env_data), use_container_width=True)

            # ---------------------------------------
            # 第二部分：指标指纹对比 (Indicator Fingerprint)
            # ---------------------------------------
            st.subheader("2. 🌡️ 指标预热偏差检测 (Warm-up Bias)")
            st.caption("检查同一时间点的指标值是否一致。如果数值不同，说明**历史数据长度**不同导致指标计算结果有偏差。")
            
            mf = manual['indicator_fingerprint']
            rf = rolling['indicator_fingerprint']
            
            if mf and rf:
                # 尝试对齐时间
                t_diff_msg = "✅ 时间点匹配" if mf['sample_time'] == rf['sample_time'] else f"⚠️ 取样时间不同 (Man:{mf['sample_time']} vs Roll:{rf['sample_time']})"
                
                fp_data = [
                    {"指标": "取样时间", "手动": mf['sample_time'], "滚动": rf['sample_time'], "差异": t_diff_msg},
                    {"指标": "EMA Fast", "手动": f"{mf['ema_fast']:.4f}", "滚动": f"{rf['ema_fast']:.4f}", "差异": abs(mf['ema_fast'] - rf['ema_fast'])},
                    {"指标": "EMA Slow", "手动": f"{mf['ema_slow']:.4f}", "滚动": f"{rf['ema_slow']:.4f}", "差异": abs(mf['ema_slow'] - rf['ema_slow'])},
                    {"指标": "RSI", "手动": f"{mf['rsi']:.2f}", "滚动": f"{rf['rsi']:.2f}", "差异": abs(mf['rsi'] - rf['rsi'])},
                    {"指标": "数据物理起点", "手动": mf['data_start_date'], "滚动": rf['data_start_date'], "差异": "决定预热是否充分"}
                ]
                st.dataframe(pd.DataFrame(fp_data), use_container_width=True)
                
                if abs(mf['ema_slow'] - rf['ema_slow']) > 0.01:
                    st.error("🚨 **发现严重指标偏差！** 即使参数相同，由于数据物理起点不同，EMA计算结果不一致，这会导致进出场信号不同步。")
                    st.markdown("**解决方案**：在滚动回测中，确保 `train` 和 `test` 数据连接时有足够的重叠区(Lookback Buffer)用于指标预热。")
            else:
                st.info("无法获取指标指纹 (可能是数据为空1)")

            # ---------------------------------------
            # 第三部分：全参数逐项对比 (The Full Diff)
            # ---------------------------------------
            st.subheader("3. 🔧 全参数逐项对比 (Parameter Diff)")
            
            m_conf = manual['config']
            r_conf = rolling['config']
            
            # 展平字典以便对比 (处理嵌套的权重)
            def flatten_dict(d, parent_key='', sep='.'):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)

            flat_m = flatten_dict(m_conf)
            flat_r = flatten_dict(r_conf)
            
            all_keys = sorted(set(flat_m.keys()) | set(flat_r.keys()))
            
            diff_rows = []
            
            # 定义不需要对比的无关参数1
            ignore_keys = ['start_date', 'end_date', 'initial_capital', 'symbols'] 
            
            for k in all_keys:
                if any(ign in k for ign in ignore_keys): continue
                
                val_m = flat_m.get(k, 'N/A')
                val_r = flat_r.get(k, 'N/A')
                
                # 格式化浮点数
                if isinstance(val_m, float): val_m = round(val_m, 4)
                if isinstance(val_r, float): val_r = round(val_r, 4)
                
                is_diff = val_m != val_r
                status = "❌ 不同" if is_diff else "✅ 相同1"
                
                # 如果不同，高亮显示
                if is_diff:
                    diff_rows.insert(0, {"参数名": k, "手动设置": val_m, "滚动实战": val_r, "状态": status})
                else:
                    diff_rows.append({"参数名": k, "手动设置": val_m, "滚动实战": val_r, "状态": status})
            
            df_diff = pd.DataFrame(diff_rows)
            
            # 样式设置
            def highlight_diff(row):
                return ['background-color: #ffeeba' if row['状态'] == "❌ 不同" else '' for _ in row]

            st.dataframe(df_diff.style.apply(highlight_diff, axis=1), use_container_width=True, height=600)
            
            st.info("""
            **如何解读差异：**
            1. **❌ 不同**：这是导致订单不同的直接原因。滚动回测使用的是当时优化出的“局部最优解”，而手动回测用的是你现在设置的“全局参数”。
            2. **资金差异**：如果初始本金不同，仓位大小(Position Size)就会不同，可能导致部分订单因资金不足被过滤，或触及风控线。
            3. **指标偏差**：如果EMA/RSI有微小差异，那些刚好在阈值附近的信号（例如 RSI=70.01 vs 69.99）就会产生蝴蝶效应。
            """)
            
if __name__ == "__main__":
    main()
