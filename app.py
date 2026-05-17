import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from joblib import Parallel, delayed
# =========================================================
# CONFIGURAÇÃO
# =========================================================
st.set_page_config(
    page_title="Portfolio Institucional B3",
    layout="wide"
)
st.title("📊 Portfolio Institucional B3")
st.caption("Seleção por qualidade estatística — Setup S2 · Filtro de estabilidade 2y/5y · universo profissional validado")
# =========================================================
# UNIVERSO CLASSIFICADO — validação estatística (universo profissional limpo)
# Ações + FIIs + ETFs + BDRs — 113 ativos submetidos
#
# CORE      → delta ≤ 5 pp  + Prob_2y ≥ 60%   (24 ativos)
# WATCHLIST → delta ≤ 15 pp + Prob_2y ≥ 50%   (44 ativos)
# EXCLUIDOS → delta > 15 pp ou Prob_2y < 50% ou dados insuficientes
# =========================================================
CORE = [
    "GOGL34.SA",   # [BDR]  delta 4.6 pp | 77.6% / 72.9%
    "WALM34.SA",   # [BDR]  delta 0.9 pp | 75.3% / 76.2%
    "TAEE11.SA",   # [ACAO] delta 4.3 pp | 74.2% / 69.9%
    "UGPA3.SA",    # [ACAO] delta 1.4 pp | 73.9% / 72.5%
    "BPAC11.SA",   # [ACAO] delta 0.4 pp | 73.8% / 74.2%
    "CSMG3.SA",    # [ACAO] delta 2.5 pp | 73.4% / 70.9%
    "NVDC34.SA",   # [BDR]  delta 5.0 pp | 72.3% / 67.3%
    "PRIO3.SA",    # [ACAO] delta 3.1 pp | 70.8% / 67.7%
    "ITSA4.SA",    # [ACAO] delta 5.0 pp | 70.6% / 65.6%
    "ASAI3.SA",    # [ACAO] delta 1.6 pp | 68.6% / 67.0%
    "RBRP11.SA",   # [FII]  delta 0.1 pp | 67.7% / 67.7%
    "AAPL34.SA",   # [BDR]  delta 0.7 pp | 66.4% / 65.7%
    "EQTL3.SA",    # [ACAO] delta 1.3 pp | 66.0% / 64.7%
    "AMZO34.SA",   # [BDR]  delta 1.9 pp | 65.7% / 67.6%
    "DISB34.SA",   # [BDR]  delta 0.7 pp | 65.6% / 64.9%
    "ITUB4.SA",    # [ACAO] delta 3.0 pp | 64.5% / 61.5%
    "VBBR3.SA",    # [ACAO] delta 1.8 pp | 64.4% / 66.1%
    "LREN3.SA",    # [ACAO] delta 1.5 pp | 64.1% / 65.6%
    "BBDC4.SA",    # [ACAO] delta 0.8 pp | 63.1% / 62.3%
    "BBDC3.SA",    # [ACAO] delta 3.0 pp | 61.7% / 58.6%
    "MSFT34.SA",   # [BDR]  delta 4.0 pp | 61.6% / 65.6%
    "MRCK34.SA",   # [BDR]  delta 3.7 pp | 61.5% / 57.8%
    "BBAS3.SA",    # [ACAO] delta 0.1 pp | 61.2% / 61.1%
    "SAPR11.SA",   # [ACAO] delta 0.7 pp | 60.7% / 60.0%
]
WATCHLIST = [
    "MRVE3.SA",    # [ACAO] delta  0.2 pp | 57.0% / 57.2%
    "EZTC3.SA",    # [ACAO] delta  0.9 pp | 57.6% / 56.7%
    "FLRY3.SA",    # [ACAO] delta  1.3 pp | 54.5% / 55.7%
    "SMAL11.SA",   # [ETF]  delta  1.5 pp | 57.9% / 59.4%
    "ENGI11.SA",   # [ACAO] delta  2.0 pp | 59.1% / 61.1%
    "HAPV3.SA",    # [ACAO] delta  2.3 pp | 53.3% / 55.6%
    "VILG11.SA",   # [FII]  delta  2.5 pp | 58.9% / 56.4%
    "SANB11.SA",   # [ACAO] delta  2.9 pp | 51.6% / 48.7%
    "LWSA3.SA",    # [ACAO] delta  4.9 pp | 59.2% / 64.1%
    "TEND3.SA",    # [ACAO] delta  5.1 pp | 52.1% / 57.2%
    "KNRI11.SA",   # [FII]  delta  5.1 pp | 54.2% / 49.2%
    "JNJB34.SA",   # [BDR]  delta  5.2 pp | 64.6% / 59.4%
    "PEPB34.SA",   # [BDR]  delta  5.2 pp | 62.9% / 57.7%
    "MGLU3.SA",    # [ACAO] delta  5.4 pp | 61.9% / 56.5%
    "TSLA34.SA",   # [BDR]  delta  5.4 pp | 68.4% / 63.0%
    "RURA11.SA",   # [FII]  delta  5.4 pp | 57.4% / 52.0%
    "BRAP4.SA",    # [ACAO] delta  5.9 pp | 77.0% / 71.1%
    "RDOR3.SA",    # [ACAO] delta  6.0 pp | 72.5% / 66.5%
    "HFOF11.SA",   # [FII]  delta  6.1 pp | 56.3% / 50.1%
    "USIM5.SA",    # [ACAO] delta  6.2 pp | 60.3% / 66.5%
    "RECR11.SA",   # [FII]  delta  6.3 pp | 63.2% / 56.9%
    "HGRE11.SA",   # [FII]  delta  6.4 pp | 70.0% / 63.6%
    "JPMC34.SA",   # [BDR]  delta  6.8 pp | 78.0% / 71.2%
    "RECV3.SA",    # [ACAO] delta  7.1 pp | 62.1% / 54.9%
    "PVBI11.SA",   # [FII]  delta  7.3 pp | 67.5% / 60.3%
    "DIRR3.SA",    # [ACAO] delta  7.4 pp | 63.4% / 56.0%
    "TOTS3.SA",    # [ACAO] delta  8.1 pp | 69.4% / 61.3%
    "HSML11.SA",   # [FII]  delta  8.3 pp | 54.0% / 45.7%
    "BOVA11.SA",   # [ETF]  delta  8.4 pp | 64.5% / 56.2%
    "SBSP3.SA",    # [ACAO] delta  8.4 pp | 71.9% / 63.5%
    "CYRE3.SA",    # [ACAO] delta  8.5 pp | 75.5% / 67.0%
    "RBRF11.SA",   # [FII]  delta  8.6 pp | 68.6% / 60.0%
    "WEGE3.SA",    # [ACAO] delta  8.8 pp | 60.8% / 69.6%
    "GGBR4.SA",    # [ACAO] delta  9.8 pp | 75.5% / 65.7%
    "SUZB3.SA",    # [ACAO] delta 10.0 pp | 50.0% / 60.0%
    "PETR4.SA",    # [ACAO] delta 10.2 pp | 67.8% / 57.6%
    "MCCI11.SA",   # [FII]  delta 11.2 pp | 54.8% / 43.6%
    "VALE3.SA",    # [ACAO] delta 11.3 pp | 82.9% / 71.6%
    "CPFE3.SA",    # [ACAO] delta 12.7 pp | 85.2% / 72.5%
    "DIVO11.SA",   # [ETF]  delta 13.4 pp | 68.8% / 55.4%
    "GOAU4.SA",    # [ACAO] delta 14.1 pp | 75.3% / 61.2%
    "RBRR11.SA",   # [FII]  delta 14.3 pp | 52.8% / 38.5%
    "PETR3.SA",    # [ACAO] delta 14.4 pp | 72.8% / 58.4%
    "JSRE11.SA",   # [FII]  delta 14.7 pp | 70.3% / 55.6%
]
# Excluídos: não entram no scanner
EXCLUIDOS = {
    "HGCR11": "prob 11.1% < 50%",
    "CPTS11":  "prob 41.2% < 50%",
    "XPLG11":  "prob 33.3% < 50%",
    "HGRU11":  "prob 40.0% < 50%",
    "HGLG11":  "prob 18.8% < 50%",
    "MXRF11":  "prob 18.8% < 50%",
    "CMIG4":   "prob 40.0% < 50%",
    "KNIP11":  "prob 0.0% < 50%",
    "ALZR11":  "prob 2.9% < 50%",
    "KNSC11":  "prob 15.4% < 50%",
    "IRDM11":  "prob 25.0% < 50%",
    "VGIR11":  "prob 23.1% < 50%",
    "TGAR11":  "prob 34.0% < 50%",
    "BRCO11":  "prob 40.7% < 50%",
    "VISC11":  "prob 31.7% < 50%",
    "CSNA3":   "prob 47.0% < 50%",
    "RZTR11":  "delta 15.2 pp",
    "GGRC11":  "delta 15.9 pp",
    "BEEF3":   "delta 17.2 pp",
    "BTLG11":  "delta 22.3 pp",
    "XPML11":  "delta 22.4 pp",
    "VRTA11":  "delta 23.3 pp",
    "IVVB11":  "delta 23.9 pp",
    "TRXF11":  "delta 27.1 pp",
    "JBSS3":   "dados insuficientes",
    "BRFS3":   "dados insuficientes",
    "MRFG3":   "dados insuficientes",
    "ELET3":   "dados insuficientes",
    "ELET6":   "dados insuficientes",
    "RAIL3":   "dados insuficientes",
    "CCRO3":   "dados insuficientes",
    "AZUL4":   "dados insuficientes",
    "GOLL4":   "dados insuficientes",
    "CRFB3":   "dados insuficientes",
    "ARZZ3":   "dados insuficientes",
    "POSI3":   "dados insuficientes",
    "MALL11":  "dados insuficientes",
    "KNCR11":  "dados insuficientes",
    "CVBI11":  "dados insuficientes",
    "META34":  "dados insuficientes",
    "SBUX34":  "dados insuficientes",
    "KOCA34":  "dados insuficientes",
    "COST34":  "dados insuficientes",
    "PFEF34":  "dados insuficientes",
    "NKEE34":  "dados insuficientes",
}
EMA     = 69
ADX_MIN = 25
# Delta validado (2y vs 5y) — Portfólio Operacional
# Usado somente para Rank_op (visualização): Score × (1 - delta/20)
CORE_DELTA = {
    "GOGL34": 4.6, "WALM34": 0.9, "TAEE11": 4.3, "UGPA3":  1.4,
    "BPAC11": 0.4, "CSMG3":  2.5, "NVDC34": 5.0, "PRIO3":  3.1,
    "ITSA4":  5.0, "ASAI3":  1.6, "RBRP11": 0.1, "AAPL34": 0.7,
    "EQTL3":  1.3, "AMZO34": 1.9, "DISB34": 0.7, "ITUB4":  3.0,
    "VBBR3":  1.8, "LREN3":  1.5, "BBDC4":  0.8, "BBDC3":  3.0,
    "MSFT34": 4.0, "MRCK34": 3.7, "BBAS3":  0.1, "SAPR11": 0.7,
}
# =========================================================
# CONTROLES
# =========================================================
st.sidebar.header("⚙️ Parâmetros")
gain    = st.sidebar.slider("Gain %",        1.0, 15.0, 3.0)
stop    = st.sidebar.slider("Stop %",        1.0, 15.0, 5.0)
janela  = st.sidebar.slider("Janela (dias)", 5,   60,   25)
periodo = st.sidebar.selectbox("Histórico", ["2y", "5y"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("**Composição do sistema**")
st.sidebar.markdown(f"🔵 Portfólio Operacional: {len(CORE)} ativos")
st.sidebar.markdown(f"🟡 Universo de Observação: {len(WATCHLIST)} ativos")
st.sidebar.markdown(f"🔴 Excluídos: {len(EXCLUIDOS)} ativos")
# =========================================================
# DADOS
# =========================================================
@st.cache_data(ttl=3600)
def get_data(ticker: str, periodo_param: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=periodo_param, interval="1d",
                         auto_adjust=False, progress=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    return df
# =========================================================
# ANÁLISE  (backtest e score intocados)
# =========================================================
def analisar(ticker: str) -> dict | None:
    try:
        raw = get_data(ticker, periodo)
        if raw is None or len(raw) < 120:
            return None
        df = raw.copy()
        close = df["Close"].squeeze()
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()
        df["EMA69"] = ta.trend.ema_indicator(close, window=EMA)
        df["EMA9"]  = ta.trend.ema_indicator(close, window=9)
        adx_ind     = ta.trend.ADXIndicator(high, low, close, window=14)
        df["ADX"]   = adx_ind.adx()
        stoch_ind   = ta.momentum.StochasticOscillator(
            high, low, close, window=14, smooth_window=3
        )
        df["STOCH"] = stoch_ind.stoch()
        df["EMA_SLOPE"] = df["EMA69"].diff()
        df["TREND"] = np.where(
            (df["Close"] > df["EMA69"]) & (df["EMA_SLOPE"] > 0) & (df["ADX"] > ADX_MIN),
            1, 0
        )
        # Backtest (lógica original — entrada em TREND)
        close_arr = df["Close"].to_numpy(dtype=float)
        high_arr  = df["High"].to_numpy(dtype=float)
        low_arr   = df["Low"].to_numpy(dtype=float)
        trend_arr = df["TREND"].to_numpy(dtype=int)
        wins = trades = 0
        n = len(df)
        for i in range(120, n - janela):
            if trend_arr[i] == 0:
                continue
            entry = close_arr[i]
            if not np.isfinite(entry):
                continue
            tp = entry * (1 + gain / 100)
            sl = entry * (1 - stop / 100)
            fut_h = high_arr[i + 1:i + 1 + janela]
            fut_l = low_arr [i + 1:i + 1 + janela]
            sh = np.where(fut_l  <= sl)[0]; first_stop = sh[0] if len(sh) else janela + 1
            th = np.where(fut_h  >= tp)[0]; first_tp   = th[0] if len(th) else janela + 1
            trades += 1
            if first_tp < first_stop:
                wins += 1
        prob = (wins / trades * 100) if trades > 0 else 0
        # Setups (NaN guard)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        def safe(v):
            x = float(v)
            return x if np.isfinite(x) else None
        lc, le69, le9 = safe(last["Close"]), safe(last["EMA69"]), safe(last["EMA9"])
        ladx, lstoch  = safe(last["ADX"]),   safe(last["STOCH"])
        pc, pe69, pe9 = safe(prev["Close"]), safe(prev["EMA69"]), safe(prev["EMA9"])
        if any(v is None for v in [lc, le69, le9, ladx, lstoch, pc, pe69, pe9]):
            return None
        # S2 — único setup exibido como sinal principal
        s2 = pe9 < pc and lc > le9
        score = prob * 0.5 + ladx * 0.8 + (1 if lc > le69 else 0) * 20
        return {
            "Ticker":  ticker.replace(".SA", ""),
            "Preço":   round(lc, 2),
            "Prob%":   round(prob, 1),
            "Score":   round(score, 1),
            "Regime":  "TREND" if int(last["TREND"]) == 1 else "SIDEWAYS",
            "S2":      s2,
            "Trades":  trades,
        }
    except Exception:
        return None
# =========================================================
# GRÁFICO DE DETALHE
# =========================================================
@st.cache_data(ttl=3600)
def preparar_detalhe(ticker: str, periodo_param: str) -> pd.DataFrame | None:
    raw = get_data(ticker, periodo_param)
    if raw is None or len(raw) < 50:
        return None
    df = raw.copy()
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    df["EMA9"]   = ta.trend.ema_indicator(close, window=9)
    df["EMA29"]  = ta.trend.ema_indicator(close, window=29)
    df["EMA69"]  = ta.trend.ema_indicator(close, window=EMA)
    df["EMA169"] = ta.trend.ema_indicator(close, window=169)
    adx_ind     = ta.trend.ADXIndicator(high, low, close, window=14)
    df["ADX"]   = adx_ind.adx()
    stoch_ind   = ta.momentum.StochasticOscillator(
        high, low, close, window=14, smooth_window=3
    )
    df["STOCH"] = stoch_ind.stoch()
    df["EMA_SLOPE"] = df["EMA69"].diff()
    df["TREND"] = np.where(
        (df["Close"] > df["EMA69"]) & (df["EMA_SLOPE"] > 0) & (df["ADX"] > ADX_MIN),
        1, 0
    )
    # S2 histórico (único setup no gráfico)
    df["S2"] = (
        (df["Close"].shift(1) < df["EMA9"].shift(1)) &
        (df["Close"] > df["EMA9"])
    ).fillna(False)
    return df
def plotar_detalhe(ticker_sa: str, ticker_label: str) -> None:
    df = preparar_detalhe(ticker_sa, periodo)
    if df is None:
        st.warning(f"Dados insuficientes para {ticker_label}.")
        return
    fig = go.Figure()
    # Faixas de regime
    trend_vals = df["TREND"].values
    datas      = df.index
    i = 0
    while i < len(trend_vals):
        val = trend_vals[i]; j = i
        while j < len(trend_vals) and trend_vals[j] == val:
            j += 1
        fig.add_vrect(
            x0=str(datas[i]), x1=str(datas[j - 1]),
            fillcolor="rgba(38,166,154,0.08)" if val == 1 else "rgba(120,120,120,0.05)",
            layer="below", line_width=0
        )
        i = j
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(),   close=df["Close"].squeeze(),
        name="Preço",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
    ))
    # EMAs
    for nome, cor, lw in [
        ("EMA9",   "#64B5F6", 1.2),
        ("EMA29",  "#FFB74D", 1.4),
        ("EMA69",  "#CE93D8", 2.0),
        ("EMA169", "#EF9A9A", 1.4),
    ]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[nome], name=nome,
            line=dict(color=cor, width=lw), opacity=0.85,
        ))
    # S2 — único sinal marcado no gráfico
    s2_pts = df[df["S2"]]
    if not s2_pts.empty:
        fig.add_trace(go.Scatter(
            x=s2_pts.index,
            y=s2_pts["Low"].squeeze() * 0.992,
            mode="markers",
            name="S2 — Cruzamento EMA9",
            marker=dict(symbol="circle", color="#00E5FF", size=9,
                        line=dict(width=1, color="rgba(0,0,0,0.4)")),
        ))
    ultimo_regime = "TREND 🟢" if int(df["TREND"].iloc[-1]) == 1 else "SIDEWAYS ⚪"
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.98,
        text=f"<b>Regime: {ultimo_regime}</b>",
        showarrow=False, font=dict(size=13, color="#ffffff"),
        bgcolor="rgba(0,0,0,0.45)", borderpad=5,
    )
    fig.update_layout(
        title=f"📈 {ticker_label} — Candlestick · EMAs · S2",
        height=560, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
    )
    st.plotly_chart(fig, use_container_width=True)
# =========================================================
# EXECUÇÃO — SCANNER
# =========================================================
if st.button("🚀 Rodar Scanner"):
    with st.spinner("Analisando universo..."):
        res_core  = Parallel(n_jobs=1, prefer="threads")(
            delayed(analisar)(t) for t in CORE
        )
        res_watch = Parallel(n_jobs=1, prefer="threads")(
            delayed(analisar)(t) for t in WATCHLIST
        )
    core_rows  = [r for r in res_core  if r is not None]
    watch_rows = [r for r in res_watch if r is not None]
    if not core_rows and not watch_rows:
        st.error("Nenhum ativo retornou dados. Tente novamente.")
        st.stop()
    # Ranking composto para o Portfólio Operacional: Score × (1 - delta/20)
    # Penaliza ativos com maior instabilidade entre períodos, sem alterar o Score original
    op_df = pd.DataFrame(core_rows)
    op_df["Delta_pp"] = op_df["Ticker"].map(CORE_DELTA).fillna(0.0)
    op_df["Rank_op"]  = (op_df["Score"] * (1 - op_df["Delta_pp"] / 20)).round(1)
    op_df = op_df.sort_values("Rank_op", ascending=False)
    st.session_state["core_df"]  = op_df
    st.session_state["watch_df"] = pd.DataFrame(watch_rows).sort_values("Score", ascending=False)
# ─── Exibe resultados ──────────────────────────────────────────────────────
if "core_df" in st.session_state:
    core_df  = st.session_state["core_df"]
    watch_df = st.session_state["watch_df"]
    # ══ TRADES OPERACIONAIS ═══════════════════════════════════════════════
    # Filtro de entrada: Core + S2 ativo + Regime TREND + Prob% ≥ 65%
    # Regime TREND já implica: Close > EMA69, slope > 0, ADX > 25 (> 20)
    trades_op = core_df[
        (core_df["S2"] == True) &
        (core_df["Regime"] == "TREND") &
        (core_df["Prob%"] >= 65.0)
    ].copy()
    st.markdown("---")
    st.subheader("🟢 Trades Operacionais")
    st.caption(
        "Ativos do Core que passam em todos os filtros de entrada simultâneos: "
        "S2 ativo · Regime TREND (Close > EMA69 · slope > 0 · ADX > 25) · Prob% ≥ 65%"
    )
    if trades_op.empty:
        st.info("Nenhum ativo passa no filtro de entrada agora.")
    else:
        cols_trades = ["Ticker", "Preço", "Score", "Prob%", "S2"]
        cols_trades = [c for c in cols_trades if c in trades_op.columns]
        st.dataframe(
            trades_op[cols_trades].sort_values("Prob%", ascending=False),
            use_container_width=True,
        )
    # ══ PORTFÓLIO OPERACIONAL ════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔵 Portfólio Operacional")
    st.caption(
        "24 ativos com edge estatisticamente estável · delta 2y/5y ≤ 5 pp · Prob ≥ 60% · "
        "Ordenado por Rank_op = Score × (1 − Δ/20)"
    )
    # Alerta S2 — EXCLUSIVO do Portfólio Operacional
    core_s2 = core_df[core_df["S2"] == True]
    if not core_s2.empty:
        sinais = core_s2.sort_values("Rank_op", ascending=False)
        tickers_s2 = "  ·  ".join(
            f"{r['Ticker']} (Rank {r['Rank_op']})" for _, r in sinais.iterrows()
        )
        st.success(f"🔷 SINAL S2 ATIVO — Portfólio Operacional: {tickers_s2}")
    else:
        st.info("Nenhum sinal S2 ativo no Portfólio Operacional no momento.")
    # Colunas exibidas: Rank_op primeiro, depois demais métricas
    cols_op = ["Rank_op", "Ticker", "Preço", "Score", "Prob%", "Regime", "S2",
               "Trades", "Delta_pp"]
    cols_op = [c for c in cols_op if c in core_df.columns]
    st.dataframe(core_df[cols_op], use_container_width=True)
    st.download_button(
        "⬇️ Exportar Portfólio Operacional CSV",
        data=core_df[cols_op].to_csv(index=False).encode("utf-8"),
        file_name="portfolio_operacional.csv",
        mime="text/csv",
    )
    # ══ UNIVERSO DE OBSERVAÇÃO ════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🟡 Universo de Observação")
    st.caption(
        "44 ativos com edge presente mas instabilidade entre períodos (delta 5–15 pp) · "
        "Apenas análise — sem sinais automáticos"
    )
    # Sem alerta S2: tabela apenas para referência analítica
    cols_obs = ["Ticker", "Preço", "Score", "Prob%", "Regime", "S2", "Trades"]
    cols_obs = [c for c in cols_obs if c in watch_df.columns]
    st.dataframe(watch_df[cols_obs], use_container_width=True)
    st.download_button(
        "⬇️ Exportar Universo de Observação CSV",
        data=watch_df[cols_obs].to_csv(index=False).encode("utf-8"),
        file_name="universo_observacao.csv",
        mime="text/csv",
    )
    # ── Excluídos ─────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🔴 Ativos fora do sistema — ver motivo"):
        exc_rows = [{"Ticker": k, "Motivo": v} for k, v in EXCLUIDOS.items()]
        st.dataframe(pd.DataFrame(exc_rows), use_container_width=True)
    # ══ GRÁFICO DE DETALHE ════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔍 Detalhe do Ativo")
    st.caption("Análise disponível para todos os ativos do sistema — Portfólio Operacional e Universo de Observação")
    all_df = pd.concat([
        core_df.assign(Grupo="Operacional"),
        watch_df.assign(Grupo="Observação")
    ], ignore_index=True)
    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        selecionado = st.selectbox(
            "Selecione o ativo:",
            all_df["Ticker"].tolist(),
            format_func=lambda t: (
                f"🔵 {t}" if t in core_df["Ticker"].values else f"🟡 {t}"
            ),
            index=0
        )
    with col_info:
        linha        = all_df[all_df["Ticker"] == selecionado].iloc[0]
        regime_badge = "🟢 TREND" if linha["Regime"] == "TREND" else "⚪ SIDEWAYS"
        grupo_badge  = "🔵 Operacional" if linha["Grupo"] == "Operacional" else "🟡 Observação"
        rank_txt     = f"Rank_op {linha['Rank_op']}" if "Rank_op" in linha and pd.notna(linha.get("Rank_op")) else ""
        s2_txt       = "🔷 **S2 ATIVO**" if linha["S2"] else "S2 inativo"
        if linha["Grupo"] == "Observação" and linha["S2"]:
            s2_txt = "S2 presente (observação — sem sinal automático)"
        st.markdown(
            f"**{grupo_badge}** &nbsp;|&nbsp;"
            f"**Preço:** R$ {linha['Preço']:.2f} &nbsp;|&nbsp;"
            f"**Score:** {linha['Score']} &nbsp;|&nbsp;"
            f"**Prob%:** {linha['Prob%']}% &nbsp;|&nbsp;"
            f"**Regime:** {regime_badge}"
            + (f" &nbsp;|&nbsp; **{rank_txt}**" if rank_txt else "")
            + f" &nbsp;|&nbsp; {s2_txt}"
        )
    ticker_sa = selecionado if selecionado.endswith(".SA") else selecionado + ".SA"
    plotar_detalhe(ticker_sa, selecionado)
