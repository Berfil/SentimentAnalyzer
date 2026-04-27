"""
Streamlit dashboard for Japanese brand sentiment analysis.

Run:
    streamlit run src/dashboard.py
"""

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_FILE = DATA_DIR / "scraped_comments.csv"
ANNOTATIONS_FILE = DATA_DIR / "annotations.json"


def load_annotations() -> dict:
    if ANNOTATIONS_FILE.exists():
        return json.loads(ANNOTATIONS_FILE.read_text(encoding="utf-8"))
    return {}


def save_annotations(data: dict):
    ANNOTATIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

st.set_page_config(
    page_title="SentimentAggregator",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1a1a2e;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox select,
[data-testid="stSidebar"] .stNumberInput input {
    background: #16213e !important;
    border: 1px solid #2d3561 !important;
    color: #e2e8f0 !important;
    border-radius: 8px;
}
[data-testid="stSidebar"] hr {
    border-color: #2d3561 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #a5b4fc !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stSidebar"] .stButton button {
    background: #6366f1 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100%;
    padding: 0.6rem 1rem !important;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #4f46e5 !important;
}

/* ── KPI cards ── */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e8ecf4;
    border-radius: 12px;
    padding: 1.2rem 1.5rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stMetric"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #1a1a2e !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    gap: 0.5rem;
    border-bottom: 2px solid #e8ecf4;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: 0.5rem 1.2rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    border: none !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #6366f1 !important;
    border-bottom: 2px solid #6366f1 !important;
    font-weight: 600 !important;
}

/* ── Buttons (main area) ── */
.stButton button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.15s;
}

/* ── Divider ── */
hr { border-color: #e8ecf4 !important; margin: 1.5rem 0 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #e8ecf4;
    border-radius: 12px;
    overflow: hidden;
}

/* ── Alert / info boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #e8ecf4 !important;
    border-radius: 10px !important;
    background: white;
}
</style>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.25rem;">
  <span style="font-size:2rem;">📊</span>
  <div>
    <h1 style="margin:0; font-size:1.75rem; font-weight:700; color:#1a1a2e; line-height:1.2;">
      SentimentAggregator
    </h1>
    <p style="margin:0; font-size:0.85rem; color:#64748b;">
      Japanese brand sentiment monitoring &amp; analysis
    </p>
  </div>
</div>
<hr style="margin-top:1rem;">
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 新規分析 / New Analysis")
    keyword = st.text_input("キーワード (Keyword)", placeholder="例: 資生堂, モスバーガー")

    source = st.selectbox(
        "データソース (Source)",
        options=["yahoo", "cosme", "tabelog", "kakaku", "twitter", "all"],
        format_func=lambda x: {
            "yahoo": "Yahoo Japan News",
            "cosme": "@cosme (化粧品)",
            "tabelog": "Tabelog (レストラン)",
            "kakaku": "Kakaku.com (価格.com)",
            "twitter": "Twitter / X",
            "all": "すべて (All)",
        }[x],
    )

    col1, col2 = st.columns(2)
    with col1:
        max_articles = st.number_input("Yahoo 記事数", min_value=1, max_value=50, value=20)
    with col2:
        max_items = st.number_input(
            "製品/店舗数" if source not in ("yahoo", "twitter") else "（無効）",
            min_value=1, max_value=20, value=5,
            disabled=(source in ("yahoo", "twitter")),
            help="@cosme: products  |  Tabelog: restaurants  |  Kakaku: products",
        )

    if source in ("twitter", "all"):
        max_tweets = st.number_input("ツイート数 (Max tweets)", min_value=10, max_value=500, value=100, step=10)
        bearer_token = st.text_input("Twitter Bearer Token", type="password", help="Twitter API v2 Bearer Token from developer.twitter.com")
    else:
        max_tweets = 100
        bearer_token = ""

    output_file = st.text_input(
        "保存先 (Output file)",
        value=str(RESULTS_FILE.relative_to(ROOT)),
    )

    run_btn = st.button("▶ 分析開始 (Run Analysis)", type="primary", disabled=not keyword)

    st.divider()
    st.header("📂 既存データ読み込み")
    uploaded = st.file_uploader("CSV をアップロード", type="csv")
    load_default = st.button("デフォルトCSV を読み込む")

# ── Run analysis ──────────────────────────────────────────────────────────────
if run_btn and keyword:
    out_path = ROOT / output_file
    cmd = [
        sys.executable, str(ROOT / "src" / "analyse.py"),
        "--keyword", keyword,
        "--source", source,
        "--max_articles", str(max_articles),
        "--max_products", str(max_items),
        "--max_restaurants", str(max_items),
        "--max_tweets", str(max_tweets),
        "--output", str(out_path),
    ]
    if bearer_token:
        cmd += ["--bearer_token", bearer_token]

    with st.spinner(f"「{keyword}」を分析中... (this may take a few minutes)"):
        log_placeholder = st.empty()
        log_lines = []

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
        )
        for line in proc.stdout:
            log_lines.append(line.rstrip())
            log_placeholder.code("\n".join(log_lines[-20:]), language="")
        proc.wait()

    if proc.returncode == 0:
        st.success("✅ 分析完了！(Analysis complete)")
        if out_path.exists():
            full_df = pd.read_csv(out_path, encoding="utf-8-sig")
            # Show only results for this keyword if the column exists
            if "keyword" in full_df.columns:
                st.session_state["df"] = full_df[full_df["keyword"] == keyword].copy()
            else:
                st.session_state["df"] = full_df
            st.session_state["keyword"] = keyword
    else:
        st.error("❌ エラーが発生しました。上のログを確認してください。")

# ── Load existing data ────────────────────────────────────────────────────────
if uploaded is not None:
    st.session_state["df"] = pd.read_csv(uploaded, encoding="utf-8-sig")
    st.session_state["keyword"] = uploaded.name.replace(".csv", "")

if load_default and RESULTS_FILE.exists():
    loaded = pd.read_csv(RESULTS_FILE, encoding="utf-8-sig")
    st.session_state["df"] = loaded
    st.session_state["keyword"] = "全データ"
    st.session_state["full_df"] = loaded

# ── Display results ───────────────────────────────────────────────────────────
df: pd.DataFrame | None = st.session_state.get("df")
current_keyword: str = st.session_state.get("keyword", "")

if df is None or df.empty:
    st.info("左のサイドバーから分析を開始するか、既存のCSVを読み込んでください。\n\nUse the sidebar to run a new analysis or load an existing CSV.")
    st.stop()

# Ensure sentiment column exists
if "sentiment" not in df.columns:
    st.error("このCSVにはsentiment列がありません。analyse.pyで生成したCSVを使用してください。")
    st.stop()

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "neutral": "#95a5a6",
    "negative": "#e74c3c",
}

# If the full dataset is loaded, allow filtering by keyword
full_df: pd.DataFrame | None = st.session_state.get("full_df")
if full_df is not None and "keyword" in full_df.columns:
    available_keywords = sorted(full_df["keyword"].dropna().unique().tolist())
    selected_kw = st.selectbox(
        "キーワードで絞り込む (Filter by keyword)",
        options=["すべて (All)"] + available_keywords,
    )
    if selected_kw != "すべて (All)":
        df = full_df[full_df["keyword"] == selected_kw].copy()
        current_keyword = selected_kw
    else:
        df = full_df
        current_keyword = "全データ"
    st.session_state["df"] = df

tab_analysis, tab_compare = st.tabs(["📊 分析 / Analysis", "🆚 ブランド比較 / Brand Comparison"])

# ── Analysis tab ──────────────────────────────────────────────────────────────
with tab_analysis:
    st.markdown(f"""
    <h2 style="font-size:1.3rem; font-weight:600; color:#1a1a2e; margin-bottom:1rem;">
      「{current_keyword}」 — Sentiment Overview
    </h2>
    """, unsafe_allow_html=True)

    counts = df["sentiment"].value_counts()
    total = len(df)
    pos = counts.get("positive", 0)
    neu = counts.get("neutral", 0)
    neg = counts.get("negative", 0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Comments", total)
    k2.metric("Positive 😊", f"{pos}", delta=f"{pos/total*100:.0f}%")
    k3.metric("Neutral 😐", f"{neu}", delta=f"{neu/total*100:.0f}%", delta_color="off")
    k4.metric("Negative 😞", f"{neg}", delta=f"{neg/total*100:.0f}%", delta_color="inverse")

    st.divider()

    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.markdown("##### Sentiment Breakdown")
        pie_df = df["sentiment"].value_counts().reset_index()
        pie_df.columns = ["sentiment", "count"]
        fig_pie = px.pie(
            pie_df,
            names="sentiment",
            values="count",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.markdown("##### Confidence Score Distribution")
        if "score" in df.columns:
            fig_hist = px.histogram(
                df,
                x="score",
                color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                nbins=20,
                barmode="overlay",
                opacity=0.7,
                labels={"score": "Confidence Score", "count": "Comments"},
            )
            fig_hist.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("スコア列がありません。")

    if "scraped_date" in df.columns and df["scraped_date"].nunique() > 1:
        st.markdown("##### Sentiment Over Time")

        # ── Alert check ───────────────────────────────────────────────────────
        alert_threshold = st.sidebar.slider(
            "⚠️ Alert threshold (positive % drop)",
            min_value=5, max_value=30, value=10, step=5,
            help="Show a warning if positive sentiment drops by more than this % week-over-week",
        )
        dates_sorted = sorted(df["scraped_date"].dropna().unique())
        if len(dates_sorted) >= 2:
            cutoff = str(date.fromisoformat(dates_sorted[-1]) - timedelta(days=7))
            recent = df[df["scraped_date"] > cutoff]
            prior  = df[df["scraped_date"] <= cutoff]
            if not prior.empty and not recent.empty:
                recent_pos = (recent["sentiment"] == "positive").sum() / len(recent) * 100
                prior_pos  = (prior["sentiment"]  == "positive").sum() / len(prior)  * 100
                drop = prior_pos - recent_pos
                if drop >= alert_threshold:
                    st.warning(
                        f"⚠️ **Sentiment alert** — positive % dropped **{drop:.1f} points** "
                        f"({prior_pos:.1f}% → {recent_pos:.1f}%) over the past 7 days."
                    )
                elif recent_pos - prior_pos >= alert_threshold:
                    st.success(
                        f"📈 Positive sentiment rose **{recent_pos - prior_pos:.1f} points** "
                        f"({prior_pos:.1f}% → {recent_pos:.1f}%) over the past 7 days."
                    )

        # ── Build daily % + 7-day rolling average ────────────────────────────
        trend = (
            df.groupby(["scraped_date", "sentiment"])
            .size()
            .reset_index(name="count")
        )
        total_by_date = df.groupby("scraped_date").size().reset_index(name="total")
        trend = trend.merge(total_by_date, on="scraped_date")
        trend["pct"] = trend["count"] / trend["total"] * 100
        trend["scraped_date"] = pd.to_datetime(trend["scraped_date"])
        trend = trend.sort_values("scraped_date")

        fig_trend = go.Figure()
        for sentiment, color in SENTIMENT_COLORS.items():
            s = trend[trend["sentiment"] == sentiment].set_index("scraped_date")["pct"]
            s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).fillna(None)
            rolling = s.rolling(7, min_periods=1).mean()

            # Daily dots
            fig_trend.add_trace(go.Scatter(
                x=s.index, y=s.values,
                mode="markers", name=sentiment,
                marker=dict(color=color, size=7),
                legendgroup=sentiment,
            ))
            # Rolling average line
            fig_trend.add_trace(go.Scatter(
                x=rolling.index, y=rolling.values,
                mode="lines", name=f"{sentiment} (7-day avg)",
                line=dict(color=color, dash="dash", width=2),
                legendgroup=sentiment,
            ))

        # ── Annotations ───────────────────────────────────────────────────────
        annotations = load_annotations()
        kw_annotations = annotations.get(current_keyword, [])
        for ann in kw_annotations:
            fig_trend.add_vline(
                x=ann["date"], line_dash="dot", line_color="gray", line_width=1,
                annotation_text=ann["text"], annotation_position="top",
            )

        fig_trend.update_layout(
            margin=dict(t=10, b=10),
            yaxis_ticksuffix="%",
            yaxis_title="% of comments",
            xaxis_title="Date",
            legend_title="Sentiment",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # ── Add / delete annotation form ──────────────────────────────────────
        with st.expander("📌 イベントを追加 / Manage annotations"):
            ann_col1, ann_col2, ann_col3 = st.columns([1, 2, 1])
            with ann_col1:
                ann_date = st.date_input("Date", value=date.today())
            with ann_col2:
                ann_text = st.text_input("Event label", placeholder="e.g. New product launch")
            with ann_col3:
                st.write("")
                st.write("")
                if st.button("Add annotation") and ann_text:
                    annotations.setdefault(current_keyword, []).append(
                        {"date": str(ann_date), "text": ann_text}
                    )
                    save_annotations(annotations)
                    st.rerun()

            if kw_annotations:
                st.write("**Existing annotations:**")
                for i, ann in enumerate(kw_annotations):
                    c1, c2 = st.columns([5, 1])
                    c1.write(f"`{ann['date']}` — {ann['text']}")
                    if c2.button("Delete", key=f"del_ann_{i}"):
                        annotations[current_keyword].pop(i)
                        save_annotations(annotations)
                        st.rerun()

    if "source" in df.columns:
        st.markdown("##### Sentiment by Source")
        source_sent = df.groupby(["source", "sentiment"]).size().reset_index(name="count")
        fig_src = px.bar(
            source_sent,
            x="source",
            y="count",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            barmode="group",
            labels={"source": "Source", "count": "Comments"},
        )
        fig_src.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig_src, use_container_width=True)

    st.divider()
    st.markdown("##### Notable Comments")

    score_col = "score" if "score" in df.columns else None

    CARD_COLORS = {"positive": "#dcfce7", "neutral": "#f1f5f9", "negative": "#fee2e2"}
    CARD_BORDER  = {"positive": "#86efac", "neutral": "#cbd5e1", "negative": "#fca5a5"}
    CARD_LABEL   = {"positive": "😊 Positive", "neutral": "😐 Neutral", "negative": "😞 Negative"}

    def show_top(sentiment: str, tab, n: int = 5):
        sub = df[df["sentiment"] == sentiment]
        if score_col:
            sub = sub.nlargest(n, score_col)
        else:
            sub = sub.head(n)
        if sub.empty:
            tab.info("No comments found.")
            return
        bg = CARD_COLORS[sentiment]
        border = CARD_BORDER[sentiment]
        for _, row in sub.iterrows():
            title = row.get("article_title", row.get("product_name", ""))
            score_str = f"{row[score_col]:.2f}" if score_col else ""
            source_str = row.get("source", "")
            tab.markdown(f"""
<div style="background:{bg}; border:1px solid {border}; border-radius:10px;
            padding:1rem 1.25rem; margin-bottom:0.75rem;">
  <div style="display:flex; justify-content:space-between; align-items:center;
              margin-bottom:0.4rem;">
    <span style="font-size:0.75rem; font-weight:600; color:#475569;
                 text-transform:uppercase; letter-spacing:0.05em;">
      {source_str} · {title[:60]}
    </span>
    <span style="font-size:0.8rem; font-weight:700; color:#475569;">
      {score_str}
    </span>
  </div>
  <p style="margin:0; font-size:0.95rem; line-height:1.6; color:#1e293b;">
    {row['comment'][:300]}
  </p>
</div>
""", unsafe_allow_html=True)

    top_pos, top_neg, top_neu = st.tabs(["😊 Positive", "😞 Negative", "😐 Neutral"])
    show_top("positive", top_pos)
    show_top("negative", top_neg)
    show_top("neutral", top_neu)

    st.divider()
    st.markdown("##### All Comments")

    filter_sentiment = st.multiselect(
        "感情フィルタ (Filter by sentiment)",
        options=["positive", "neutral", "negative"],
        default=["positive", "neutral", "negative"],
    )

    filtered_df = df[df["sentiment"].isin(filter_sentiment)]
    display_cols = [c for c in ["article_title", "product_name", "sentiment", "score", "comment", "source"] if c in filtered_df.columns]
    st.dataframe(
        filtered_df[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=400,
        column_config={
            "sentiment": st.column_config.TextColumn("感情"),
            "score": st.column_config.NumberColumn("スコア", format="%.2f"),
            "comment": st.column_config.TextColumn("コメント", width="large"),
        },
    )

    st.caption(f"表示中: {len(filtered_df)} / {total} コメント")

    csv_bytes = filtered_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇ CSVをダウンロード (Download CSV)",
        data=csv_bytes,
        file_name=f"sentiment_{current_keyword}.csv",
        mime="text/csv",
    )

# ── Brand comparison tab ──────────────────────────────────────────────────────
with tab_compare:
    st.markdown("""
    <h2 style="font-size:1.3rem; font-weight:600; color:#1a1a2e; margin-bottom:1rem;">
      Brand Comparison
    </h2>
    """, unsafe_allow_html=True)

    compare_df = st.session_state.get("full_df") or df

    if "keyword" not in compare_df.columns or compare_df["keyword"].nunique() < 2:
        st.info(
            "複数のブランドデータが必要です。\n\n"
            "Load the full dataset via **デフォルトCSV を読み込む**, or run analyses for multiple brands first."
        )
    else:
        available_brands = sorted(compare_df["keyword"].dropna().unique().tolist())
        selected_brands = st.multiselect(
            "比較するブランドを選択 (Select brands to compare)",
            options=available_brands,
            default=available_brands[:min(4, len(available_brands))],
        )

        if len(selected_brands) < 2:
            st.info("2つ以上のブランドを選択してください。(Select at least 2 brands)")
        else:
            cdf = compare_df[compare_df["keyword"].isin(selected_brands)]

            # ── Summary metrics table
            stats = []
            for brand in selected_brands:
                bdf = cdf[cdf["keyword"] == brand]
                n = len(bdf)
                bcounts = bdf["sentiment"].value_counts()
                stats.append({
                    "ブランド": brand,
                    "総数": n,
                    "ポジティブ %": round(bcounts.get("positive", 0) / n * 100, 1),
                    "ニュートラル %": round(bcounts.get("neutral", 0) / n * 100, 1),
                    "ネガティブ %": round(bcounts.get("negative", 0) / n * 100, 1),
                    "平均スコア": round(bdf["score"].mean(), 3) if "score" in bdf.columns else "-",
                })

            stats_df = pd.DataFrame(stats)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("##### Sentiment Distribution")
            melted = stats_df.melt(
                id_vars="ブランド",
                value_vars=["ポジティブ %", "ニュートラル %", "ネガティブ %"],
                var_name="感情",
                value_name="%",
            )
            sentiment_key_map = {"ポジティブ %": "positive", "ニュートラル %": "neutral", "ネガティブ %": "negative"}
            melted["sentiment_key"] = melted["感情"].map(sentiment_key_map)

            fig_cmp = px.bar(
                melted,
                x="ブランド",
                y="%",
                color="sentiment_key",
                color_discrete_map=SENTIMENT_COLORS,
                barmode="group",
                text_auto=".1f",
                labels={"ブランド": "Brand", "%": "% of comments", "sentiment_key": "Sentiment"},
            )
            fig_cmp.update_traces(textposition="outside")
            fig_cmp.update_layout(margin=dict(t=20, b=20), yaxis_range=[0, 105])
            st.plotly_chart(fig_cmp, use_container_width=True)

            # ── Positive sentiment trend per brand (only if multi-date data exists)
            if "scraped_date" in cdf.columns and cdf["scraped_date"].nunique() > 1:
                st.markdown("##### Positive Sentiment Trend")
                fig_trend_cmp = go.Figure()
                palette = px.colors.qualitative.Set2

                for i, brand in enumerate(selected_brands):
                    bdf = cdf[cdf["keyword"] == brand].copy()
                    bdf["scraped_date"] = pd.to_datetime(bdf["scraped_date"])
                    daily = bdf.groupby("scraped_date").apply(
                        lambda g: (g["sentiment"] == "positive").sum() / len(g) * 100
                    ).rename("pct")
                    daily = daily.reindex(
                        pd.date_range(daily.index.min(), daily.index.max(), freq="D")
                    ).fillna(None)
                    rolling = daily.rolling(7, min_periods=1).mean()
                    color = palette[i % len(palette)]

                    fig_trend_cmp.add_trace(go.Scatter(
                        x=daily.index, y=daily.values,
                        mode="markers", name=brand,
                        marker=dict(color=color, size=6),
                        legendgroup=brand,
                    ))
                    fig_trend_cmp.add_trace(go.Scatter(
                        x=rolling.index, y=rolling.values,
                        mode="lines", name=f"{brand} (7-day avg)",
                        line=dict(color=color, dash="dash", width=2),
                        legendgroup=brand,
                    ))

                fig_trend_cmp.update_layout(
                    margin=dict(t=10, b=10),
                    yaxis_ticksuffix="%",
                    yaxis_title="Positive %",
                    xaxis_title="Date",
                )
                st.plotly_chart(fig_trend_cmp, use_container_width=True)
