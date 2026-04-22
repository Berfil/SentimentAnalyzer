"""
Streamlit dashboard for Japanese brand sentiment analysis.

Run:
    streamlit run src/dashboard.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_FILE = DATA_DIR / "scraped_comments.csv"

st.set_page_config(
    page_title="日本語ブランド感情分析",
    page_icon="📊",
    layout="wide",
)

st.title("📊 日本語ブランド感情分析ダッシュボード")
st.caption("Japanese Brand Sentiment Analysis Dashboard")

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
    st.header(f"「{current_keyword}」の感情分析結果")

    counts = df["sentiment"].value_counts()
    total = len(df)
    pos = counts.get("positive", 0)
    neu = counts.get("neutral", 0)
    neg = counts.get("negative", 0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("総コメント数", total)
    k2.metric("ポジティブ 😊", f"{pos}  ({pos/total*100:.0f}%)")
    k3.metric("ニュートラル 😐", f"{neu}  ({neu/total*100:.0f}%)")
    k4.metric("ネガティブ 😞", f"{neg}  ({neg/total*100:.0f}%)")

    st.divider()

    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.subheader("感情の内訳 (Sentiment Breakdown)")
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
        st.subheader("信頼スコア分布 (Confidence Score Distribution)")
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
        st.subheader("感情トレンド (Sentiment Over Time)")
        trend = (
            df.groupby(["scraped_date", "sentiment"])
            .size()
            .reset_index(name="count")
        )
        total_by_date = df.groupby("scraped_date").size().reset_index(name="total")
        trend = trend.merge(total_by_date, on="scraped_date")
        trend["pct"] = trend["count"] / trend["total"] * 100
        fig_trend = px.line(
            trend,
            x="scraped_date",
            y="pct",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            markers=True,
            labels={"scraped_date": "Date", "pct": "% of comments"},
        )
        fig_trend.update_layout(margin=dict(t=10, b=10), yaxis_ticksuffix="%")
        st.plotly_chart(fig_trend, use_container_width=True)

    if "source" in df.columns:
        st.subheader("ソース別感情 (Sentiment by Source)")
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
    st.subheader("注目コメント (Notable Comments)")

    score_col = "score" if "score" in df.columns else None

    def show_top(sentiment: str, tab, n: int = 5):
        sub = df[df["sentiment"] == sentiment]
        if score_col:
            sub = sub.nlargest(n, score_col)
        else:
            sub = sub.head(n)
        if sub.empty:
            tab.info("コメントがありません。")
            return
        for _, row in sub.iterrows():
            with tab.container():
                title = row.get("article_title", row.get("product_name", ""))
                score_str = f"  `{row[score_col]:.2f}`" if score_col else ""
                st.markdown(f"**{title}**{score_str}")
                st.markdown(f"> {row['comment'][:300]}")
                st.divider()

    top_pos, top_neg, top_neu = st.tabs(["😊 ポジティブ Top 5", "😞 ネガティブ Top 5", "😐 ニュートラル Top 5"])
    show_top("positive", top_pos)
    show_top("negative", top_neg)
    show_top("neutral", top_neu)

    st.divider()
    st.subheader("全コメント一覧 (All Comments)")

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
    st.header("ブランド比較 / Brand Comparison")

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

            # ── Grouped bar chart
            st.subheader("感情分布比較 (Sentiment Distribution)")
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
                st.subheader("ポジティブ率トレンド比較 (Positive Sentiment Trend)")
                trend_rows = []
                for brand in selected_brands:
                    bdf = cdf[cdf["keyword"] == brand]
                    for date_val, grp in bdf.groupby("scraped_date"):
                        pos_pct = (grp["sentiment"] == "positive").sum() / len(grp) * 100
                        trend_rows.append({"日付": date_val, "ブランド": brand, "ポジティブ %": round(pos_pct, 1)})
                trend_cmp_df = pd.DataFrame(trend_rows)
                fig_trend_cmp = px.line(
                    trend_cmp_df,
                    x="日付",
                    y="ポジティブ %",
                    color="ブランド",
                    markers=True,
                    labels={"日付": "Date", "ポジティブ %": "Positive %", "ブランド": "Brand"},
                )
                fig_trend_cmp.update_layout(margin=dict(t=10, b=10), yaxis_ticksuffix="%")
                st.plotly_chart(fig_trend_cmp, use_container_width=True)
