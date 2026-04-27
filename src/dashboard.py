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

SENTIMENT_COLORS = {
    "positive": "#10b981",
    "neutral":  "#71717a",
    "negative": "#E11D48",
}
CARD_BG     = {"positive": "rgba(16,185,129,0.08)",  "neutral": "rgba(113,113,122,0.08)", "negative": "rgba(225,29,72,0.08)"}
CARD_BORDER = {"positive": "rgba(16,185,129,0.25)",  "neutral": "rgba(113,113,122,0.2)",  "negative": "rgba(225,29,72,0.25)"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_annotations() -> dict:
    if ANNOTATIONS_FILE.exists():
        return json.loads(ANNOTATIONS_FILE.read_text(encoding="utf-8"))
    return {}


def save_annotations(data: dict):
    ANNOTATIONS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def chart_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Satoshi, sans-serif", size=12, color="#8F8F8F"),
        margin=dict(t=16, b=16, l=8, r=8),
        legend=dict(
            bgcolor="rgba(13,13,13,0.8)",
            bordercolor="rgba(255,255,255,0.07)",
            borderwidth=1,
            font=dict(size=11, color="#8F8F8F"),
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            linecolor="rgba(255,255,255,0.07)",
            tickfont=dict(size=11, color="#8F8F8F"),
            title_font=dict(size=12, color="#8F8F8F"),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            linecolor="rgba(255,255,255,0.07)",
            tickfont=dict(size=11, color="#8F8F8F"),
            title_font=dict(size=12, color="#8F8F8F"),
            zeroline=False,
        ),
    )
    return fig


# ── Page config (called once) ─────────────────────────────────────────────────
st.set_page_config(
    page_title="KOKORO | Japanese Sentiment Engine",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://api.fontshare.com/v2/css?f[]=cabinet-grotesk@800,700,500&f[]=satoshi@400,500&display=swap');

html, body, [class*="css"], [class*="st-"] {
    font-family: 'Satoshi', sans-serif !important;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Cabinet Grotesk', sans-serif !important;
    letter-spacing: -0.02em !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }

.stApp {
    background-color: #050505;
    background-image: radial-gradient(rgba(255,255,255,0.04) 1px, transparent 1px);
    background-size: 32px 32px;
}

/* ── Sidebar (dashboard view only) ── */
[data-testid="stSidebar"] {
    background: #050505 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: #8F8F8F !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #0D0D0D !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] label {
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #525252 !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.06) !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #E11D48 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    width: 100% !important;
    padding: 0.7rem !important;
    transition: opacity 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover { opacity: 0.85 !important; }

/* ── KPI cards ── */
[data-testid="stMetric"] {
    background: rgba(13,13,13,0.8) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 1.25rem 1.5rem !important;
}
[data-testid="stMetric"] > label {
    font-size: 0.68rem !important; font-weight: 500 !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    color: #525252 !important;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important; font-weight: 800 !important;
    color: #ffffff !important; line-height: 1.1 !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(13,13,13,0.8);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 0.25rem; gap: 0.25rem; margin-bottom: 1.5rem;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 7px !important; padding: 0.45rem 1.1rem !important;
    font-size: 0.875rem !important; font-weight: 500 !important;
    color: #525252 !important; border: none !important; background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #E11D48 !important; color: white !important; font-weight: 700 !important;
}

/* ── Containers ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(13,13,13,0.7) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}

/* ── Inputs ── */
[data-baseweb="select"] > div {
    background: rgba(13,13,13,0.8) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important; color: white !important;
}
.stTextInput input, .stNumberInput input {
    background: rgba(13,13,13,0.8) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: white !important; border-radius: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 0.875rem !important; transition: opacity 0.15s !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important; overflow: hidden !important;
    background: rgba(13,13,13,0.8) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 10px !important; border-left-width: 4px !important;
    background: rgba(13,13,13,0.8) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    background: rgba(13,13,13,0.7) !important;
}

/* ── Multiselect tags ── */
[data-baseweb="tag"] {
    background: rgba(225,29,72,0.15) !important;
    border: 1px solid rgba(225,29,72,0.3) !important;
    border-radius: 6px !important; color: #E11D48 !important;
}

/* ── Misc ── */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.75rem 0 !important; }
[data-testid="stDownloadButton"] > button {
    border-radius: 8px !important; font-weight: 600 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(13,13,13,0.8) !important; color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def show_landing():
    # Hide sidebar on landing
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────────
    st.html("""
<nav style="display:flex; align-items:center; justify-content:space-between;
            padding:1.5rem 3rem; max-width:1200px; margin:0 auto;">
  <div style="display:flex; align-items:center; gap:0.6rem;">
    <div style="width:28px; height:28px; background:white; display:flex;
                align-items:center; justify-content:center;">
      <div style="width:12px; height:12px; border-radius:50%; background:#E11D48;"></div>
    </div>
    <span style="font-size:1rem; font-weight:800; color:#ffffff;
                 letter-spacing:-0.01em; font-family:'Cabinet Grotesk',sans-serif;">
      KOKORO
    </span>
  </div>
  <div style="display:flex; align-items:center; gap:2.5rem;">
    <span style="font-size:0.875rem; color:#525252; cursor:default;">Features</span>
    <span style="font-size:0.875rem; color:#525252; cursor:default;">Sources</span>
  </div>
</nav>
""")

    # ── Hero ──────────────────────────────────────────────────────────────────
    # Glow blob
    st.html("""
<div style="position:fixed; top:-20%; right:-10%; width:600px; height:600px;
            background:rgba(225,29,72,0.06); filter:blur(120px); border-radius:50%;
            pointer-events:none; z-index:0;"></div>
<div style="position:fixed; bottom:-10%; left:-5%; width:400px; height:400px;
            background:rgba(30,30,30,0.4); filter:blur(100px); border-radius:50%;
            pointer-events:none; z-index:0;"></div>
""")

    st.html("""
<div style="text-align:center; padding:5rem 2rem 2rem; max-width:900px; margin:0 auto;
            position:relative; z-index:1;">
  <div style="display:inline-flex; align-items:center; gap:0.5rem;
              padding:0.3rem 0.9rem; border-radius:999px;
              border:1px solid rgba(225,29,72,0.3); background:rgba(225,29,72,0.06);
              margin-bottom:1.5rem;">
    <span style="width:6px; height:6px; border-radius:50%; background:#E11D48;
                 display:inline-block;"></span>
    <span style="font-size:0.65rem; font-weight:700; text-transform:uppercase;
                 letter-spacing:0.12em; color:#E11D48;">Live Sentiment Engine</span>
  </div>
  <h1 style="font-size:clamp(3rem,7vw,5.5rem); font-weight:800; line-height:1.05;
             letter-spacing:-0.03em; color:#ffffff; margin:0 0 1.5rem;
             font-family:'Cabinet Grotesk',sans-serif;">
    Decode the pulse of<br>
    <span style="background:linear-gradient(90deg,#ffffff 0%,#8F8F8F 55%,#E11D48 100%);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                 background-clip:text;">
      Japanese Social Media.
    </span>
  </h1>
  <p style="font-size:1.05rem; color:#525252; max-width:560px; margin:0 auto 3rem;
            line-height:1.75; font-family:'Satoshi',sans-serif;">
    Analyze brand sentiment across Yahoo Japan, @cosme, Tabelog and Kakaku.com
    with an NLP engine fine-tuned specifically for the Japanese market.
  </p>
</div>
""")

    # CTA buttons
    _, c1, c2, _ = st.columns([2, 1, 1, 2])
    with c1:
        if st.button("▶  Launch Dashboard", type="primary", use_container_width=True):
            st.session_state["page"] = "dashboard"
            st.rerun()
    with c2:
        st.link_button(
            "View on GitHub",
            "https://github.com/Berfil/SentimentAggregator",
            use_container_width=True,
        )

    # ── Mock dashboard preview ─────────────────────────────────────────────────
    st.html("""
<div style="max-width:900px; margin:3rem auto 0; position:relative; z-index:1;">
  <div style="position:absolute; inset:-2px; background:linear-gradient(135deg,
              rgba(225,29,72,0.15), rgba(100,100,100,0.1));
              border-radius:20px; filter:blur(8px); opacity:0.6;"></div>
  <div style="position:relative; background:rgba(13,13,13,0.9);
              border:1px solid rgba(255,255,255,0.07); border-radius:18px;
              overflow:hidden; backdrop-filter:blur(12px);">
    <div style="background:rgba(255,255,255,0.03); border-bottom:1px solid rgba(255,255,255,0.05);
                padding:0.9rem 1.5rem; display:flex; align-items:center; justify-content:space-between;">
      <div style="display:flex; gap:0.5rem;">
        <div style="width:12px;height:12px;border-radius:50%;background:#3a3a3a;"></div>
        <div style="width:12px;height:12px;border-radius:50%;background:#3a3a3a;"></div>
        <div style="width:12px;height:12px;border-radius:50%;background:#3a3a3a;"></div>
      </div>
      <span style="font-size:0.65rem; color:#525252; font-family:monospace;
                   letter-spacing:0.1em;">DASHBOARD // KOKORO_OS_v2.0</span>
      <div style="width:60px;"></div>
    </div>
    <div style="padding:1.5rem; display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:1rem;">
      <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05);
                  border-radius:12px; padding:1rem;">
        <p style="margin:0 0 0.25rem; font-size:0.6rem; text-transform:uppercase;
                  letter-spacing:0.1em; color:#525252;">Total Comments</p>
        <p style="margin:0; font-size:1.75rem; font-weight:800; color:#fff;
                  font-family:'Cabinet Grotesk',sans-serif;">4,821</p>
      </div>
      <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05);
                  border-radius:12px; padding:1rem;">
        <p style="margin:0 0 0.25rem; font-size:0.6rem; text-transform:uppercase;
                  letter-spacing:0.1em; color:#525252;">Positive</p>
        <p style="margin:0; font-size:1.75rem; font-weight:800; color:#10b981;
                  font-family:'Cabinet Grotesk',sans-serif;">63%</p>
      </div>
      <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05);
                  border-radius:12px; padding:1rem;">
        <p style="margin:0 0 0.25rem; font-size:0.6rem; text-transform:uppercase;
                  letter-spacing:0.1em; color:#525252;">Neutral</p>
        <p style="margin:0; font-size:1.75rem; font-weight:800; color:#71717a;
                  font-family:'Cabinet Grotesk',sans-serif;">24%</p>
      </div>
      <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05);
                  border-radius:12px; padding:1rem;">
        <p style="margin:0 0 0.25rem; font-size:0.6rem; text-transform:uppercase;
                  letter-spacing:0.1em; color:#525252;">Negative</p>
        <p style="margin:0; font-size:1.75rem; font-weight:800; color:#E11D48;
                  font-family:'Cabinet Grotesk',sans-serif;">13%</p>
      </div>
    </div>
    <div style="padding:0 1.5rem 1.5rem;">
      <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05);
                  border-radius:12px; padding:1.25rem;">
        <p style="margin:0 0 1rem; font-size:0.6rem; text-transform:uppercase;
                  letter-spacing:0.1em; color:#525252;">Sentiment Over Time</p>
        <div style="display:flex; align-items:flex-end; gap:4px; height:80px;">
          <div style="background:rgba(225,29,72,0.4);  flex:1; height:40%; border-radius:3px 3px 0 0;"></div>
          <div style="background:rgba(225,29,72,0.5);  flex:1; height:55%; border-radius:3px 3px 0 0;"></div>
          <div style="background:rgba(225,29,72,0.4);  flex:1; height:35%; border-radius:3px 3px 0 0;"></div>
          <div style="background:rgba(113,113,122,0.5);flex:1; height:50%; border-radius:3px 3px 0 0;"></div>
          <div style="background:rgba(16,185,129,0.5); flex:1; height:70%; border-radius:3px 3px 0 0;"></div>
          <div style="background:rgba(16,185,129,0.7); flex:1; height:85%; border-radius:3px 3px 0 0;"></div>
          <div style="background:rgba(16,185,129,0.9); flex:1; height:95%; border-radius:3px 3px 0 0;"></div>
          <div style="background:rgba(16,185,129,0.8); flex:1; height:80%; border-radius:3px 3px 0 0;"></div>
        </div>
      </div>
    </div>
  </div>
</div>
""")

    # ── Stats row ─────────────────────────────────────────────────────────────
    st.html("""
<div style="max-width:900px; margin:5rem auto 0; display:grid;
            grid-template-columns:1fr 1fr 1fr; gap:1px;
            background:rgba(255,255,255,0.06); border-radius:16px; overflow:hidden;
            border:1px solid rgba(255,255,255,0.06);">
  <div style="background:#050505; padding:2.5rem; text-align:center;">
    <p style="margin:0 0 0.3rem; font-size:2.5rem; font-weight:800; color:#E11D48;
              font-family:'Cabinet Grotesk',sans-serif; letter-spacing:-0.03em;">4</p>
    <p style="margin:0; font-size:0.8rem; color:#525252;">Data sources</p>
  </div>
  <div style="background:#050505; padding:2.5rem; text-align:center;
              border-left:1px solid rgba(255,255,255,0.06);
              border-right:1px solid rgba(255,255,255,0.06);">
    <p style="margin:0 0 0.3rem; font-size:2.5rem; font-weight:800; color:#ffffff;
              font-family:'Cabinet Grotesk',sans-serif; letter-spacing:-0.03em;">ja</p>
    <p style="margin:0; font-size:0.8rem; color:#525252;">Japanese NLP model</p>
  </div>
  <div style="background:#050505; padding:2.5rem; text-align:center;">
    <p style="margin:0 0 0.3rem; font-size:2.5rem; font-weight:800; color:#10b981;
              font-family:'Cabinet Grotesk',sans-serif; letter-spacing:-0.03em;">&#8734;</p>
    <p style="margin:0; font-size:0.8rem; color:#525252;">Daily monitoring</p>
  </div>
</div>
""")

    # ── Sources section ───────────────────────────────────────────────────────
    st.html("""
<div style="max-width:900px; margin:5rem auto 0;">
  <p style="font-size:0.65rem; font-weight:700; text-transform:uppercase;
            letter-spacing:0.12em; color:#525252; margin-bottom:1rem;">
    Data Sources
  </p>
  <h2 style="font-size:2.5rem; font-weight:800; color:#ffffff; margin:0 0 0.75rem;
             font-family:'Cabinet Grotesk',sans-serif; letter-spacing:-0.02em;">
    The platforms that matter<br>
    <span style="color:#525252;">in Japan.</span>
  </h2>
  <p style="color:#525252; font-size:0.9rem; line-height:1.7; margin:0 0 3rem;
            max-width:480px;">
    Built for the Japanese internet. Not a generic global tool &#8212; every scraper
    is written specifically for the sites Japanese consumers actually use.
  </p>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;">
    <div style="background:rgba(13,13,13,0.8); border:1px solid rgba(255,255,255,0.07);
                border-radius:16px; padding:1.75rem; backdrop-filter:blur(12px);">
      <div style="font-size:1.5rem; margin-bottom:0.75rem;">&#128240;</div>
      <h4 style="margin:0 0 0.4rem; font-size:1rem; font-weight:700; color:#fff;
                 font-family:'Cabinet Grotesk',sans-serif;">Yahoo Japan News</h4>
      <p style="margin:0; font-size:0.8rem; color:#525252; line-height:1.6;">
        News article comment sections &#8212; the pulse of public opinion on current events and brands.
      </p>
    </div>
    <div style="background:rgba(13,13,13,0.8); border:1px solid rgba(255,255,255,0.07);
                border-radius:16px; padding:1.75rem; backdrop-filter:blur(12px);
                transform:translateY(12px);">
      <div style="font-size:1.5rem; margin-bottom:0.75rem;">&#128132;</div>
      <h4 style="margin:0 0 0.4rem; font-size:1rem; font-weight:700; color:#fff;
                 font-family:'Cabinet Grotesk',sans-serif;">@cosme</h4>
      <p style="margin:0; font-size:0.8rem; color:#525252; line-height:1.6;">
        Japan&#8217;s largest beauty review platform. Essential for cosmetics and skincare brands.
      </p>
    </div>
    <div style="background:rgba(13,13,13,0.8); border:1px solid rgba(255,255,255,0.07);
                border-radius:16px; padding:1.75rem; backdrop-filter:blur(12px);">
      <div style="font-size:1.5rem; margin-bottom:0.75rem;">&#127869;</div>
      <h4 style="margin:0 0 0.4rem; font-size:1rem; font-weight:700; color:#fff;
                 font-family:'Cabinet Grotesk',sans-serif;">Tabelog</h4>
      <p style="margin:0; font-size:0.8rem; color:#525252; line-height:1.6;">
        Japan&#8217;s definitive restaurant review site. Critical for F&amp;B and hospitality brands.
      </p>
    </div>
    <div style="background:rgba(13,13,13,0.8); border:1px solid rgba(255,255,255,0.07);
                border-radius:16px; padding:1.75rem; backdrop-filter:blur(12px);
                transform:translateY(12px);">
      <div style="font-size:1.5rem; margin-bottom:0.75rem;">&#128722;</div>
      <h4 style="margin:0 0 0.4rem; font-size:1rem; font-weight:700; color:#fff;
                 font-family:'Cabinet Grotesk',sans-serif;">Kakaku.com</h4>
      <p style="margin:0; font-size:0.8rem; color:#525252; line-height:1.6;">
        Japan&#8217;s top price comparison and product review site. Key for electronics and tech.
      </p>
    </div>
  </div>
</div>
""")

    # ── CTA banner ────────────────────────────────────────────────────────────
    st.html("""
<div style="max-width:900px; margin:7rem auto 2rem;">
  <div style="background:#0D0D0D; border:1px solid rgba(255,255,255,0.07);
              border-radius:28px; padding:4rem; text-align:center;
              position:relative; overflow:hidden;">
    <div style="position:absolute; top:0; right:0; width:300px; height:300px;
                background:rgba(225,29,72,0.06); filter:blur(80px); border-radius:50%;
                pointer-events:none;"></div>
    <h2 style="font-size:2.5rem; font-weight:800; color:#ffffff; margin:0 0 1rem;
               font-family:'Cabinet Grotesk',sans-serif; letter-spacing:-0.02em;
               position:relative;">
      Ready to listen to Japan?
    </h2>
    <p style="color:#525252; font-size:0.9rem; margin:0 0 2.5rem; line-height:1.7;
              position:relative;">
      Open the dashboard and start monitoring your brand sentiment today.
    </p>
  </div>
</div>
""")

    _, cc, _ = st.columns([2, 1, 2])
    with cc:
        if st.button("▶  Launch Dashboard", key="cta_btn", type="primary", use_container_width=True):
            st.session_state["page"] = "dashboard"
            st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.html("""
<div style="max-width:900px; margin:4rem auto 2rem;
            border-top:1px solid rgba(255,255,255,0.06); padding-top:2rem;
            display:flex; justify-content:space-between; align-items:center;">
  <div style="display:flex; align-items:center; gap:0.5rem;">
    <div style="width:18px; height:18px; background:white; display:flex;
                align-items:center; justify-content:center;">
      <div style="width:8px; height:8px; border-radius:50%; background:#E11D48;"></div>
    </div>
    <span style="font-size:0.8rem; font-weight:800; color:#ffffff;
                 font-family:'Cabinet Grotesk',sans-serif;">KOKORO</span>
  </div>
  <span style="font-size:0.65rem; color:#525252; font-family:monospace;
               letter-spacing:0.08em;">35.6762&#176; N, 139.6503&#176; E // TOKYO</span>
</div>
""")


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def show_dashboard():
    st.markdown("""
    <style>
    .block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding:1.5rem 1rem 1.25rem; border-bottom:1px solid rgba(255,255,255,0.06);
                    margin-bottom:1.25rem;">
          <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.3rem;">
            <div style="width:24px; height:24px; background:white; display:flex;
                        align-items:center; justify-content:center; flex-shrink:0;">
              <div style="width:10px; height:10px; border-radius:50%; background:#E11D48;"></div>
            </div>
            <span style="font-size:0.9rem; font-weight:800; color:#ffffff;
                         font-family:'Cabinet Grotesk',sans-serif;">KOKORO</span>
          </div>
          <p style="margin:0; font-size:0.6rem; color:#525252; padding-left:2.1rem;
                    text-transform:uppercase; letter-spacing:0.1em;">Sentiment Engine</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**New Analysis**")
        keyword = st.text_input("Keyword", placeholder="e.g. 資生堂, モスバーガー",
                                label_visibility="collapsed")
        st.caption("Brand name, product, or campaign keyword")

        source = st.selectbox("Source", options=["yahoo", "cosme", "tabelog", "kakaku", "twitter", "all"],
            format_func=lambda x: {"yahoo": "📰  Yahoo Japan News", "cosme": "💄  @cosme",
                "tabelog": "🍽️  Tabelog", "kakaku": "🛒  Kakaku.com",
                "twitter": "𝕏  Twitter / X", "all": "🌐  All sources"}[x])

        col1, col2 = st.columns(2)
        with col1:
            max_articles = st.number_input("Articles", min_value=1, max_value=50, value=20)
        with col2:
            max_items = st.number_input("Items", min_value=1, max_value=20, value=5,
                                        disabled=(source in ("yahoo", "twitter")))

        if source in ("twitter", "all"):
            max_tweets = st.number_input("Max tweets", min_value=10, max_value=500, value=100, step=10)
            bearer_token = st.text_input("Bearer Token", type="password")
        else:
            max_tweets, bearer_token = 100, ""

        output_file = st.text_input("Output file", value=str(RESULTS_FILE.relative_to(ROOT)))
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        run_btn = st.button("▶  Run Analysis", type="primary", disabled=not keyword)

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        st.markdown("""<p style="font-size:0.65rem; font-weight:500; text-transform:uppercase;
            letter-spacing:0.1em; color:#525252 !important; margin:0 0 0.5rem;">Load Data</p>""",
            unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
        load_default = st.button("Load default CSV")

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        alert_threshold = st.slider("Alert threshold (positive % drop)",
                                    min_value=5, max_value=30, value=10, step=5)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        if st.button("← Back to Home"):
            st.session_state["page"] = "landing"
            st.rerun()

    # ── Run analysis ──────────────────────────────────────────────────────────
    if run_btn and keyword:
        out_path = ROOT / output_file
        cmd = [sys.executable, str(ROOT / "src" / "analyse.py"),
               "--keyword", keyword, "--source", source,
               "--max_articles", str(max_articles), "--max_products", str(max_items),
               "--max_restaurants", str(max_items), "--max_tweets", str(max_tweets),
               "--output", str(out_path)]
        if bearer_token:
            cmd += ["--bearer_token", bearer_token]

        with st.spinner(f"Analysing「{keyword}」…"):
            log_box = st.empty()
            log_lines: list[str] = []
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
            for line in proc.stdout:
                log_lines.append(line.rstrip())
                log_box.code("\n".join(log_lines[-20:]), language="")
            proc.wait()

        if proc.returncode == 0:
            st.success("Analysis complete.")
            if out_path.exists():
                full_df = pd.read_csv(out_path, encoding="utf-8-sig")
                st.session_state["df"] = (
                    full_df[full_df["keyword"] == keyword].copy()
                    if "keyword" in full_df.columns else full_df)
                st.session_state["keyword"] = keyword
        else:
            st.error("Something went wrong — check the log above.")

    if uploaded is not None:
        st.session_state["df"] = pd.read_csv(uploaded, encoding="utf-8-sig")
        st.session_state["keyword"] = uploaded.name.replace(".csv", "")

    if load_default and RESULTS_FILE.exists():
        loaded = pd.read_csv(RESULTS_FILE, encoding="utf-8-sig")
        st.session_state["df"] = loaded
        st.session_state["keyword"] = "All data"
        st.session_state["full_df"] = loaded

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:2rem;">
      <div style="display:inline-flex; align-items:center; gap:0.5rem; padding:0.25rem 0.75rem;
                  border-radius:999px; border:1px solid rgba(225,29,72,0.25);
                  background:rgba(225,29,72,0.05); margin-bottom:1rem;">
        <span style="width:6px;height:6px;border-radius:50%;background:#E11D48;
                     display:inline-block;"></span>
        <span style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                     letter-spacing:0.12em;color:#E11D48;">Live Sentiment Engine</span>
      </div>
      <h1 style="margin:0;font-size:2.2rem;font-weight:800;color:#ffffff;
                 letter-spacing:-0.03em;line-height:1.1;
                 font-family:'Cabinet Grotesk',sans-serif;">
        Brand Sentiment Monitor
      </h1>
      <p style="margin:0.5rem 0 0;font-size:0.875rem;color:#525252;">
        Japanese social media · real-time analysis
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Empty state ───────────────────────────────────────────────────────────
    df: pd.DataFrame | None = st.session_state.get("df")
    current_keyword: str = st.session_state.get("keyword", "")

    if df is None or df.empty:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    padding:6rem 2rem;text-align:center;">
          <div style="width:64px;height:64px;background:rgba(13,13,13,0.8);
                      border:1px solid rgba(255,255,255,0.07);border-radius:16px;
                      display:flex;align-items:center;justify-content:center;
                      font-size:1.8rem;margin-bottom:1.5rem;">📊</div>
          <h2 style="margin:0 0 0.5rem;font-size:1.5rem;font-weight:800;color:#ffffff;
                     font-family:'Cabinet Grotesk',sans-serif;letter-spacing:-0.02em;">
            No data loaded yet
          </h2>
          <p style="margin:0;color:#525252;max-width:380px;line-height:1.7;font-size:0.875rem;">
            Enter a keyword in the sidebar and click
            <strong style="color:#8F8F8F;">Run Analysis</strong>
            to scrape and analyse sentiment, or load an existing CSV.
          </p>
        </div>
        """, unsafe_allow_html=True)
        return

    if "sentiment" not in df.columns:
        st.error("This CSV has no sentiment column. Use a file generated by analyse.py.")
        return

    # ── Keyword filter ────────────────────────────────────────────────────────
    full_df: pd.DataFrame | None = st.session_state.get("full_df")
    if full_df is not None and "keyword" in full_df.columns:
        available_keywords = sorted(full_df["keyword"].dropna().unique().tolist())
        selected_kw = st.selectbox("Filter by keyword", options=["All"] + available_keywords)
        if selected_kw != "All":
            df = full_df[full_df["keyword"] == selected_kw].copy()
            current_keyword = selected_kw
        else:
            df = full_df
            current_keyword = "All data"
        st.session_state["df"] = df

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_analysis, tab_compare = st.tabs(["📊  Analysis", "🆚  Brand Comparison"])

    # ── Analysis tab ──────────────────────────────────────────────────────────
    with tab_analysis:
        counts = df["sentiment"].value_counts()
        total  = len(df)
        pos = counts.get("positive", 0)
        neu = counts.get("neutral",  0)
        neg = counts.get("negative", 0)

        st.markdown(f"""<p style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
            letter-spacing:0.1em;color:#525252;margin-bottom:0.75rem;">{current_keyword}</p>""",
            unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total comments", f"{total:,}")
        k2.metric("Positive 😊",    f"{pos:,}",  delta=f"{pos/total*100:.1f}%")
        k3.metric("Neutral 😐",     f"{neu:,}",  delta=f"{neu/total*100:.1f}%",  delta_color="off")
        k4.metric("Negative 😞",    f"{neg:,}",  delta=f"{neg/total*100:.1f}%",  delta_color="inverse")

        st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

        col_pie, col_hist = st.columns(2)

        with col_pie:
            with st.container(border=True):
                st.markdown("**Sentiment breakdown**")
                pie_df = df["sentiment"].value_counts().reset_index()
                pie_df.columns = ["sentiment", "count"]
                fig_pie = px.pie(pie_df, names="sentiment", values="count",
                                 color="sentiment", color_discrete_map=SENTIMENT_COLORS, hole=0.5)
                fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                                      textfont=dict(size=13))
                fig_pie.update_layout(showlegend=False)
                chart_layout(fig_pie, height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

        with col_hist:
            with st.container(border=True):
                st.markdown("**Confidence score distribution**")
                if "score" in df.columns:
                    fig_hist = px.histogram(df, x="score", color="sentiment",
                        color_discrete_map=SENTIMENT_COLORS, nbins=20,
                        barmode="overlay", opacity=0.75,
                        labels={"score": "Confidence", "count": "Comments"})
                    chart_layout(fig_hist, height=300)
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.info("No score column in this dataset.")

        if "source" in df.columns and df["source"].nunique() > 1:
            with st.container(border=True):
                st.markdown("**Comments by source**")
                src_df = df.groupby(["source", "sentiment"]).size().reset_index(name="count")
                fig_src = px.bar(src_df, x="source", y="count", color="sentiment",
                    color_discrete_map=SENTIMENT_COLORS, barmode="group",
                    labels={"source": "Source", "count": "Comments"})
                chart_layout(fig_src, height=280)
                st.plotly_chart(fig_src, use_container_width=True)

        if "scraped_date" in df.columns and df["scraped_date"].nunique() > 1:
            with st.container(border=True):
                st.markdown("**Sentiment over time**")

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
                            st.warning(f"⚠️ Positive sentiment dropped **{drop:.1f} pts** "
                                       f"({prior_pos:.1f}% → {recent_pos:.1f}%) over the past 7 days.")
                        elif recent_pos - prior_pos >= alert_threshold:
                            st.success(f"📈 Positive sentiment rose **{recent_pos-prior_pos:.1f} pts** "
                                       f"({prior_pos:.1f}% → {recent_pos:.1f}%) over the past 7 days.")

                trend = df.groupby(["scraped_date","sentiment"]).size().reset_index(name="count")
                totals = df.groupby("scraped_date").size().reset_index(name="total")
                trend = trend.merge(totals, on="scraped_date")
                trend["pct"] = trend["count"] / trend["total"] * 100
                trend["scraped_date"] = pd.to_datetime(trend["scraped_date"])
                trend = trend.sort_values("scraped_date")

                fig_trend = go.Figure()
                for sentiment, color in SENTIMENT_COLORS.items():
                    s = trend[trend["sentiment"]==sentiment].set_index("scraped_date")["pct"]
                    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).fillna(None)
                    rolling = s.rolling(7, min_periods=1).mean()
                    fig_trend.add_trace(go.Scatter(x=s.index, y=s.values, mode="markers",
                        name=sentiment, marker=dict(color=color, size=7, opacity=0.5),
                        legendgroup=sentiment))
                    fig_trend.add_trace(go.Scatter(x=rolling.index, y=rolling.values, mode="lines",
                        name=f"{sentiment} (7d avg)", line=dict(color=color, width=2.5),
                        legendgroup=sentiment))

                annotations = load_annotations()
                for ann in annotations.get(current_keyword, []):
                    fig_trend.add_vline(x=ann["date"], line_dash="dot",
                        line_color="rgba(255,255,255,0.15)", line_width=1,
                        annotation_text=ann["text"], annotation_position="top",
                        annotation_font=dict(size=11, color="#525252"))

                fig_trend.update_layout(yaxis_ticksuffix="%", yaxis_title="% of comments")
                chart_layout(fig_trend, height=340)
                st.plotly_chart(fig_trend, use_container_width=True)

                with st.expander("📌  Manage event annotations"):
                    a1, a2, a3 = st.columns([1, 2, 1])
                    ann_date = a1.date_input("Date", value=date.today())
                    ann_text = a2.text_input("Label", placeholder="e.g. Product launch")
                    a3.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
                    if a3.button("Add") and ann_text:
                        annotations.setdefault(current_keyword, []).append(
                            {"date": str(ann_date), "text": ann_text})
                        save_annotations(annotations)
                        st.rerun()
                    for i, ann in enumerate(annotations.get(current_keyword, [])):
                        c1, c2 = st.columns([6, 1])
                        c1.markdown(f"`{ann['date']}` — {ann['text']}")
                        if c2.button("✕", key=f"del_{i}"):
                            annotations[current_keyword].pop(i)
                            save_annotations(annotations)
                            st.rerun()

        # Notable comments
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown("""<p style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
            letter-spacing:0.1em;color:#525252;margin-bottom:0.75rem;">Notable comments</p>""",
            unsafe_allow_html=True)

        score_col = "score" if "score" in df.columns else None

        def comment_card(row: pd.Series, sentiment: str) -> str:
            bg     = CARD_BG[sentiment]
            border = CARD_BORDER[sentiment]
            color  = SENTIMENT_COLORS[sentiment]
            label  = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}[sentiment]
            title  = str(row.get("article_title") or row.get("product_name") or "")[:70]
            source = str(row.get("source") or "")
            score  = f"{row[score_col]:.2f}" if score_col else ""
            meta   = f"{source}  ·  {title}" if title else source
            text   = str(row["comment"])[:320]
            return f"""
<div style="background:{bg};border:1px solid {border};border-radius:12px;
            padding:1rem 1.25rem;margin-bottom:0.6rem;backdrop-filter:blur(12px);">
  <div style="display:flex;justify-content:space-between;align-items:center;
              margin-bottom:0.6rem;gap:0.5rem;flex-wrap:wrap;">
    <div style="display:flex;align-items:center;gap:0.5rem;">
      <span style="color:{color};border:1px solid {border};font-size:0.62rem;
                   font-weight:700;padding:0.12rem 0.5rem;border-radius:999px;
                   text-transform:uppercase;letter-spacing:0.08em;background:rgba(0,0,0,0.3);">
        {label}
      </span>
      <span style="font-size:0.75rem;color:#525252;font-weight:500;">{meta}</span>
    </div>
    <span style="font-size:0.75rem;font-weight:700;color:#8F8F8F;">{score}</span>
  </div>
  <p style="margin:0;font-size:0.88rem;line-height:1.7;color:#d4d4d4;">{text}</p>
</div>"""

        def show_top(sentiment: str, tab, n: int = 5):
            sub = df[df["sentiment"] == sentiment]
            sub = sub.nlargest(n, score_col) if score_col else sub.head(n)
            if sub.empty:
                tab.info("No comments found.")
                return
            tab.markdown("".join(comment_card(row, sentiment) for _, row in sub.iterrows()),
                         unsafe_allow_html=True)

        t_pos, t_neg, t_neu = st.tabs(["😊  Positive", "😞  Negative", "😐  Neutral"])
        show_top("positive", t_pos)
        show_top("negative", t_neg)
        show_top("neutral",  t_neu)

        # Full table
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            tc1, tc2 = st.columns([3, 1])
            tc1.markdown("**All comments**")
            filter_sent = tc2.multiselect("Filter",
                options=["positive", "neutral", "negative"],
                default=["positive", "neutral", "negative"],
                label_visibility="collapsed")
            filtered_df = df[df["sentiment"].isin(filter_sent)]
            display_cols = [c for c in ["article_title","sentiment","score","comment","source","scraped_date"]
                            if c in filtered_df.columns]
            st.dataframe(filtered_df[display_cols].reset_index(drop=True),
                use_container_width=True, height=380,
                column_config={
                    "article_title": st.column_config.TextColumn("Title", width="medium"),
                    "sentiment":     st.column_config.TextColumn("Sentiment"),
                    "score":         st.column_config.NumberColumn("Score", format="%.2f"),
                    "comment":       st.column_config.TextColumn("Comment", width="large"),
                    "source":        st.column_config.TextColumn("Source"),
                    "scraped_date":  st.column_config.TextColumn("Date"),
                })
            c1, c2 = st.columns([3, 1])
            c1.caption(f"{len(filtered_df):,} of {total:,} comments")
            csv_bytes = filtered_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            c2.download_button("⬇  Download CSV", data=csv_bytes,
                file_name=f"sentiment_{current_keyword}.csv", mime="text/csv")

    # ── Brand comparison tab ───────────────────────────────────────────────────
    with tab_compare:
        compare_df = st.session_state.get("full_df") or df

        if "keyword" not in compare_df.columns or compare_df["keyword"].nunique() < 2:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;
                        padding:5rem 2rem;text-align:center;">
              <div style="width:56px;height:56px;background:rgba(13,13,13,0.8);
                          border:1px solid rgba(255,255,255,0.07);border-radius:14px;
                          display:flex;align-items:center;justify-content:center;
                          font-size:1.5rem;margin-bottom:1.25rem;">🆚</div>
              <h3 style="margin:0 0 0.5rem;color:#ffffff;font-weight:800;
                         font-family:'Cabinet Grotesk',sans-serif;">No comparison data yet</h3>
              <p style="color:#525252;max-width:360px;line-height:1.7;font-size:0.875rem;">
                Run analyses for at least two different brands, then load the full dataset
                using <strong style="color:#8F8F8F;">Load default CSV</strong> in the sidebar.
              </p>
            </div>""", unsafe_allow_html=True)
        else:
            available_brands = sorted(compare_df["keyword"].dropna().unique().tolist())
            selected_brands = st.multiselect("Select brands to compare", options=available_brands,
                default=available_brands[:min(4, len(available_brands))])

            if len(selected_brands) < 2:
                st.info("Select at least 2 brands to compare.")
            else:
                cdf = compare_df[compare_df["keyword"].isin(selected_brands)]
                stats = []
                for brand in selected_brands:
                    bdf = cdf[cdf["keyword"] == brand]
                    n = len(bdf)
                    bc = bdf["sentiment"].value_counts()
                    stats.append({"Brand": brand, "Comments": n,
                        "Positive %":  round(bc.get("positive", 0) / n * 100, 1),
                        "Neutral %":   round(bc.get("neutral",  0) / n * 100, 1),
                        "Negative %":  round(bc.get("negative", 0) / n * 100, 1),
                        "Avg score":   round(bdf["score"].mean(), 3) if "score" in bdf.columns else "-"})
                stats_df = pd.DataFrame(stats)

                with st.container(border=True):
                    st.markdown("**Summary**")
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)

                st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

                with st.container(border=True):
                    st.markdown("**Sentiment distribution by brand**")
                    melted = stats_df.melt(id_vars="Brand",
                        value_vars=["Positive %", "Neutral %", "Negative %"],
                        var_name="Sentiment", value_name="Pct")
                    melted["sentiment_key"] = melted["Sentiment"].map(
                        {"Positive %": "positive", "Neutral %": "neutral", "Negative %": "negative"})
                    fig_cmp = px.bar(melted, x="Brand", y="Pct", color="sentiment_key",
                        color_discrete_map=SENTIMENT_COLORS, barmode="group", text_auto=".1f",
                        labels={"Pct": "% of comments", "sentiment_key": "Sentiment"})
                    fig_cmp.update_traces(textposition="outside", textfont_size=11)
                    fig_cmp.update_layout(yaxis_range=[0, 108])
                    chart_layout(fig_cmp, height=340)
                    st.plotly_chart(fig_cmp, use_container_width=True)

                if "scraped_date" in cdf.columns and cdf["scraped_date"].nunique() > 1:
                    with st.container(border=True):
                        st.markdown("**Positive sentiment trend by brand**")
                        palette = px.colors.qualitative.Set2
                        fig_tc = go.Figure()
                        for i, brand in enumerate(selected_brands):
                            bdf = cdf[cdf["keyword"] == brand].copy()
                            bdf["scraped_date"] = pd.to_datetime(bdf["scraped_date"])
                            daily = (bdf.groupby("scraped_date")
                                     .apply(lambda g: (g["sentiment"]=="positive").sum()/len(g)*100)
                                     .rename("pct"))
                            daily = daily.reindex(
                                pd.date_range(daily.index.min(), daily.index.max(), freq="D")
                            ).fillna(None)
                            rolling = daily.rolling(7, min_periods=1).mean()
                            color = palette[i % len(palette)]
                            fig_tc.add_trace(go.Scatter(x=daily.index, y=daily.values,
                                mode="markers", name=brand,
                                marker=dict(color=color, size=6, opacity=0.5), legendgroup=brand))
                            fig_tc.add_trace(go.Scatter(x=rolling.index, y=rolling.values,
                                mode="lines", name=f"{brand} (7d avg)",
                                line=dict(color=color, width=2.5), legendgroup=brand))
                        fig_tc.update_layout(yaxis_ticksuffix="%", yaxis_title="Positive %")
                        chart_layout(fig_tc, height=340)
                        st.plotly_chart(fig_tc, use_container_width=True)


# ── Router ────────────────────────────────────────────────────────────────────
if st.session_state.get("page", "landing") == "landing":
    show_landing()
else:
    show_dashboard()
