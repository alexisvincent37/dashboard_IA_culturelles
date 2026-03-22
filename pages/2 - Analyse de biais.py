import streamlit as st
import numpy as np
import polars as pl
import plotly.express as px

from data.data_manager import (
    load_bias_data,
    get_bias_means,
    bias_score_color,
    render_bias_score_mini,
    render_bias_bar,
    render_bias_domain_bar,
    bias_badges_html,
    BIAS_LEGEND_HTML,
)

st.set_page_config(page_title="IA Culturelles - Analyse de biais", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&display=swap');

[data-testid="stSidebarCollapseButton"] span { display: none; }
html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
.block-container { padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 97%; }

div[data-testid="stMetric"] { background: #0F1117; padding: 12px 16px; border-radius: 8px; border: 1px solid #2A3050; min-height: 90px; display: flex; flex-direction: column; justify-content: center; }
div[data-testid="stMetricValue"] { font-size: 1.55rem !important; font-weight: 700 !important; font-family: 'Syne', sans-serif !important; color: #F1F5F9 !important; }
div[data-testid="stMetricLabel"] { font-size: 0.65rem !important; color: #64748B !important; text-transform: uppercase; letter-spacing: 0.08em; }
div[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

[data-testid="stSidebar"] { background: #0A0C12; border-right: 1px solid #1E2130; }
[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace; }

h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; letter-spacing: -0.03em; color: #F1F5F9 !important; margin-bottom: 4px !important; }
h3, h4 { font-family: 'DM Mono', monospace !important; font-size: 0.68rem !important; color: #475569 !important; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 500 !important; margin-bottom: 0.7rem !important; margin-top: 0 !important; }
hr { margin: 0.3rem 0 0.8rem 0; border-color: #1E2130; }

.stSelectbox label { font-family: 'DM Mono', monospace !important; font-size: 0.68rem !important; color: #475569 !important; text-transform: uppercase; letter-spacing: 0.08em; }

.badge { padding: 3px 10px; border-radius: 4px; font-size: 0.70em; font-family: 'DM Mono', monospace; border: 1px solid; display: inline-block; margin: 2px 2px 2px 0; }
.badge-r { background: rgba(248,113,113,0.08); color: #F87171; border-color: rgba(248,113,113,0.2); }
.badge-a { background: rgba(251,191,36,0.08);  color: #FBBF24; border-color: rgba(251,191,36,0.2); }
.badge-g { background: rgba(52,211,153,0.08);  color: #34D399; border-color: rgba(52,211,153,0.2); }
.badge-b { background: rgba(96,165,250,0.08);  color: #60A5FA; border-color: rgba(96,165,250,0.2); }

.block-card { background: #0F1117; border: 1px solid #2A3050; border-radius: 8px; padding: 18px 20px; margin-bottom: 14px; }
.block-header { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #475569; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid #1E2130; display: flex; justify-content: space-between; }
.block-source { font-size: 0.58rem; color: #2A3050; text-transform: none; letter-spacing: 0; }
.score-label { font-family: 'DM Mono', monospace; font-size: 0.62rem; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.score-value-green { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 700; color: #34D399; line-height: 1; }
.score-value-amber { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 700; color: #FBBF24; line-height: 1; }
.score-value-red   { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 700; color: #F87171; line-height: 1; }
.score-value-blue  { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 700; color: #60A5FA; line-height: 1; }
.score-desc { font-family: 'DM Mono', monospace; font-size: 0.70rem; color: #64748B; margin: 8px 0 10px; line-height: 1.6; }
.methodology-box { background: #0A0C12; border: 1px solid #1E2130; border-radius: 6px; padding: 10px 14px; font-size: 0.68rem; color: #475569; line-height: 1.9; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DONNÉES
# ---------------------------------------------------------------------------
REAL_DATA = load_bias_data()

if not REAL_DATA:
    st.error("Fichier bias_scores.parquet introuvable. Lance d'abord run_bias.py.")
    st.stop()

MODELS = sorted(list(REAL_DATA.keys()))
MEAN   = get_bias_means(REAL_DATA)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Contrôle")
    st.markdown("---")
    selected = st.selectbox("Modèle", MODELS, index=0)
    st.markdown("---")
    st.markdown("""<div class="methodology-box">
<div style="color:#64748B;font-size:0.60rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">Méthode d'analyse</div>
<span style="color:#94A3B8;">Joconde</span><br>
TF-IDF · conv.parquet<br>
Regex culturelle · filtrage FR<br>
Réf. 580k œuvres<br><br>
<span style="color:#94A3B8;">Basilic</span><br>
Matching keywords géocodés<br>
Réf. 86k équipements<br><br>
<span style="color:#60A5FA;">━━</span> IA &nbsp;&nbsp;
<span style="color:#F59E0B;">━━</span> Référence
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
d = REAL_DATA[selected]

st.title(f"🔍 {selected}")
st.markdown(
    '<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#475569;margin-bottom:2px;">'
    'ANALYSE DE BIAIS CULTURELS &nbsp;·&nbsp; Joconde + Basilic'
    '</div>',
    unsafe_allow_html=True
)



st.markdown("---")
st.markdown("#### Synthèse")
k1, k2, k3, k4 = st.columns(4)

dj  = d["joconde_score"]      - MEAN["joconde_score"]
djc = d["joconde_couverture"] - MEAN["joconde_couverture"]
db  = d["basilic_score"]      - MEAN["basilic_score"]
dbc = d["basilic_couverture"] - MEAN["basilic_couverture"]

k1.metric("Biais œuvres (Joconde)", f"{d['joconde_score']:.1f}/100",   f"{dj:+.1f} vs moyenne", delta_color="inverse")
k2.metric("Couverture Joconde",     f"{d['joconde_couverture']:.1f}%", f"{djc:+.1f}% vs moyenne", delta_color="normal")
k3.metric("Biais lieux (Basilic)",  f"{d['basilic_score']:.1f}/100",   f"{db:+.1f} vs moyenne", delta_color="inverse")
k4.metric("Couverture Basilic",     f"{d['basilic_couverture']:.1f}%", f"{dbc:+.1f}% vs moyenne", delta_color="normal")

st.markdown("---")

# ── JOCONDE ──────────────────────────────────────────────────────────────────
st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#4A90E2;
text-transform:uppercase;letter-spacing:0.14em;margin-bottom:12px;">
◆ &nbsp;Biais œuvres &amp; artistes — Réf. Joconde</div>""", unsafe_allow_html=True)

jcol1, jcol2 = st.columns(2)

with jcol1:
    cj = bias_score_color("score",      d["joconde_score"])
    cc = bias_score_color("couverture", d["joconde_couverture"])
    st.markdown(f"""<div class="block-card">
<div class="block-header">
<span>Score biais &amp; couverture</span>
<span class="block-source">collections-musees-france.fr</span>
</div>
<div style="display:flex;gap:32px;margin-bottom:14px;">
{render_bias_score_mini("Biais Joconde", d['joconde_score'], cj)}
{render_bias_score_mini("Couverture", d['joconde_couverture'], cc, suffix="%")}
</div>
<div class="score-desc">{d['joconde_desc']}</div>
<div>{bias_badges_html(d['joconde_badges'])}</div>
</div>""", unsafe_allow_html=True)

    domain_bars = "".join(render_bias_domain_bar(k, v) for k, v in d["joconde_domaines"].items())
    st.markdown(f"""<div class="block-card">
<div class="block-header">
<span>Écart par domaine vs Joconde</span>
<span class="block-source">en points · ligne centrale = référence</span>
</div>
{domain_bars}
</div>""", unsafe_allow_html=True)

with jcol2:
    periode_bars = "".join(
        render_bias_bar(p, v["ia"], v["ref"], max_val=80)
        for p, v in d["joconde_periodes"].items()
    )
    st.markdown(f"""<div class="block-card">
<div class="block-header">
<span>Distribution temporelle — IA vs référence</span>
<span class="block-source">% des citations</span>
</div>
{BIAS_LEGEND_HTML}
{periode_bars}
</div>""", unsafe_allow_html=True)

st.markdown("---")

# ── BASILIC ──────────────────────────────────────────────────────────────────
st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#34D399;
text-transform:uppercase;letter-spacing:0.14em;margin-bottom:12px;">
◆ &nbsp;Biais géographique &amp; lieux — Réf. Basilic</div>""", unsafe_allow_html=True)

bcol1, bcol2 = st.columns(2)

with bcol1:
    cb  = bias_score_color("score",      d["basilic_score"])
    cbc = bias_score_color("couverture", d["basilic_couverture"])
    st.markdown(f"""<div class="block-card">
<div class="block-header">
<span>Score biais &amp; couverture</span>
<span class="block-source">data.culture.gouv.fr · Basilic 2020</span>
</div>
<div style="display:flex;gap:32px;margin-bottom:14px;">
{render_bias_score_mini("Biais Basilic", d['basilic_score'], cb)}
{render_bias_score_mini("Couverture", d['basilic_couverture'], cbc, suffix="%")}
</div>
<div class="score-desc">{d['basilic_desc']}</div>
<div>{bias_badges_html(d['basilic_badges'])}</div>
</div>""", unsafe_allow_html=True)

    type_bars = "".join(
        render_bias_bar(t, v["ia"], v["ref"], max_val=80)
        for t, v in d["basilic_types"].items()
    )
    st.markdown(f"""<div class="block-card">
<div class="block-header">
<span>Répartition par type d'équipement</span>
<span class="block-source">% des citations</span>
</div>
{BIAS_LEGEND_HTML}
{type_bars}
</div>""", unsafe_allow_html=True)

with bcol2:
    region_bars = "".join(
        render_bias_bar(r, v["ia"], v["ref"], max_val=65)
        for r, v in d["basilic_regions"].items()
    )
    st.markdown(f"""<div class="block-card">
<div class="block-header">
<span>Distribution régionale — IA vs référence</span>
<span class="block-source">% des lieux cités</span>
</div>
{BIAS_LEGEND_HTML}
{region_bars}
</div>""", unsafe_allow_html=True)