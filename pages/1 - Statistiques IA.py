import streamlit as st
import plotly.express as px
import polars as pl
from Accueil import df_conv, df_votes, df_react, model_structure, modinfo
from data.data_manager import nbtotalconv, nbreactrow, satisfaction, audience, winrate, ranking, plot_treemap, electric_conso_total, \
                              electric_conso_avg, nb_tokens, nb_tokens_avg, carbon_footprint, cost_estimation, plot_panel_graph, \
                              nb_turn, mean_turn, pick_rate, deep_engagement_rate, get_global_benchmarks, get_model_badges, nbparam

st.set_page_config(page_title="Cockpit IA Culturelles", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&display=swap');
            
[data-testid="stSidebarCollapseButton"] span { display: none; }

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
.block-container { padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 97%; }

div[data-testid="stMetric"] {
    background: #0F1117;
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid #2A3050;
    min-height: 90px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
div[data-testid="stMetricValue"] { font-size: 1.55rem !important; font-weight: 700 !important; font-family: 'Syne', sans-serif !important; color: #F1F5F9 !important; }
div[data-testid="stMetricLabel"] { font-size: 0.65rem !important; color: #64748B !important; text-transform: uppercase; letter-spacing: 0.08em; }
div[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

[data-testid="stSidebar"] { background: #0A0C12; border-right: 1px solid #1E2130; }
[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace; }

h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; letter-spacing: -0.03em; color: #F1F5F9 !important; margin-bottom: 4px !important; }
h3, h4 { font-family: 'DM Mono', monospace !important; font-size: 0.68rem !important; color: #475569 !important;
          text-transform: uppercase; letter-spacing: 0.12em; font-weight: 500 !important; margin-bottom: 0.7rem !important; margin-top: 0 !important; }
hr { margin: 0.3rem 0 0.8rem 0; border-color: #1E2130; }

div[data-testid="stProgress"] > div { background: #1E2130; border-radius: 2px; height: 3px; }
div[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #4A90E2, #34D399); border-radius: 2px; }

.stSelectbox label { font-family: 'DM Mono', monospace !important; font-size: 0.68rem !important; color: #475569 !important; text-transform: uppercase; letter-spacing: 0.08em; }

.badge-container { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 12px; }
.badge-util, .badge-crea, .badge-clar, .badge-warn { padding: 3px 10px; border-radius: 4px; font-size: 0.72em; font-family: 'DM Mono', monospace; border: 1px solid; }
.badge-util { background: rgba(52,211,153,0.08); color: #34d399; border-color: rgba(52,211,153,0.2); }
.badge-crea { background: rgba(167,139,250,0.08); color: #a78bfa; border-color: rgba(167,139,250,0.2); }
.badge-clar { background: rgba(96,165,250,0.08); color: #60a5fa; border-color: rgba(96,165,250,0.2); }
.badge-warn { background: rgba(251,191,36,0.08); color: #fbbf24; border-color: rgba(251,191,36,0.2); }

.rank-highlight { color: #4A90E2; font-family: 'Syne', sans-serif; }
.mono-sm { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: #475569; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Contrôle")
    if model_structure:
        families = sorted(list(model_structure.keys()))
        selected_family = st.selectbox("Modèle :", families, index=0)
        versions = sorted(model_structure.get(selected_family, []))
        version_options = ["Tous"] + versions if versions else ["Tous"]
        selected_version = st.selectbox("Version :", version_options)
        st.divider()
    else:
        st.error("Données non chargées.")
        selected_family, selected_version = "Inconnu", "Tous"

rang_actuel = ranking(df_votes, selected_family, model_structure)
st.markdown(
    f"<h1>{selected_family} "
    f"<span style='font-size:1.1rem;color:#334155;font-weight:400;font-family:DM Mono,monospace;'>"
    f"| Classement : <span class='rank-highlight'>{rang_actuel}</span></span></h1>",
    unsafe_allow_html=True
)

if rang_actuel and "/" in str(rang_actuel):
    try:
        pos, total = map(int, str(rang_actuel).split("/"))
        st.progress((total - pos + 1) / total)
    except Exception:
        st.divider()
else:
    st.divider()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("📌 Version / Taille", selected_version, help=nbparam(df_conv, selected_family, selected_version))
kpi2.metric("💬 Conversations", nbtotalconv(df_conv, selected_family, selected_version),
            help=f"Basé sur {audience(df_conv, selected_family, selected_version)} utilisateurs différents")
kpi3.metric("🏆 Taux de victoire", f"{winrate(df_votes, selected_family, selected_version)}%",
            help=f"Basé sur {nbtotalconv(df_votes, selected_family, selected_version)} votes")
kpi4.metric("⭐ Satisfaction", satisfaction(df_react, selected_family, selected_version),
            help=nbreactrow(df_react, selected_family, selected_version))

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    with st.container(border=True):
        st.markdown("#### 🎨 Répartition des 5 premières catégories les plus consultées")
        plot_treemap(df_conv, selected_family, selected_version)

    with st.container(border=True):
        st.markdown("#### ✨ Profil Créatif & Engagement")
        eng1, eng2, eng3 = st.columns(3)
        eng1.metric(
            "Tours de parole",
            nb_turn(df_conv, selected_family, selected_version),
            help=f"En moyenne {mean_turn(df_conv, selected_family, selected_version)} tours par conversation"
        )

        pop_val_str = pick_rate(df_conv, selected_family, selected_version)
        pop_val = float(pop_val_str.replace('%', ''))
        diff = round(pop_val - 3.7, 1)
        eng2.metric(
            label="Attractivité (Choix)",
            value=f"{pop_val}%",
            delta=f"{diff}% vs moy.",
            help="Pourcentage de fois où ce modèle a été choisi lorsqu'il était en compétition (en mode manuel). Moyenne actuelle de tous les modèles : 3.7%"
        )

        eng3.metric(
            "Engagement profond",
            deep_engagement_rate(df_conv, selected_family, selected_version),
            help="Pourcentage de conversations où l'utilisateur a interagi avec au moins 3 messages du modèle."
        )

        st.markdown(get_model_badges(df_react, df_conv, selected_family, selected_version), unsafe_allow_html=True)

with col_right:
    with st.container(border=True):
        st.markdown("#### 🌱 Consommation")

        eco1, eco2 = st.columns(2)
        eco1.metric(
            "Conso. Total",
            f"{electric_conso_total(df_conv, selected_family, selected_version)} kWh",
            help=f"En moyenne {electric_conso_avg(df_conv, selected_family, selected_version)} kWh par conversation"
        )

        total_tokens = nb_tokens(df_conv, selected_family, selected_version)
        avg_tokens = nb_tokens_avg(df_conv, selected_family, selected_version)
        eco2.metric("Tokens", total_tokens, help=f"En moyenne {avg_tokens} tokens par conversation")

        eco3, eco4 = st.columns(2)
        n_conv = nbtotalconv(df_conv, selected_family, selected_version)
        eco3.metric(
            "Empreinte CO₂",
            f"{carbon_footprint(df_conv, selected_family, selected_version)} kg CO₂e",
            help=f"En moyenne {round(carbon_footprint(df_conv, selected_family, selected_version) / n_conv * 1000, 3) if n_conv else 0} g CO₂e par conversation"
        )

        cost_val = cost_estimation(modinfo, selected_family, selected_version)
        total_cost = "N/A" if cost_val is None else f"{round(cost_val * total_tokens, 2)}$"
        ver_txt = selected_version if selected_version != "Tous" else "toutes versions"
        help_cost = f"Prix pour un token {selected_family} {ver_txt} : {cost_val}$" if cost_val is not None else ""
        eco4.metric("Coût estimé", total_cost, help=help_cost)

    with st.container(border=True):
        st.markdown("#### 📊 Évolution")
        fig = plot_panel_graph(df_conv, df_votes, selected_family, selected_version)
        if fig:
            st.plotly_chart(fig, width="stretch")
        else:
            st.markdown(
                "<div style='padding:20px;text-align:center;color:#334155;background:#0A0C12;border-radius:6px;font-family:DM Mono,monospace;font-size:0.75rem;'>Aucune donnée</div>",
                unsafe_allow_html=True
            )