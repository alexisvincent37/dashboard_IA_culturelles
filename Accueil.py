import streamlit as st
import os
from data.data_manager import load_data, get_home_kpis, get_mod_info

df_conv, df_votes, df_react, model_structure = load_data()
modinfo = get_mod_info()

if __name__ == "__main__":
    st.set_page_config(page_title="IA Culturelles - Accueil", layout="wide")

    total_conversations, total_families = get_home_kpis(df_conv, model_structure)

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&display=swap');

    [data-testid="stSidebarCollapseButton"] span { display: none; }
    html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
    .block-container { padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 97%; }

    div[data-testid="stMetric"] {
        background: #0F1117; padding: 12px 16px; border-radius: 8px;
        border: 1px solid #2A3050; min-height: 90px;
        display: flex; flex-direction: column; justify-content: center;
    }
    div[data-testid="stMetricValue"] { font-size: 1.55rem !important; font-weight: 700 !important; font-family: 'Syne', sans-serif !important; color: #F1F5F9 !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.65rem !important; color: #64748B !important; text-transform: uppercase; letter-spacing: 0.08em; }
    div[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

    [data-testid="stSidebar"] { background: #0A0C12; border-right: 1px solid #1E2130; }
    [data-testid="stSidebar"] * { font-family: 'DM Mono', monospace; }

    h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; letter-spacing: -0.03em; color: #F1F5F9 !important; margin-bottom: 4px !important; }
    h3, h4 { font-family: 'DM Mono', monospace !important; font-size: 0.68rem !important; color: #475569 !important; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 500 !important; margin-bottom: 0.7rem !important; margin-top: 0 !important; }
    hr { margin: 0.3rem 0 0.8rem 0; border-color: #1E2130; }

    .stSelectbox label { font-family: 'DM Mono', monospace !important; font-size: 0.68rem !important; color: #475569 !important; text-transform: uppercase; letter-spacing: 0.08em; }

    .badge-container { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 12px; }
    .badge-util, .badge-crea, .badge-clar, .badge-warn { padding: 3px 10px; border-radius: 4px; font-size: 0.72em; font-family: 'DM Mono', monospace; border: 1px solid; }
    .badge-util { background: rgba(52,211,153,0.08); color: #34d399; border-color: rgba(52,211,153,0.2); }
    .badge-crea { background: rgba(167,139,250,0.08); color: #a78bfa; border-color: rgba(167,139,250,0.2); }
    .badge-clar { background: rgba(96,165,250,0.08); color: #60a5fa; border-color: rgba(96,165,250,0.2); }
    .badge-warn { background: rgba(251,191,36,0.08); color: #fbbf24; border-color: rgba(251,191,36,0.2); }

    .feature-card {
        background: #0F1117; border: 1px solid #2A3050; border-radius: 10px;
        padding: 20px 22px; height: 100%; box-sizing: border-box;
    }
    .feature-icon { font-size: 1.4rem; margin-bottom: 10px; }
    .feature-title { font-family: 'Syne', sans-serif; font-size: 1.0rem; font-weight: 700; color: #F1F5F9; margin-bottom: 6px; }
    .feature-desc { font-family: 'DM Mono', monospace; font-size: 0.70rem; color: #64748B; line-height: 1.7; }
    .feature-tag { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.62em; font-family: 'DM Mono', monospace; margin-top: 10px; border: 1px solid; }
    .tag-blue { background: rgba(96,165,250,0.08); color: #60A5FA; border-color: rgba(96,165,250,0.2); }
    .tag-green { background: rgba(52,211,153,0.08); color: #34D399; border-color: rgba(52,211,153,0.2); }
    .tag-amber { background: rgba(251,191,36,0.08); color: #FBBF24; border-color: rgba(251,191,36,0.2); }
    .tag-purple { background: rgba(167,139,250,0.08); color: #a78bfa; border-color: rgba(167,139,250,0.2); }

    .data-row { display: flex; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 1px solid #1E2130; }
    .data-row:last-child { border-bottom: none; }
    .data-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
    .data-label { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: #94A3B8; }
    .data-value { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: #475569; margin-left: auto; }

    .section-label { font-family: 'DM Mono', monospace; font-size: 0.62rem; color: #475569; text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 14px; }
    .hero-sub { font-family: 'DM Mono', monospace; font-size: 0.80rem; color: #475569; line-height: 1.8; max-width: 680px; }
    .partner-badge { display: inline-flex; align-items: center; gap: 8px; background: #0F1117; border: 1px solid #2A3050; border-radius: 6px; padding: 6px 14px; font-family: 'DM Mono', monospace; font-size: 0.68rem; color: #64748B; margin-right: 8px; margin-top: 8px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎨 IA Culturelles")
    st.markdown(
        '<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#475569;margin-bottom:16px;">'
        'SAISON 4 · OPEN DATA UNIVERSITY · MINISTÈRE DE LA CULTURE'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-sub">'
        'Ce dashboard analyse les données de la plateforme <strong style="color:#94A3B8">Compar:IA</strong> '
        'pour explorer les biais culturels, les performances et les usages des IA génératives '
        'sur des contenus francophones. Les données sont croisées avec les bases ouvertes du '
        'Ministère de la Culture — <strong style="color:#94A3B8">Joconde</strong> et '
        '<strong style="color:#94A3B8">Basilic</strong>.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="margin-top:14px;">
        <span class="partner-badge">🏛️ Ministère de la Culture</span>
        <span class="partner-badge">🤖 Compar:IA · beta.gouv.fr</span>
        <span class="partner-badge">📚 Open Data University</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Conversations analysées", f"{total_conversations:,}".replace(",", " "))
    k2.metric("Familles de modèles", total_families)
    k3.metric("Volume de données", "2 Go")
    k4.metric("Bases de référence", "2 · Joconde + Basilic")

    st.markdown("---")
    st.markdown('<div class="section-label">Pages du dashboard</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Statistiques IA</div>
            <div class="feature-desc">
                Analyse détaillée par modèle et par version. Winrate, satisfaction utilisateurs,
                consommation énergétique, empreinte carbone, coût estimé par token.
                Répartition des catégories de conversations et évolution temporelle.
            </div>
            <span class="feature-tag tag-blue">27 modèles</span>
            <span class="feature-tag tag-green">Par version</span>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Analyse de biais</div>
            <div class="feature-desc">
                Croisement des conversations culturelles avec les bases Joconde (988k œuvres)
                et Basilic (86k équipements). Mesure des biais temporels, géographiques
                et par domaine artistique pour chaque modèle.
            </div>
            <span class="feature-tag tag-amber">Joconde</span>
            <span class="feature-tag tag-amber">Basilic</span>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🏟️</div>
            <div class="feature-title">Arena</div>
            <div class="feature-desc">
                Explorateur de conversations. Parcours les duels entre modèles,
                filtre par catégorie, résultat ou longueur. Consulte les retours
                utilisateurs et les commentaires laissés sur chaque réponse.
            </div>
            <span class="feature-tag tag-purple">Votes Compar:IA</span>
            <span class="feature-tag tag-green">Retours utilisateurs</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_data, col_method = st.columns(2)

    with col_data:
        st.markdown('<div class="section-label">Sources de données</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0F1117;border:1px solid #2A3050;border-radius:8px;padding:16px 18px;">
            <div class="data-row">
                <div class="data-dot" style="background:#60A5FA;"></div>
                <div class="data-label">conversations.parquet</div>
                <div class="data-value">311 k conversations</div>
            </div>
            <div class="data-row">
                <div class="data-dot" style="background:#34D399;"></div>
                <div class="data-label">votes.parquet</div>
                <div class="data-value">Préférences utilisateurs</div>
            </div>
            <div class="data-row">
                <div class="data-dot" style="background:#a78bfa;"></div>
                <div class="data-label">reactions.parquet</div>
                <div class="data-value">Likes · Dislikes · Tags</div>
            </div>
            <div class="data-row">
                <div class="data-dot" style="background:#F59E0B;"></div>
                <div class="data-label">joconde.csv</div>
                <div class="data-value">988k œuvres · sep |</div>
            </div>
            <div class="data-row">
                <div class="data-dot" style="background:#F59E0B;"></div>
                <div class="data-label">basilic.csv</div>
                <div class="data-value">86k équipements · sep ;</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_method:
        st.markdown('<div class="section-label">Méthodologie</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0F1117;border:1px solid #2A3050;border-radius:8px;padding:16px 18px;">
            <div class="data-row">
                <div class="data-dot" style="background:#60A5FA;"></div>
                <div class="data-label">Cleaning Polars</div>
                <div class="data-value">Regex · Normalisation modèles</div>
            </div>
            <div class="data-row">
                <div class="data-dot" style="background:#34D399;"></div>
                <div class="data-label">Filtrage culturel FR</div>
                <div class="data-value">Regex catégories · keywords</div>
            </div>
            <div class="data-row">
                <div class="data-dot" style="background:#a78bfa;"></div>
                <div class="data-label">TF-IDF</div>
                <div class="data-value">Top 300 termes par modèle</div>
            </div>
            <div class="data-row">
                <div class="data-dot" style="background:#F59E0B;"></div>
                <div class="data-label">Matching référentiels</div>
                <div class="data-value">Regex pré-compilées · word boundary</div>
            </div>
            <div class="data-row">
                <div class="data-dot" style="background:#F87171;"></div>
                <div class="data-label">Score biais</div>
                <div class="data-value">MAE distribution IA vs réf.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'DM Mono\',monospace;font-size:0.60rem;color:#2A3050;text-align:center;">'
        'Données Compar:IA · Ministère de la Culture · Open Data University · Saison 4 · 2025-2026'
        '</div>',
        unsafe_allow_html=True
    )