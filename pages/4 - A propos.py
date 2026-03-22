import streamlit as st

st.set_page_config(page_title="IA Culturelles - A propos", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&display=swap');

[data-testid="stSidebarCollapseButton"] span { display: none; }
html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
.block-container { padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 97%; }

[data-testid="stSidebar"] { background: #0A0C12; border-right: 1px solid #1E2130; }
[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace; }

h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; letter-spacing: -0.03em; color: #F1F5F9 !important; margin-bottom: 4px !important; }
hr { margin: 0.3rem 0 0.8rem 0; border-color: #1E2130; }

.card {
    background: #0F1117;
    border: 1px solid #2A3050;
    border-radius: 10px;
    padding: 20px 24px;
    height: 100%;
    box-sizing: border-box;
}
.card-accent-green { border-left: 3px solid #34D399; }
.card-accent-blue  { border-left: 3px solid #60A5FA; }
.card-accent-purple{ border-left: 3px solid #A78BFA; }
.card-accent-amber { border-left: 3px solid #FBBF24; }
.card-accent-red   { border-left: 3px solid #F87171; }

.member-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 4px;
}
.member-role {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 10px;
}
.member-desc {
    font-family: 'DM Mono', monospace;
    font-size: 0.70rem;
    color: #64748B;
    line-height: 1.7;
}

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1E2130;
}

.limit-card {
    background: #0F1117;
    border: 1px solid #2A3050;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.limit-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.limit-body {
    font-family: 'DM Mono', monospace;
    font-size: 0.70rem;
    color: #475569;
    line-height: 1.7;
}

.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 0.60em;
    font-family: 'DM Mono', monospace;
    margin-top: 10px;
    border: 1px solid;
}
.tag-green  { background: rgba(52,211,153,0.08);  color: #34D399; border-color: rgba(52,211,153,0.2); }
.tag-blue   { background: rgba(96,165,250,0.08);  color: #60A5FA; border-color: rgba(96,165,250,0.2); }
.tag-purple { background: rgba(167,139,250,0.08); color: #A78BFA; border-color: rgba(167,139,250,0.2); }
.tag-amber  { background: rgba(251,191,36,0.08);  color: #FBBF24; border-color: rgba(251,191,36,0.2); }

.partner-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #0F1117;
    border: 1px solid #2A3050;
    border-radius: 6px;
    padding: 6px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #64748B;
    margin-right: 8px;
    margin-top: 8px;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.80rem;
    color: #475569;
    line-height: 1.8;
    max-width: 760px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("A propos")
st.markdown(
    '<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#475569;margin-bottom:16px;">'
    'MASTER MECEN · UNIVERSITE DE TOURS · OPEN DATA UNIVERSITY · SAISON 4 · 2025/2026 · Dernière mise a jour : Mars 2026'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-sub">'
    'Ce dashboard a ete realise dans le cadre du defi <strong style="color:#94A3B8">IA Culturelles</strong> '
    'propose par le <strong style="color:#94A3B8">ministere de la Culture</strong> pour la Saison 4 '
    "d'<strong style=\"color:#94A3B8\">Open Data University</strong>. "
    'Il explore les usages, performances et biais culturels des modeles d\'IA conversationnelle '
    'a partir des donnees ouvertes de la plateforme Compar:IA, croisees avec les referentiels '
    'Joconde et Basilic du ministere de la Culture.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("""
<div style="margin-top:14px;margin-bottom:4px;">
    <span class="partner-badge">Ministere de la Culture</span>
    <span class="partner-badge">Compar:IA · beta.gouv.fr</span>
    <span class="partner-badge">Open Data University</span>
    <span class="partner-badge">Universite de Tours · MECEN</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Equipe
# ---------------------------------------------------------------------------

st.markdown('<div class="section-label">Equipe projet</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card card-accent-green">
        <div class="member-name">Abdul BALOGOUN</div>
        <div class="member-role">Data Engineering · Traitement des donnees</div>
        <div class="member-desc">
            Conception du pipeline de nettoyage et de transformation des donnees Compar:IA.
            Normalisation des noms de modeles, extraction des versions, construction
            des jeux de donnees nettoyees et preparation du corpus pour l'analyse de biais.
        </div>
        <span class="tag tag-green">data_cleaning.py</span>
        <span class="tag tag-green">Polars</span>
        <span class="tag tag-green">Pipeline</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card card-accent-blue">
        <div class="member-name">Jawad GRIB</div>
        <div class="member-role">Design · Interface Streamlit</div>
        <div class="member-desc">
            Conception graphique et design des pages du dashboard. Mise en forme
            de l'interface Streamlit, choix des composants visuels, typographie,
            palette de couleurs et coherence visuelle de l'ensemble des pages.
        </div>
        <span class="tag tag-blue">Streamlit</span>
        <span class="tag tag-blue">CSS</span>
        <span class="tag tag-blue">UI / UX</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card card-accent-purple">
        <div class="member-name">Alexis VINCENT</div>
        <div class="member-role">Backend · Fonctions & Analyse</div>
        <div class="member-desc">
            Developpement des fonctions de calcul et d'agregation des donnees :
            winrate, satisfaction, consommation, tokens, biais. Conception
            de la logique metier dans data_manager.py et du pipeline
            d'analyse de biais TF-IDF dans run_bias.py.
        </div>
        <span class="tag tag-purple">data_manager.py</span>
        <span class="tag tag-purple">run_bias.py</span>
        <span class="tag tag-purple">TF-IDF · sklearn</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------------------------
# Contexte du defi
# ---------------------------------------------------------------------------

st.markdown('<div class="section-label">Contexte du defi</div>', unsafe_allow_html=True)

col_ctx, col_obj = st.columns(2)

with col_ctx:
    st.markdown("""
    <div class="card">
        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#475569;
        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;padding-bottom:8px;
        border-bottom:1px solid #1E2130;">Problematique</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.70rem;color:#64748B;line-height:1.8;">
            Le marche des IA generatives est domine par des acteurs americains et chinois,
            creant un biais linguistique et culturel au detriment des contenus francophones.
            Le ministere de la Culture cherche a evaluer la place accordee par ces modeles
            aux contenus culturels francais, et a mesurer les ecarts de representation
            entre les reponses IA et les referentiels institutionnels.
            <br><br>
            Ce projet repond a la question : <strong style="color:#94A3B8">comment mesurer
            objectivement les biais culturels des IA conversationnelles sur des contenus
            francophones ?</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_obj:
    st.markdown("""
    <div class="card">
        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#475569;
        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;padding-bottom:8px;
        border-bottom:1px solid #1E2130;">Ce que le dashboard permet</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.70rem;color:#64748B;line-height:1.8;">
            <span style="color:#34D399;">—</span> Comparer les performances et winrates de 28 modeles
            sur des conversations francophones reelles<br><br>
            <span style="color:#60A5FA;">—</span> Mesurer les biais temporels, geographiques
            et par domaine artistique via les referentiels Joconde et Basilic<br><br>
            <span style="color:#A78BFA;">—</span> Explorer les conversations individuelles
            via l'Arena pour auditer les reponses modele par modele<br><br>
            <span style="color:#FBBF24;">—</span> Evaluer l'impact environnemental
            (consommation kWh, empreinte CO2) de chaque modele
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------------------------
# Limites
# ---------------------------------------------------------------------------

st.markdown('<div class="section-label">Limites et points de vigilance</div>', unsafe_allow_html=True)

lim1, lim2 = st.columns(2)

with lim1:
    st.markdown("""
    <div class="limit-card">
        <div class="limit-title">Representativite des donnees</div>
        <div class="limit-body">
            Les conversations analysees proviennent de la plateforme Compar:IA,
            dont les utilisateurs sont majoritairement des personnes sensibilisees
            au numerique et a l'open data. Ce biais de selection implique que
            les usages observes ne sont pas necessairement representatifs
            de l'ensemble des utilisateurs d'IA generatives.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="limit-card">
        <div class="limit-title">Methode de detection des biais</div>
        <div class="limit-body">
            L'analyse de biais repose sur un matching par keywords et TF-IDF,
            une approche lexicale qui ne capture pas la semantique profonde
            des reponses. Un modele peut citer correctement une periode
            sans en avoir une representation fidele, et inversement.
            Une approche par LLM juge aurait ete plus precise mais
            plus couteuse en calcul.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="limit-card">
        <div class="limit-title">Donnees de consommation energetique</div>
        <div class="limit-body">
            Les valeurs de consommation kWh sont
            fournies directement par Compar:IA et dependent des
            estimations declarees par les fournisseurs de modeles.
            Ces chiffres sont indicatifs et peuvent sous-estimer
            la consommation reelle, notamment pour les modeles
            dont l'infrastructure n'est pas transparente. De plus,
            l'empreinte CO2 est estimée à partir des données françaises de mix énergétique.
        </div>
    </div>
    """, unsafe_allow_html=True)

with lim2:
    st.markdown("""
    <div class="limit-card">
        <div class="limit-title">Perimetre temporel</div>
        <div class="limit-body">
            Les donnees couvrent les conversations collectees sur Compar:IA
            jusqu'en fevrier 2026. Les modeles evoluent rapidement :
            les scores de performance, winrates et biais observes
            refletent l'etat des modeles sur cette periode et
            peuvent ne plus etre valables pour des versions
            posterieures ou des mises a jour majeures.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="limit-card">
        <div class="limit-title">Referentiels culturels</div>
        <div class="limit-body">
            Les referentiels Joconde et Basilic, bien que representatifs
            des collections et equipements institutionnels francais,
            ne couvrent pas l'integralite de la production culturelle.
            La culture populaire, les contenus numeriques natifs
            et une partie du patrimoine immatériel en sont absents,
            ce qui peut biaiser l'interpretation des scores de couverture.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="limit-card">
        <div class="limit-title">Normalisation des noms de modeles</div>
        <div class="limit-body">
            La normalisation des noms de modeles (ex: regroupement
            de toutes les variantes GPT sous "gpt") peut masquer
            des differences de performance importantes entre versions.
            Certains modeles peu frequents dans les donnees
            peuvent presenter des scores statistiquement peu fiables
            du fait d'un faible nombre d'observations.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    '<div style="font-family:\'DM Mono\',monospace;font-size:0.60rem;color:#2A3050;text-align:center;">'
    'Abdul BALOGOUN · Jawad GRIB · Alexis VINCENT &nbsp;·&nbsp; '
    'Master MECEN · Universite de Tours &nbsp;·&nbsp; '
    'Open Data University Saison 4 · 2025/2026 &nbsp;·&nbsp; '
    'Ministere de la Culture'
    '</div>',
    unsafe_allow_html=True,
)