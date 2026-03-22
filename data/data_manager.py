import polars as pl
import os
import json
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import requests

@st.cache_data
def load_data():
    path_conv = "data/cleaned/conv.parquet"
    path_votes = "data/cleaned/vot.parquet"
    path_react = "data/cleaned/react.parquet"
    
    if not all(os.path.exists(p) for p in [path_conv, path_votes, path_react]):
        return None, None, None, {}

    df_conv = pl.read_parquet(path_conv)
    df_votes = pl.read_parquet(path_votes)
    df_react = pl.read_parquet(path_react)
    
    m_conv = pl.concat([
        df_conv.select([pl.col("base_model_a").alias("family"), pl.col("version_a").alias("version")]),
        df_conv.select([pl.col("base_model_b").alias("family"), pl.col("version_b").alias("version")])
    ])
    
    all_models = m_conv.unique().sort("family")

    structure = {}
    for row in all_models.iter_rows(named=True):
        fam = row['family']
        ver = row['version']
        if fam not in structure: 
            structure[fam] = []
        if ver and str(ver).strip() != "": 
            if ver not in structure[fam]:
                structure[fam].append(ver)

    return df_conv, df_votes, df_react, structure

@st.cache_data
def get_home_kpis(df, model_structure):
    if df is None:
        return 0, 0
    return df.height, len(model_structure.keys())

@st.cache_data(ttl=86400)
def get_mod_info():
    url = "https://openrouter.ai/api/v1/models"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None
    

def nbtotalconv(df, model, version):
    if model is None or df is None:
        return 0
    
    if version == "Tous":
        cond = (pl.col("base_model_a") == model) | (pl.col("base_model_b") == model)
    else:
        cond = ((pl.col("base_model_a") == model) & (pl.col("version_a") == version)) | \
               ((pl.col("base_model_b") == model) & (pl.col("version_b") == version))
    
    return df.filter(cond).height

def nbparam(df, model, version):
    """
    Retourne le nombre total de paramètres d'un modèle pour une version donnée.

    Retourne "N/A" si version == "Tous" car les paramètres sont propres à chaque
    version — les agréger n'aurait pas de sens (ex: GPT-4o mini vs GPT-4o).
    On prend la valeur modale : total_params est une propriété fixe du modèle.

    Note : la colonne source s'appelle "model_a_acitve_params" dans le dataset
    Compar:IA (faute de frappe conservée pour coller au schéma réel).

    Args:
        df:      DataFrame de conversations.
        model:   Nom de la famille de modèle.
        version: Version spécifique. "Tous" retourne "N/A".

    Returns:
        Chaîne formatée "X Mds params", ou "N/A" si version == "Tous" ou info manquante.
    """
    if df is None or model is None or version == "Tous":
        return "N/A"

    cond = (
        ((pl.col("base_model_a") == model) & (pl.col("version_a") == version)) |
        ((pl.col("base_model_b") == model) & (pl.col("version_b") == version))
    )
    subset = df.filter(cond)

    if subset.is_empty():
        return "N/A"

    vals_a = subset.filter(pl.col("base_model_a") == model)["model_a_total_params"].drop_nulls()
    vals_b = subset.filter(pl.col("base_model_b") == model)["model_b_total_params"].drop_nulls()
    vals   = pl.concat([vals_a, vals_b])

    if vals.is_empty():
        return "N/A"

    most_common = vals.value_counts(sort=True).row(0)[0]
    return f"{most_common} Mds params"

def audience(df, model, version):
    if model is None or df is None:
        return 0
    
    if version == "Tous":
        cond = (pl.col("base_model_a") == model) | (pl.col("base_model_b") == model)
    else:
        cond = ((pl.col("base_model_a") == model) & (pl.col("version_a") == version)) | \
               ((pl.col("base_model_b") == model) & (pl.col("version_b") == version))
    
    return df.filter(cond).select("visitor_id").unique().height

def winrate(df, model, version):
    if model is None or df is None:
        return None

    if version == "Tous":
        cond = (pl.col("base_model_a") == model) | (pl.col("base_model_b") == model)
    else:
        cond = ((pl.col("base_model_a") == model) & (pl.col("version_a") == version)) | \
               ((pl.col("base_model_b") == model) & (pl.col("version_b") == version))
            
    subset = df.filter(cond)
    total = subset.height

    if total == 0:
        return None
    
    nul = subset.filter(pl.col("both_equal") == True).height
    win = subset.filter((pl.col("chosen_base_model") == model) & (pl.col("both_equal") == False)).height

    winratio = (win + (nul * 0.5)) / total
    return round(winratio*100, 2)

@st.cache_data
def ranking(df, x, listmodel):
    if listmodel is None or df is None:
        return None
    
    rank = {}
    level = 1
    for model in listmodel:
        score = winrate(df, model, "Tous")
        if score is not None:
            rank[model] = score

    rank = dict(sorted(rank.items(), key=lambda item: item[1], reverse=True))
    
    for model in rank.keys():
        rank[model] = level
        level += 1

    return f"{rank[x] if x in rank else None } / {len(rank)}"
        

def satisfaction(df, model, version):
    if model is None or df is None:
        return None
    
    cond = pl.col("model") == model
    if version != "Tous":
        cond = cond & (pl.col("version") == version)
    
    subset = df.filter(cond)
    if subset.height == 0:
        return None
        
    satisfaction_moy = subset.select(pl.col("liked")).mean().item()
    return f"{round(satisfaction_moy*100, 2)}%" if satisfaction_moy is not None else None

def nbreactrow(df, model, version):
    if model is None or df is None:
        return "0 réactions"
    
    cond = pl.col("model") == model
    if version != "Tous":
        cond = cond & (pl.col("version") == version)
        
    nbligne = df.filter(cond).height
    return f"Calculé sur {nbligne} réactions"


def categories_keyword(df, model, version):
    if model is None or df is None:
        return None

    if version == "Tous":
        cond = (pl.col("base_model_a") == model) | (pl.col("base_model_b") == model)
    else:
        cond = (
            ((pl.col("base_model_a") == model) & (pl.col("version_a") == version)) |
            ((pl.col("base_model_b") == model) & (pl.col("version_b") == version))
        )
    
    subset = df.filter(
        cond & pl.col("languages").list.contains("fr")
    )

    if subset.is_empty():
        return None
    
    df_res = (
        subset
        .explode("categories")
        .explode("keywords")
        .drop_nulls(["categories", "keywords"])
    )

    if df_res.is_empty():
        return None

    df_ready = (
        df_res
        .with_columns([
            pl.col("categories").str.to_lowercase(),
            pl.col("keywords").str.to_lowercase()
        ])
        .group_by(["categories", "keywords"])
        .len()
        .sort("len", descending=True)
        .head(50)
    )
    return df_ready


def plot_treemap(df, model, version):
    df_ready = categories_keyword(df, model, version)
    
    if df_ready is None or df_ready.is_empty():
        st.info("Aucune donnée disponible.")
        return

    top_cats = (
        df_ready.group_by("categories")
        .agg(pl.col("len").sum())
        .sort("len", descending=True)
        .head(5)
        .get_column("categories")
    )
    
    df_filtered = df_ready.filter(pl.col("categories").is_in(top_cats))

    fig = px.treemap(
        df_filtered.to_pandas(),
        path=["categories", "keywords"],
        values="len",
        color="categories",
        color_discrete_sequence=px.colors.qualitative.Vivid 
    )

    fig.update_traces(
        textinfo="label+value",
        marker=dict(line=dict(width=1, color='#0e1117')),
        textfont=dict(size=14, color="white")
    )

    fig.update_layout(
        hovermode=False,
        margin=dict(t=0, l=0, r=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200
    )

    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


def electric_conso_total(df, model, version):
    if df is None or model is None: return 0
    
    mask_a = (pl.col("base_model_a") == model)
    mask_b = (pl.col("base_model_b") == model)
    
    if version != "Tous":
        mask_a = mask_a & (pl.col("version_a") == version)
        mask_b = mask_b & (pl.col("version_b") == version)
    
    conso_a = df.filter(mask_a).select(pl.col("total_conv_a_kwh").sum()).item() or 0
    conso_b = df.filter(mask_b).select(pl.col("total_conv_b_kwh").sum()).item() or 0

    return round(conso_a + conso_b, 2)

def electric_conso_avg(df, model, version):
    if df is None or model is None: return 0.0
    
    if version == "Tous":
        cond = (pl.col("base_model_a") == model) | (pl.col("base_model_b") == model)
    else:
        cond = ((pl.col("base_model_a") == model) & (pl.col("version_a") == version)) | \
               ((pl.col("base_model_b") == model) & (pl.col("version_b") == version))
    
    subset = df.filter(cond)
    if subset.is_empty(): return 0.0
    
    conso_serie = subset.select(
        pl.when(pl.col("base_model_a") == model)
        .then(pl.col("total_conv_a_kwh"))
        .otherwise(pl.col("total_conv_b_kwh"))
        .alias("conso")
    )

    return round(conso_serie["conso"].mean(), 3)

def nb_tokens(df, model, version):
    if df is None or model is None: return 0
    
    mask_a = (pl.col("base_model_a") == model)
    mask_b = (pl.col("base_model_b") == model)
    
    if version != "Tous":
        mask_a = mask_a & (pl.col("version_a") == version)
        mask_b = mask_b & (pl.col("version_b") == version)
        
    tok_a = df.filter(mask_a).select(pl.col("total_conv_a_output_tokens").sum()).item() or 0
    tok_b = df.filter(mask_b).select(pl.col("total_conv_b_output_tokens").sum()).item() or 0
    
    return int(tok_a + tok_b)

def nb_tokens_avg(df, model, version):
    if df is None or model is None: return 0.0
    
    if version == "Tous":
        cond = (pl.col("base_model_a") == model) | (pl.col("base_model_b") == model)
    else:
        cond = ((pl.col("base_model_a") == model) & (pl.col("version_a") == version)) | \
               ((pl.col("base_model_b") == model) & (pl.col("version_b") == version))
        
    subset = df.filter(cond)
    if subset.is_empty(): return 0.0
    
    tokens_serie = subset.select(
        pl.when(pl.col("base_model_a") == model)
        .then(pl.col("total_conv_a_output_tokens"))
        .otherwise(pl.col("total_conv_b_output_tokens"))
        .alias("tokens")
    )
    
    return round(tokens_serie["tokens"].mean(), 3)


def carbon_footprint(df, model, version):
    moyenne_conso_france_kwh = 0.057
    conso_fr = electric_conso_total(df, model, version) * moyenne_conso_france_kwh
    return round(conso_fr, 2)


def cost_estimation(info, model, version):
    if not info or "data" not in info or not model:
        return None

    family = model.lower()
    vers = version.lower() if version else "tous"

    family_match = [m for m in info["data"] if family in m.get('id', '').lower()]
    
    if not family_match:
        return None
        
    if vers == "tous":
        prices = []
        for m in family_match:
            try:
                p = float(m.get("pricing", {}).get("completion", 0) or 0)
                prices.append(p)
            except (ValueError, TypeError):
                continue
        
        if not prices: return 0.0
        if all(p == 0 for p in prices): return "Gratuit"
        return round(sum(prices) / len(prices), 6)
        
    vers_match = [m for m in family_match if vers in m.get('id', '').lower()]
    if not vers_match:
        return None
        
    try:
        val = vers_match[0].get("pricing", {}).get("completion", 0)
        return float(val) if val else 0.0
    except (ValueError, TypeError):
        return 0.0
        

def tokens_frequency_panel_graph(df, model, version):
    if df is None: return None
    
    m_a = (pl.col("base_model_a") == model)
    m_b = (pl.col("base_model_b") == model)
    if version != "Tous":
        m_a &= (pl.col("version_a") == version)
        m_b &= (pl.col("version_b") == version)

    df_a = df.filter(m_a).select([
        pl.col("timestamp").dt.strftime("%Y-%m").alias("month"),
        pl.col("total_conv_a_output_tokens").alias("tokens")
    ])
    df_b = df.filter(m_b).select([
        pl.col("timestamp").dt.strftime("%Y-%m").alias("month"),
        pl.col("total_conv_b_output_tokens").alias("tokens")
    ])

    monthly_sum = pl.concat([df_a, df_b]).group_by("month").agg(pl.col("tokens").sum())
    
    total_all_time = monthly_sum.select(pl.col("tokens").sum()).item()
    if total_all_time == 0: return None
    
    return monthly_sum.with_columns(
        (pl.col("tokens") / total_all_time * 100).alias("frequency")
    ).sort("month")


def panel_winrate_graph(df, model, version):
    if df is None: return None
    
    m_a = (pl.col("base_model_a") == model)
    m_b = (pl.col("base_model_b") == model)
    if version != "Tous":
        m_a &= (pl.col("version_a") == version)
        m_b &= (pl.col("version_b") == version)

    score_expr = (pl.when(pl.col("both_equal") == True).then(0.5)
                  .when(pl.col("chosen_base_model") == model).then(1.0)
                  .otherwise(0.0).alias("score"))

    df_a = df.filter(m_a).select([
        pl.col("timestamp").dt.strftime("%Y-%m").alias("month"),
        score_expr
    ])
    df_b = df.filter(m_b).select([
        pl.col("timestamp").dt.strftime("%Y-%m").alias("month"),
        score_expr
    ])

    return (pl.concat([df_a, df_b])
            .group_by("month")
            .agg((pl.col("score").mean() * 100).round(2).alias("winrate"))
            .sort("month"))


def plot_panel_graph(df, df2, model, version):
    df_freq = tokens_frequency_panel_graph(df, model, version)
    df_win = panel_winrate_graph(df2, model, version)
    
    if df_freq is None or df_freq.is_empty() or df_win is None or df_win.is_empty():
        return None

    merged = df_freq.join(df_win, on="month").to_pandas()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=merged["month"], y=merged["frequency"], name="Fréquence d'utilsation", mode="lines+markers", line=dict(color="#636EFA")),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=merged["month"], y=merged["winrate"], name="Winrate", mode="lines+markers", line=dict(color="#00CC96", dash="dot")),
        secondary_y=True
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), margin=dict(t=10, l=10, r=10, b=10),
        hovermode="x unified",
        height=150
    )

    fig.update_yaxes(title_text="Part de marché", ticksuffix="%", secondary_y=False)
    fig.update_yaxes(title_text="Winrate", ticksuffix="%", range=[0, 100], secondary_y=True)

    return fig


def nb_turn(df, model, version):
    if df is None: return 0
    
    cond_a = (pl.col("base_model_a") == model)
    cond_b = (pl.col("base_model_b") == model)
    
    if version != "Tous":
        cond_a &= (pl.col("version_a") == version)
        cond_b &= (pl.col("version_b") == version)

    res = df.filter(cond_a | cond_b).select(pl.col("conv_turns").sum()).item()
    
    return int(res) if res is not None else 0


def mean_turn(df, model, version):
    if df is None: return 0.0
    
    cond_a = (pl.col("base_model_a") == model)
    cond_b = (pl.col("base_model_b") == model)
    
    if version != "Tous":
        cond_a &= (pl.col("version_a") == version)
        cond_b &= (pl.col("version_b") == version)

    subset = df.filter(cond_a | cond_b)
    
    if subset.is_empty():
        return 0.0

    count_a = subset.filter(cond_a).height
    count_b = subset.filter(cond_b).height
    total_appearances = count_a + count_b
    
    sum_turns_a = subset.filter(cond_a).select(pl.col("conv_turns").sum()).item() or 0
    sum_turns_b = subset.filter(cond_b).select(pl.col("conv_turns").sum()).item() or 0
    
    res = (sum_turns_a + sum_turns_b) / total_appearances
    
    return round(res, 2)


def pick_rate(df, model, version):
    if df is None or "custom_models_selection" not in df.columns: 
        return "0.0%"
    
    manual_df = df.filter(pl.col("mode") == "custom")
    
    if manual_df.is_empty():
        return "0.0%"
    
    total_manual = manual_df.height

    cond = (pl.col("base_model_a") == model) | (pl.col("base_model_b") == model)
    if version != "Tous":
        cond = ((pl.col("base_model_a") == model) & (pl.col("version_a") == version)) | \
               ((pl.col("base_model_b") == model) & (pl.col("version_b") == version))

    is_selected = (
        pl.col("custom_models_selection")
        .list.join(" ")
        .str.to_lowercase()
        .str.contains(model.lower())
        .fill_null(False)
    )
    
    nb_picks = manual_df.filter(cond & is_selected).height
    
    rate = (nb_picks / total_manual) * 100
    
    return f"{round(rate, 1)}%"


def deep_engagement_rate(df, model, version):
    if df is None or "conv_turns" not in df.columns: 
        return "0.0%"
    
    cond = (pl.col("base_model_a") == model) | (pl.col("base_model_b") == model)
    if version != "Tous":
        cond = ((pl.col("base_model_a") == model) & (pl.col("version_a") == version)) | \
               ((pl.col("base_model_b") == model) & (pl.col("version_b") == version))
    
    subset = df.filter(cond)
    if subset.is_empty(): return "0.0%"
    
    long_convs = subset.filter(pl.col("conv_turns") > 2).height
    total = subset.height
    
    res = (long_convs / total) * 100
    return f"{round(res, 1)}%"


@st.cache_data
def get_global_benchmarks(df_react, df_conv):
    if df_react is None or df_react.is_empty():
        return {"u": 0.2, "c": 0.2, "f": 0.2, "i": 0.1, "inc": 0.1, "sup": 0.1, "tok": 500, "turn": 1.5}
    
    bench = df_react.select([
        pl.col("useful").mean().alias("u"),
        pl.col("creative").mean().alias("c"),
        pl.col("clear_formatting").mean().alias("f"),
        pl.col("instructions_not_followed").mean().alias("i"),
        pl.col("incorrect").mean().alias("inc"),
        pl.col("superficial").mean().alias("sup")
    ]).to_dicts()[0]
    
    avg_tokens = (df_conv.select([
        (pl.col("total_conv_a_output_tokens").mean() + pl.col("total_conv_b_output_tokens").mean()) / 2
    ]).item() or 500)
    
    avg_turns = df_conv.select(pl.col("conv_turns").mean()).item() or 1.5
    
    bench["tok"] = avg_tokens
    bench["turn"] = avg_turns
    return bench

def get_model_badges(df_react, df_conv, model, version):
    if df_react is None or df_react.is_empty(): return ""
    bench = get_global_benchmarks(df_react, df_conv)
    
    cond_r = pl.col("model") == model
    if version != "Tous": cond_r = cond_r & (pl.col("version") == version)
    subset_r = df_react.filter(cond_r)
    if subset_r.is_empty(): return ""

    scores = subset_r.select([
        pl.col("useful").mean().alias("u"),
        pl.col("creative").mean().alias("c"),
        pl.col("clear_formatting").mean().alias("f"),
        pl.col("instructions_not_followed").mean().alias("i"),
        pl.col("incorrect").mean().alias("inc"),
        pl.col("superficial").mean().alias("sup")
    ]).to_dicts()[0]

    from data.data_manager import nb_tokens_avg, mean_turn, winrate
    model_avg_tokens = nb_tokens_avg(df_conv, model, version)
    model_mean_turn = mean_turn(df_conv, model, version)
    model_winrate = winrate(pl.read_parquet("data/cleaned/vot.parquet"), model, version) or 0

    badges = []
    if (scores["u"] or 0) > (bench["u"] or 0):
        badges.append("<span class='badge-util'>🎯 Utile</span>")
    if (scores["c"] or 0) > (bench["c"] or 0):
        badges.append("<span class='badge-crea'>✨ Créatif</span>")
    if (scores["f"] or 0) > (bench["f"] or 0):
        badges.append("<span class='badge-clar'>📖 Clair</span>")
    if (scores["i"] or 1) < (bench["i"] or 1):
        badges.append("<span class='badge-warn'>⚠️ Discipliné</span>")
    if (scores["inc"] or 1) < (bench["inc"] or 1):
        badges.append("<span style='background-color: rgba(16, 185, 129, 0.1); color: #10b981; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; border: 1px solid rgba(16, 185, 129, 0.2);'>✅ Précis</span>")
    if (scores["sup"] or 1) < (bench["sup"] or 1):
        badges.append("<span style='background-color: rgba(99, 102, 241, 0.1); color: #6366f1; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; border: 1px solid rgba(63, 102, 241, 0.2);'>🛡️ Fiable</span>")
    if model_avg_tokens > bench["tok"]:
        badges.append("<span style='background-color: rgba(139, 92, 246, 0.1); color: #8b5cf6; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; border: 1px solid rgba(139, 92, 246, 0.2);'>🧠 Expert</span>")
    if model_mean_turn > bench["turn"]:
        badges.append("<span style='background-color: rgba(20, 184, 166, 0.1); color: #14b8a6; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; border: 1px solid rgba(20, 184, 166, 0.2);'>🤝 Captivant</span>")
    if model_winrate > 55:
        badges.append("<span style='background-color: rgba(239, 68, 68, 0.1); color: #ef4444; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; border: 1px solid rgba(239, 68, 68, 0.2);'>🔥 Challenger</span>")
    if subset_r.height > 500:
        badges.append("<span style='background-color: rgba(245, 158, 11, 0.1); color: #f59e0b; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; border: 1px solid rgba(245, 158, 11, 0.2);'>👑 Populaire</span>")
        
    return f"<div class='badge-container'>{' '.join(badges)}</div>" if badges else ""



BIAS_PATH = os.path.join("data", "bias_analysis", "bias_scores.parquet")
 
@st.cache_data
def load_bias_data() -> dict:
    """
    Charge bias_scores.parquet et retourne un dict {model: {...}} prêt à l'affichage.
    Toutes les valeurs viennent directement du fichier généré par run_bias.py.

    Returns:
        Dict indexé par nom de modèle, chaque entrée contenant scores, couvertures,
        distributions périodes/domaines/régions/types, top termes TF-IDF,
        badges et descriptions textuelles. Retourne {} si le fichier est absent.
    """
    if not os.path.exists(BIAS_PATH):
        return {}

    df   = pl.read_parquet(BIAS_PATH)
    data = {}

    for row in df.to_dicts():
        model      = row["model"]
        j_per      = json.loads(row.get("joconde_periodes") or "{}")
        j_dom      = json.loads(row.get("joconde_domaines") or "{}")
        b_reg      = json.loads(row.get("basilic_regions")  or "{}")
        b_typ      = json.loads(row.get("basilic_types")    or "{}")
        tfidf_d    = json.loads(row.get("tfidf_discriminants") or "[]")
        tfidf_c    = json.loads(row.get("tfidf_communs")       or "[]")
        j_score    = row.get("joconde_score",      0)
        b_score    = row.get("basilic_score",      0)
        j_cov      = row.get("joconde_couverture", 0)
        b_cov      = row.get("basilic_couverture", 0)

        j_badges, b_badges = [], []

        if j_per.get("Moyen Âge", {}).get("ia", 0) > 50:
            j_badges.append(("r", "Obsession Moyen Âge"))
        if j_per.get("XXe siècle", {}).get("ia", 0) < 15:
            j_badges.append(("a", "Art moderne sous-cité"))
        if j_score >= 35:
            j_badges.append(("r", "Fort biais historique"))
        else:
            j_badges.append(("g", "Adéquation historique OK"))

        if b_reg.get("Île-de-France", {}).get("ia", 0) > 25:
            b_badges.append(("r", "Parisiano-centrisme"))
        if b_typ.get("Théâtres & Opéras", {}).get("ia", 0) > 20:
            b_badges.append(("a", "Biais spectacle vivant"))
        if b_score >= 40:
            b_badges.append(("r", "Invisibilisation des régions"))

        if not j_badges: j_badges = [("b", "Score standard")]
        if not b_badges: b_badges = [("b", "Score standard")]

        data[model] = {
            "joconde_score":      j_score,
            "joconde_couverture": j_cov,
            "joconde_periodes":   j_per,
            "joconde_domaines":   j_dom,
            "joconde_badges":     j_badges,
            "joconde_desc":       "Analyse sémantique sur corpus filtré (conversations culturelles FR) vs 580k oeuvres Joconde.",
            "basilic_score":      b_score,
            "basilic_couverture": b_cov,
            "basilic_regions":    b_reg,
            "basilic_types":      b_typ,
            "basilic_badges":     b_badges,
            "basilic_desc":       f"Cartographie des équipements cités par {model} vs le maillage territorial Basilic (86k équipements).",
            "tfidf_discriminants": tfidf_d,
            "tfidf_communs":       tfidf_c,
        }

    return data
 
 
def get_bias_means(data: dict) -> dict:
    """Calcule les moyennes cross-modèles pour les 4 métriques principales."""
    import numpy as np
    if not data:
        return {k: 0 for k in ["joconde_score", "joconde_couverture", "basilic_score", "basilic_couverture"]}
    models = list(data.keys())
    return {
        k: round(float(np.mean([data[m][k] for m in models])), 2)
        for k in ["joconde_score", "joconde_couverture", "basilic_score", "basilic_couverture"]
    }
 
 
def bias_score_color(metric: str, value: float) -> str:
    """Retourne la classe CSS de couleur selon la métrique et la valeur."""
    if "couverture" in metric:
        return "green" if value >= 60 else ("amber" if value >= 45 else "red")
    return "green" if value <= 28 else ("amber" if value <= 35 else "red")
 
 
def render_bias_score_mini(label: str, value: float, color: str, suffix: str = "/100") -> str:
    return f"""<div>
<div class="score-label">{label}</div>
<div class="score-value-{color}">{value}{suffix}</div>
</div>"""
 
 
def render_bias_bar(label: str, ia_val: float, ref_val: float, max_val: float = 70) -> str:
    """
    Double barre IA (bleu) vs Référence (ambre/doré).
    La couleur ambre ressort bien sur fond sombre contrairement au gris.
    """
    ia_pct  = round(min(ia_val  / max_val * 100, 100), 1)
    ref_pct = round(min(ref_val / max_val * 100, 100), 1)
    diff    = round(ia_val - ref_val, 1)
    diff_color = "#F87171" if diff > 3 else ("#34D399" if diff < -3 else "#64748B")
    diff_str   = f"+{diff}" if diff > 0 else str(diff)
    return f"""<div style="margin-bottom:11px;">
<div style="display:flex;justify-content:space-between;margin-bottom:3px;">
<span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#94A3B8;">{label}</span>
<span style="font-family:'DM Mono',monospace;font-size:0.62rem;color:{diff_color};">{diff_str}%</span>
</div>
<div style="height:5px;background:#1E2130;border-radius:2px;margin-bottom:3px;">
<div style="height:100%;width:{ia_pct}%;background:#60A5FA;border-radius:2px;opacity:0.9;"></div>
</div>
<div style="height:4px;background:#1E2130;border-radius:2px;">
<div style="height:100%;width:{ref_pct}%;background:#F59E0B;border-radius:2px;opacity:0.75;"></div>
</div>
</div>"""
 
 
def render_bias_domain_bar(label: str, delta: float) -> str:
    """Barre centrée pour les écarts de domaine (positif = sur-représenté)."""
    color  = "#F87171" if delta > 5 else ("#34D399" if delta < -5 else "#64748B")
    sign   = f"+{delta}" if delta >= 0 else str(delta)
    width  = min(abs(delta) * 1.8, 50)
    left   = 50 if delta < 0 else 50 - width
    return f"""<div style="margin-bottom:9px;">
<div style="display:flex;justify-content:space-between;margin-bottom:3px;">
<span style="font-family:'DM Mono',monospace;font-size:0.63rem;color:#94A3B8;">{label}</span>
<span style="font-family:'DM Mono',monospace;font-size:0.62rem;color:{color};">{sign}pts</span>
</div>
<div style="position:relative;height:4px;background:#1E2130;border-radius:2px;">
<div style="position:absolute;top:0;height:100%;left:50%;width:1px;background:#2A3050;"></div>
<div style="position:absolute;height:100%;left:{left}%;width:{width}%;background:{color};border-radius:2px;opacity:0.85;"></div>
</div>
</div>"""
 
 
def bias_badges_html(badges: list) -> str:
    """Convertit une liste de badges (couleur, texte) en HTML."""
    return "".join(f'<span class="badge badge-{t}">{txt}</span>' for t, txt in badges)


def render_tfidf_terms(discriminants: list[str], communs: list[str]) -> str:
    """
    Génère un bloc HTML affichant les top termes TF-IDF en deux sections :
    - discriminants : termes qui distinguent ce modèle des autres (tags bleus)
    - communs       : termes culturels partagés par tous les modèles (tags gris)

    Args:
        discriminants : liste ordonnée par score TF-IDF décroissant.
        communs       : liste de termes culturels présents mais non discriminants.

    Returns:
        HTML string avec deux blocs de tags, ou chaîne vide si les deux listes sont vides.
    """
    if not discriminants and not communs:
        return ""

    def _tags(terms: list[str], color: str, border: str) -> str:
        return "".join(
            f'<span style="display:inline-block;background:rgba({color},0.08);color:rgba({color},1);'
            f'border:1px solid rgba({border},0.2);border-radius:4px;padding:2px 8px;'
            f'font-family:DM Mono,monospace;font-size:0.62rem;margin:2px;">{t}</span>'
            for t in terms
        )

    disc_html = ""
    if discriminants:
        disc_html = f"""
<div style="margin-bottom:10px;">
<div style="font-family:DM Mono,monospace;font-size:0.60rem;color:#475569;
text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">
Signature propre — distingue ce modèle des autres
</div>
<div style="display:flex;flex-wrap:wrap;gap:2px;">
{_tags(discriminants, "96,165,250", "96,165,250")}
</div>
</div>"""

    comm_html = ""
    if communs:
        comm_html = f"""
<div>
<div style="font-family:DM Mono,monospace;font-size:0.60rem;color:#475569;
text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">
Vocabulaire commun — partagé avec tous les modèles
</div>
<div style="display:flex;flex-wrap:wrap;gap:2px;">
{_tags(communs, "100,116,139", "100,116,139")}
</div>
</div>"""

    return disc_html + comm_html
 
 
BIAS_LEGEND_HTML = """<div style="display:flex;gap:16px;margin-bottom:12px;">
<span style="display:flex;align-items:center;gap:6px;font-size:0.62rem;color:#64748B;">
<span style="display:inline-block;width:20px;height:4px;background:#60A5FA;border-radius:2px;opacity:0.9;"></span> IA
</span>
<span style="display:flex;align-items:center;gap:6px;font-size:0.62rem;color:#64748B;">
<span style="display:inline-block;width:20px;height:4px;background:#F59E0B;border-radius:2px;opacity:0.75;"></span> Référence
</span>
</div>"""