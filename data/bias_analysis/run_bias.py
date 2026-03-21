import polars as pl
import numpy as np
import json
import os
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from collections import Counter

script_dir = os.path.dirname(os.path.abspath(__file__))

BIAS_CONV   = os.path.join(script_dir, "bias_conv.parquet")
JOCONDE_CSV = os.path.join(script_dir, "joconde.csv")
BASILIC_CSV = os.path.join(script_dir, "basilic.csv")
OUTPUT      = os.path.join(script_dir, "bias_scores.parquet")

# ---------------------------------------------------------------------------
# KEYWORDS — version robuste
# Règles : pas de mots courts ambigus, pas de sous-chaînes de mots courants,
# on préfère des termes composés ou suffisamment longs/spécifiques
# ---------------------------------------------------------------------------

PERIODES_JOCONDE = [
    "Moyen Âge", "Renaissance", "XVIIe siècle", "XVIIIe siècle",
    "XIXe siècle", "XXe siècle", "Art contemporain",
]

# PERIODES_KEYWORDS — deux dicts séparés :
# _REF  : calqué sur le format réel de Joconde ("10e siècle", "16e siècle"...)
# _CORPUS : termes utilisés dans les conversations IA

PERIODES_KEYWORDS_REF = {
    # Joconde utilise "Xe siècle" en chiffres arabes, ex: "10e siècle", "16e siècle"
    # Moyen Âge = du 5e au 15e siècle
    "Moyen Âge":        ["5e siècle", "6e siècle", "7e siècle", "8e siècle", "9e siècle",
                         "10e siècle", "11e siècle", "12e siècle", "13e siècle",
                         "14e siècle", "15e siècle", "médiéval", "gothique", "roman"],
    "Renaissance":      ["16e siècle", "renaissance", "maniérisme"],
    "XVIIe siècle":     ["17e siècle", "baroque"],
    "XVIIIe siècle":    ["18e siècle", "rococo", "néoclassique"],
    "XIXe siècle":      ["19e siècle", "romantisme", "impressionnisme", "réalisme",
                         "naturalisme", "symbolisme", "art nouveau"],
    "XXe siècle":       ["20e siècle", "cubisme", "surréalisme", "abstrait",
                         "fauvisme", "expressionnisme", "dadaïsme", "bauhaus"],
    "Art contemporain": ["21e siècle", "contemporain"],
}

PERIODES_KEYWORDS = {
    # Termes utilisés dans les conversations IA (registre courant)
    "Moyen Âge":        ["médiéval", "médiévale", "médiévaux", "gothique",
                         "moyen-âge", "moyen âge", "roman art", "féodal",
                         "moyen age", "xe siècle", "xie siècle", "xiie siècle",
                         "xiiie siècle", "xive siècle", "xve siècle"],
    "Renaissance":      ["renaissance", "xvie siècle", "humanisme", "maniérisme",
                         "quattrocento", "cinquecento", "16e siècle"],
    "XVIIe siècle":     ["xviie siècle", "baroque", "classicisme", "17e siècle",
                         "grand siècle", "louis xiv"],
    "XVIIIe siècle":    ["xviiie siècle", "rococo", "lumières", "18e siècle",
                         "néoclassique", "louis xv", "louis xvi"],
    "XIXe siècle":      ["xixe siècle", "romantisme", "romantique", "impressionnisme",
                         "impressionniste", "réalisme", "naturalisme", "symbolisme",
                         "19e siècle", "art nouveau"],
    "XXe siècle":       ["xxe siècle", "cubisme", "cubiste", "surréalisme", "surréaliste",
                         "abstrait", "abstraction", "dadaïsme", "fauvisme", "expressionnisme",
                         "20e siècle", "art moderne", "bauhaus"],
    "Art contemporain": ["art contemporain", "xxie siècle", "installation artistique",
                         "art numérique", "performance artistique", "21e siècle",
                         "street art", "land art"],
}

DOMAINES_JOCONDE = [
    "Peinture", "Sculpture", "Arts décoratifs", "Dessin",
    "Photographie", "Arts graphiques", "Archéologie",
]

DOMAINES_KEYWORDS = {
    # "photo" → trop ambigu (photovoltaïque, photographique technique...)
    # "verre" → ambigu (matériau, verre à boire...)
    # "décor" → ambigu (décoration intérieure, décor de théâtre...)
    "Peinture":        ["peinture", "tableau", "huile sur toile", "toile peinte",
                         "portrait peint", "aquarelle", "fresque", "peintre", "peindre"],
    "Sculpture":       ["sculpture", "sculpteur", "statue", "bas-relief",
                         "bronze sculptural", "marbre sculpté", "taille de pierre"],
    "Arts décoratifs": ["arts décoratifs", "tapisserie", "céramique", "orfèvrerie",
                         "faïence", "mobilier ancien", "ébénisterie", "vitrail"],
    "Dessin":          ["dessin", "esquisse", "crayon", "pastel", "gouache", "lavis",
                         "croquis", "dessinateur"],
    "Photographie":    ["photographie", "photographe", "cliché photographique",
                         "tirage argentique", "daguerréotype", "argentique"],
    "Arts graphiques": ["gravure", "estampe", "lithographie", "sérigraphie",
                         "xylographie", "taille-douce", "eau-forte"],
    "Archéologie":     ["archéologie", "archéologique", "fouille", "antiquité",
                         "gallo-romain", "préhistoire", "mérovingien", "néolithique",
                         "paléolithique", "site archéologique"],
}

REGIONS_FR = [
    "Île-de-France", "Auvergne-Rhône-Alpes", "Occitanie", "Nouvelle-Aquitaine",
    "Grand Est", "Hauts-de-France", "Bretagne", "Normandie", "Pays de la Loire",
    "Provence-Alpes-Côte d'Azur", "Centre-Val de Loire", "Bourgogne-Franche-Comté", "Corse",
]

REGIONS_KEYWORDS = {
    # On évite les mots trop courts ou trop génériques
    # "seine" → match "seine-saint-denis" mais aussi "saisine", "scène" proche...
    # On préfère des villes/noms propres clairement géolocalisés
    "Île-de-France":              ["paris", "versailles", "louvre", "île-de-france",
                                   "saint-denis", "vincennes", "fontainebleau"],
    "Auvergne-Rhône-Alpes":       ["lyon", "grenoble", "clermont-ferrand", "auvergne",
                                   "rhône-alpes", "annecy", "chambéry", "saint-étienne"],
    "Occitanie":                  ["toulouse", "montpellier", "occitanie", "languedoc",
                                   "nîmes", "perpignan", "carcassonne", "albi"],
    "Nouvelle-Aquitaine":         ["bordeaux", "limoges", "poitiers", "nouvelle-aquitaine",
                                   "périgueux", "biarritz", "bayonne", "angoulême"],
    "Grand Est":                  ["strasbourg", "reims", "nancy", "alsace", "lorraine",
                                   "champagne", "metz", "mulhouse", "colmar"],
    "Hauts-de-France":            ["lille", "amiens", "valenciennes", "picardie",
                                   "hauts-de-france", "boulogne", "dunkerque", "arras"],
    "Bretagne":                   ["rennes", "brest", "bretagne", "quimper",
                                   "saint-malo", "lorient", "vannes"],
    "Normandie":                  ["rouen", "caen", "normandie", "cherbourg",
                                   "le havre", "bayeux", "mont-saint-michel"],
    "Pays de la Loire":           ["angers", "le mans", "pays de la loire", "nantes",
                                   "saint-nazaire", "la roche-sur-yon"],
    "Provence-Alpes-Côte d'Azur": ["marseille", "nice", "avignon", "provence",
                                   "toulon", "aix-en-provence", "cannes", "antibes"],
    "Centre-Val de Loire":        ["tours", "orléans", "blois", "val de loire",
                                   "bourges", "châteauroux", "amboise", "chenonceau"],
    "Bourgogne-Franche-Comté":    ["dijon", "besançon", "bourgogne", "franche-comté",
                                   "auxerre", "chalon-sur-saône", "mâcon"],
    "Corse":                      ["ajaccio", "bastia", "corse", "bonifacio", "corte"],
}

# Types alignés sur les vraies valeurs Basilic :
# ['Bibliothèque', 'Centre culturel', "Centre d'art", 'Centre de création artistique',
#  'Centre de création musicale', 'Cinéma', 'Conservatoire', 'Espace protégé',
#  'Librairie', 'Lieu archéologique', 'Lieu de mémoire', 'Monument', 'Musée',
#  'Opéra', 'Parc et jardin', 'Scène', "Service d'archives", 'Théâtre']
TYPES_EQUIPEMENT = [
    "Musées", "Bibliothèques", "Théâtres & Opéras", "Cinémas",
    "Centres culturels & art", "Archives", "Monuments & lieux",
]

# RÉFÉRENCE Basilic → keywords sur la colonne Type_equipement (valeurs exactes)
TYPES_KEYWORDS_REF = {
    "Musées":                 ["musée"],
    "Bibliothèques":          ["bibliothèque", "librairie"],
    "Théâtres & Opéras":      ["théâtre", "opéra", "scène"],
    "Cinémas":                ["cinéma"],
    "Centres culturels & art": ["centre culturel", "centre d'art",
                                "centre de création artistique",
                                "centre de création musicale", "conservatoire"],
    "Archives":               ["service d'archives"],
    "Monuments & lieux":      ["monument", "lieu archéologique",
                               "lieu de mémoire", "espace protégé",
                               "parc et jardin"],
}

# MATCHING corpus IA → termes utilisés dans les conversations
TYPES_KEYWORDS = {
    "Musées":                 ["musée", "galerie d'art", "beaux-arts", "pinacothèque"],
    "Bibliothèques":          ["bibliothèque", "médiathèque", "lecture publique", "librairie"],
    "Théâtres & Opéras":      ["théâtre", "opéra", "scène nationale", "spectacle vivant",
                               "comédie-française", "cirque"],
    "Cinémas":                ["cinéma", "cinémathèque", "festival de cinéma", "ciné-club"],
    "Centres culturels & art": ["centre culturel", "centre d'art", "maison de la culture",
                                "conservatoire", "école des beaux-arts", "micro-folie"],
    "Archives":               ["archives nationales", "archives départementales",
                               "service d'archives", "fonds d'archives"],
    "Monuments & lieux":      ["monument historique", "château", "cathédrale", "abbaye",
                               "lieu de mémoire", "site archéologique", "dolmen",
                               "parc naturel", "jardin historique"],
}

# Filtre d'entrée : catégories culturelles pertinentes
CULTURAL_FILTER = (
    r"(?i)(art|peint|sculpt|musée|histoire|culture|théâtre|cinéma|monument|"
    r"bibliothèque|architec|patrimoine|archéolog|littérature|spectacle|"
    r"artiste|auteur|gothique|baroque|impressionn|renaissance|médiév|"
    r"château|cathédrale|abbaye|exposition|galerie)"
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def col_to_list(series: pl.Series) -> list[str]:
    return [str(v).strip().lower() for v in series.drop_nulls().to_list() if str(v).strip()]


def build_keyword_patterns(kw_dict: dict) -> dict:
    """
    Pré-compile un pattern regex unique par catégorie
    au lieu de boucler sur chaque keyword séparément.
    """
    compiled = {}
    for cat, kws in kw_dict.items():
        # Trie par longueur décroissante pour que les expressions longues matchent en premier
        sorted_kws = sorted(kws, key=len, reverse=True)
        pattern = r'\b(?:' + '|'.join(re.escape(k) for k in sorted_kws) + r')\b'
        compiled[cat] = re.compile(pattern)
    return compiled


def count_in_corpus(corpus: str, pattern: re.Pattern) -> int:
    """Compte les occurrences d'un pattern pré-compilé dans le corpus."""
    return len(pattern.findall(corpus))


# ---------------------------------------------------------------------------
# CHARGEMENT
# ---------------------------------------------------------------------------
def extract_conv_text_vectorized(df: pl.DataFrame, col: str) -> pl.Series:
    return (
        df[col]
        .list.eval(pl.element().struct.field("content").fill_null(""))
        .list.join(" ")
        .str.to_lowercase()
        .fill_null("")
    )


def filter_and_extract_texts_by_model(
    df: pl.DataFrame, max_docs: int = 2000
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    1. Filtre sur les conversations à contenu culturel via categories/keywords
    2. Extrait le texte des conversations (position A → modèle A, B → modèle B)
    3. Échantillonne à max_docs par modèle pour la perf
    """
    df = df.filter(
        pl.col("categories").list.join(" ").fill_null("").str.contains(CULTURAL_FILTER) |
        pl.col("keywords").list.join(" ").fill_null("").str.contains(CULTURAL_FILTER)
    )
    print(f"  {df.height} conversations culturelles retenues après filtrage")

    print("  Parsing conversation_a...")
    text_a = extract_conv_text_vectorized(df, "conversation_a")
    print("  Parsing conversation_b...")
    text_b = extract_conv_text_vectorized(df, "conversation_b")

    df = df.with_columns([
        text_a.alias("_text_a"),
        text_b.alias("_text_b"),
    ])

    all_models = sorted(set(
        df["base_model_a"].drop_nulls().unique().to_list() +
        df["base_model_b"].drop_nulls().unique().to_list()
    ))

    texts: dict[str, list[str]] = {}
    full_corpus: dict[str, str] = {}
    random.seed(42)

    for model in tqdm(all_models, desc="Échantillonnage par modèle"):
        parts_a = [t for t in df.filter(pl.col("base_model_a") == model)["_text_a"].to_list() if t.strip()]
        parts_b = [t for t in df.filter(pl.col("base_model_b") == model)["_text_b"].to_list() if t.strip()]
        parts = parts_a + parts_b

        if len(parts) > max_docs:
            parts = random.sample(parts, max_docs)

        if parts:
            texts[model] = parts
            full_corpus[model] = " ".join(parts)

    return texts, full_corpus


def load_joconde(path: str) -> pl.DataFrame:
    return pl.read_csv(
        path, separator="|", infer_schema_length=0,
        ignore_errors=True, encoding="utf8",
    ).select(["Auteur", "Domaine", "Periode_de_creation", "Region", "Nom_officiel_musee"])


def load_basilic(path: str) -> pl.DataFrame:
    return pl.read_csv(
        path, separator=";", infer_schema_length=0,
        ignore_errors=True, encoding="utf8",
    ).select([
        "Nom", "Type équipement ou lieu", "Domaine",
        "Région", "Département", "libelle_geographique",
    ]).rename({
        "Type équipement ou lieu": "Type_equipement",
        "Région":                  "Region",
        "libelle_geographique":    "Commune",
    })


# ---------------------------------------------------------------------------
# TF-IDF (conservé pour usage futur / debug)
# ---------------------------------------------------------------------------
def tfidf_top_terms(
    texts_by_model: dict[str, list[str]], top_n: int = 300
) -> dict[str, set[str]]:
    all_models = list(texts_by_model.keys())
    sampled = [" ".join(v) for v in texts_by_model.values()]

    vec = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        strip_accents=None,
    )
    vec.fit(sampled)
    vocab = np.array(vec.get_feature_names_out())

    top_terms = {}
    for model in tqdm(all_models, desc="Calcul TF-IDF"):
        doc = " ".join(texts_by_model[model])
        tfidf_vec = vec.transform([doc]).toarray()[0]
        top_idx = np.argsort(tfidf_vec)[::-1][:top_n]
        top_terms[model] = set(vocab[top_idx])

    return top_terms


# ---------------------------------------------------------------------------
# SCORES JOCONDE
# ---------------------------------------------------------------------------
def compute_joconde_scores(
    full_corpus: dict[str, str], df_joconde: pl.DataFrame,
    periodes_patterns: dict, domaines_patterns: dict,
) -> dict[str, dict]:

    # Référence périodes — utilise PERIODES_KEYWORDS_REF calqué sur le format Joconde
    # ("10e siècle", "16e siècle"...) et non le registre courant des conversations
    periode_col = col_to_list(df_joconde["Periode_de_creation"])
    ref_p_patterns = build_keyword_patterns(PERIODES_KEYWORDS_REF)
    ref_p = {
        p: sum(1 for v in periode_col if ref_p_patterns[p].search(v))
        for p in PERIODES_KEYWORDS
    }
    total_p = sum(ref_p.values()) or 1
    ref_p_pct = {p: round(c / total_p * 100, 1) for p, c in ref_p.items()}

    # Référence domaines
    domaine_col = col_to_list(df_joconde["Domaine"])
    ref_d = {
        d: sum(1 for v in domaine_col if count_in_corpus(v, domaines_patterns[d]) > 0)
        for d in DOMAINES_KEYWORDS
    }
    total_d = sum(ref_d.values()) or 1
    ref_d_pct = {d: round(c / total_d * 100, 1) for d, c in ref_d.items()}

    # Top artistes Joconde pour couverture
    artistes_counter: Counter = Counter()
    for v in col_to_list(df_joconde["Auteur"]):
        for part in re.split(r"[;,/]", v):
            w = part.strip()
            if len(w) > 5:
                artistes_counter[w] += 1
    artistes_ref = {a for a, _ in artistes_counter.most_common(1500)}

    results = {}
    for model in tqdm(full_corpus.keys(), desc="Scores Joconde"):
        corpus = full_corpus[model]

        # Fréquences IA par période
        ia_p_raw = {p: count_in_corpus(corpus, periodes_patterns[p]) for p in PERIODES_KEYWORDS}
        total_ia_p = sum(ia_p_raw.values()) or 1
        ia_p_pct = {p: round(c / total_ia_p * 100, 1) for p, c in ia_p_raw.items()}

        # Fréquences IA par domaine
        ia_d_raw = {d: count_in_corpus(corpus, domaines_patterns[d]) for d in DOMAINES_KEYWORDS}
        total_ia_d = sum(ia_d_raw.values()) or 1
        ia_d_pct = {d: round(c / total_ia_d * 100, 1) for d, c in ia_d_raw.items()}

        # Couverture artistes
        matched = sum(1 for a in artistes_ref if a in corpus)
        couverture = round(min(matched / max(len(artistes_ref) * 0.01, 1), 100), 1)

        # Score biais = MAE normalisée entre distribution IA et référence
        mae_p = float(np.mean([abs(ia_p_pct[p] - ref_p_pct[p]) for p in PERIODES_JOCONDE]))
        mae_d = float(np.mean([abs(ia_d_pct[d] - ref_d_pct[d]) for d in DOMAINES_JOCONDE]))
        # On normalise : MAE max théorique ≈ 100, on scale sur 0-100
        score = round(min((mae_p + mae_d) * 1.2, 100), 1)

        results[model] = {
            "joconde_score":      score,
            "joconde_couverture": couverture,
            "joconde_periodes":   {p: {"ia": ia_p_pct[p], "ref": ref_p_pct[p]} for p in PERIODES_JOCONDE},
            "joconde_domaines":   {d: round(ia_d_pct[d] - ref_d_pct[d], 1) for d in DOMAINES_JOCONDE},
        }

    return results


# ---------------------------------------------------------------------------
# SCORES BASILIC
# ---------------------------------------------------------------------------
def compute_basilic_scores(
    full_corpus: dict[str, str], df_basilic: pl.DataFrame,
    regions_patterns: dict, types_patterns: dict,
) -> dict[str, dict]:

    # Référence régions
    regions_col = col_to_list(df_basilic["Region"])
    ref_r = {
        r: sum(1 for v in regions_col if count_in_corpus(v, regions_patterns[r]) > 0)
        for r in REGIONS_KEYWORDS
    }
    total_r = sum(ref_r.values()) or 1
    ref_r_pct = {r: round(c / total_r * 100, 1) for r, c in ref_r.items()}

    # Référence types
    # Référence : on utilise les vrais types Basilic (valeurs exactes de la colonne)
    types_col = col_to_list(df_basilic["Type_equipement"])
    ref_t_patterns = build_keyword_patterns(TYPES_KEYWORDS_REF)
    ref_t = {
        t: sum(1 for v in types_col if ref_t_patterns[t].search(v))
        for t in TYPES_KEYWORDS
    }
    total_t = sum(ref_t.values()) or 1
    ref_t_pct = {t: round(c / total_t * 100, 1) for t, c in ref_t.items()}

    # Top lieux pour couverture
    lieux_counter: Counter = Counter()
    for v in col_to_list(df_basilic["Nom"]):
        if len(v) > 6:
            lieux_counter[v] += 1
    lieux_ref = {l for l, _ in lieux_counter.most_common(1500)}

    results = {}
    for model in tqdm(full_corpus.keys(), desc="Scores Basilic"):
        corpus = full_corpus[model]

        ia_r_raw = {r: count_in_corpus(corpus, regions_patterns[r]) for r in REGIONS_KEYWORDS}
        total_ia_r = sum(ia_r_raw.values()) or 1
        ia_r_pct = {r: round(c / total_ia_r * 100, 1) for r, c in ia_r_raw.items()}

        ia_t_raw = {t: count_in_corpus(corpus, types_patterns[t]) for t in TYPES_KEYWORDS}
        total_ia_t = sum(ia_t_raw.values()) or 1
        ia_t_pct = {t: round(c / total_ia_t * 100, 1) for t, c in ia_t_raw.items()}

        matched = sum(1 for l in lieux_ref if l in corpus)
        couverture = round(min(matched / max(len(lieux_ref) * 0.005, 1), 100), 1)

        mae_r = float(np.mean([abs(ia_r_pct[r] - ref_r_pct[r]) for r in REGIONS_FR if r in ref_r_pct]))
        mae_t = float(np.mean([abs(ia_t_pct[t] - ref_t_pct[t]) for t in TYPES_EQUIPEMENT]))
        score = round(min((mae_r + mae_t) * 1.5, 100), 1)

        results[model] = {
            "basilic_score":      score,
            "basilic_couverture": couverture,
            "basilic_regions":    {r: {"ia": ia_r_pct[r], "ref": ref_r_pct.get(r, 0)} for r in REGIONS_FR},
            "basilic_types":      {t: {"ia": ia_t_pct[t], "ref": ref_t_pct[t]} for t in TYPES_EQUIPEMENT},
        }

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("[1/5] Chargement des fichiers...")
    df_bias    = pl.read_parquet(BIAS_CONV)
    df_joconde = load_joconde(JOCONDE_CSV)
    df_basilic = load_basilic(BASILIC_CSV)
    print(f"  bias_conv : {df_bias.height} lignes")
    print(f"  joconde   : {df_joconde.height} œuvres")
    print(f"  basilic   : {df_basilic.height} équipements")

    print("\n[2/5] Filtrage et extraction des conversations culturelles...")
    texts_by_model, full_corpus = filter_and_extract_texts_by_model(df_bias)
    models = sorted(texts_by_model.keys())
    print(f"  {len(models)} modèles extraits")
    for m, txts in texts_by_model.items():
        print(f"    {m} : {len(txts)} docs")

    print("\n[3/5] TF-IDF (pour debug / usage futur)...")
    top_terms = tfidf_top_terms(texts_by_model, top_n=300)

    print("\n[4/5] Scores Joconde (périodes, domaines, couverture artistes)...")
    periodes_patterns = build_keyword_patterns(PERIODES_KEYWORDS)  # pour corpus IA
    domaines_patterns = build_keyword_patterns(DOMAINES_KEYWORDS)
    scores_j = compute_joconde_scores(full_corpus, df_joconde, periodes_patterns, domaines_patterns)

    print("\n[5/5] Scores Basilic (régions, types, couverture lieux)...")
    regions_patterns = build_keyword_patterns(REGIONS_KEYWORDS)
    types_patterns   = build_keyword_patterns(TYPES_KEYWORDS)
    scores_b = compute_basilic_scores(full_corpus, df_basilic, regions_patterns, types_patterns)

    print("\nSauvegarde...")
    rows = []
    for model in models:
        j = scores_j.get(model, {})
        b = scores_b.get(model, {})
        rows.append({
            "model":              model,
            "joconde_score":      j.get("joconde_score", 0),
            "joconde_couverture": j.get("joconde_couverture", 0),
            "joconde_periodes":   json.dumps(j.get("joconde_periodes", {}), ensure_ascii=False),
            "joconde_domaines":   json.dumps(j.get("joconde_domaines", {}), ensure_ascii=False),
            "basilic_score":      b.get("basilic_score", 0),
            "basilic_couverture": b.get("basilic_couverture", 0),
            "basilic_regions":    json.dumps(b.get("basilic_regions", {}), ensure_ascii=False),
            "basilic_types":      json.dumps(b.get("basilic_types", {}), ensure_ascii=False),
        })

    df_out = pl.DataFrame(rows)
    df_out.write_parquet(OUTPUT)
    print(f"\nTerminé → {OUTPUT}")
    print(df_out.select(["model", "joconde_score", "joconde_couverture", "basilic_score", "basilic_couverture"]))


if __name__ == "__main__":
    main()