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

PERIODES_JOCONDE = [
    "Moyen Âge", "Renaissance", "XVIIe siècle", "XVIIIe siècle",
    "XIXe siècle", "XXe siècle", "Art contemporain",
]

PERIODES_KEYWORDS_REF = {
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

TYPES_EQUIPEMENT = [
    "Musées", "Bibliothèques", "Théâtres & Opéras", "Cinémas",
    "Centres culturels & art", "Archives", "Monuments & lieux",
]

TYPES_KEYWORDS_REF = {
    "Musées":                  ["musée"],
    "Bibliothèques":           ["bibliothèque", "librairie"],
    "Théâtres & Opéras":       ["théâtre", "opéra", "scène"],
    "Cinémas":                 ["cinéma"],
    "Centres culturels & art": ["centre culturel", "centre d'art",
                                "centre de création artistique",
                                "centre de création musicale", "conservatoire"],
    "Archives":                ["service d'archives"],
    "Monuments & lieux":       ["monument", "lieu archéologique",
                                "lieu de mémoire", "espace protégé",
                                "parc et jardin"],
}

TYPES_KEYWORDS = {
    "Musées":                  ["musée", "galerie d'art", "beaux-arts", "pinacothèque"],
    "Bibliothèques":           ["bibliothèque", "médiathèque", "lecture publique", "librairie"],
    "Théâtres & Opéras":       ["théâtre", "opéra", "scène nationale", "spectacle vivant",
                                "comédie-française", "cirque"],
    "Cinémas":                 ["cinéma", "cinémathèque", "festival de cinéma", "ciné-club"],
    "Centres culturels & art": ["centre culturel", "centre d'art", "maison de la culture",
                                "conservatoire", "école des beaux-arts", "micro-folie"],
    "Archives":                ["archives nationales", "archives départementales",
                                "service d'archives", "fonds d'archives"],
    "Monuments & lieux":       ["monument historique", "château", "cathédrale", "abbaye",
                                "lieu de mémoire", "site archéologique", "dolmen",
                                "parc naturel", "jardin historique"],
}

CULTURAL_FILTER = (
    r"(?i)(art|peint|sculpt|musée|histoire|culture|théâtre|cinéma|monument|"
    r"bibliothèque|architec|patrimoine|archéolog|littérature|spectacle|"
    r"artiste|auteur|gothique|baroque|impressionn|renaissance|médiév|"
    r"château|cathédrale|abbaye|exposition|galerie)"
)


def col_to_list(series: pl.Series) -> list[str]:
    """Convertit une Series Polars en liste de chaînes nettoyées (strip + lower, nulls exclus)."""
    return [str(v).strip().lower() for v in series.drop_nulls().to_list() if str(v).strip()]


def build_keyword_patterns(kw_dict: dict) -> dict:
    """
    Pré-compile un pattern regex unique par catégorie depuis un dict {catégorie: [keywords]}.

    Les keywords sont triés par longueur décroissante pour que les expressions longues
    matchent en priorité et éviter les faux positifs sur les sous-chaînes.
    """
    compiled = {}
    for cat, kws in kw_dict.items():
        sorted_kws = sorted(kws, key=len, reverse=True)
        pattern    = r"\b(?:" + "|".join(re.escape(k) for k in sorted_kws) + r")\b"
        compiled[cat] = re.compile(pattern)
    return compiled


def count_in_corpus(corpus: str, pattern: re.Pattern) -> int:
    """Compte les occurrences d'un pattern pré-compilé dans un corpus texte."""
    return len(pattern.findall(corpus))


def extract_conv_text_vectorized(df: pl.DataFrame, col: str) -> pl.Series:
    """
    Extrait et concatène le contenu de tous les messages d'une colonne de conversations.

    Chaque élément est une liste de structs {role, content}. La fonction joint tous les
    champs content en une seule chaîne par ligne, en minuscules.
    """
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
    Filtre les conversations à contenu culturel et construit les corpus texte par modèle.

    Etapes :
    1. Filtre sur categories/keywords via CULTURAL_FILTER.
    2. Extrait le texte des conversations (position A -> modèle A, B -> modèle B).
    3. Echantillonne à max_docs par modèle pour la performance (seed=42 pour reproductibilité).

    Returns:
        Tuple (texts_by_model, full_corpus) :
        - texts_by_model : {modèle: liste de docs séparés} — granularité nécessaire au TF-IDF
        - full_corpus    : {modèle: un seul bloc texte}    — optimisé pour le matching regex
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

    df = df.with_columns([text_a.alias("_text_a"), text_b.alias("_text_b")])

    all_models = sorted(set(
        df["base_model_a"].drop_nulls().unique().to_list() +
        df["base_model_b"].drop_nulls().unique().to_list()
    ))

    texts: dict[str, list[str]] = {}
    full_corpus: dict[str, str] = {}
    random.seed(42)

    for model in tqdm(all_models, desc="Echantillonnage par modèle"):
        parts_a = [t for t in df.filter(pl.col("base_model_a") == model)["_text_a"].to_list() if t.strip()]
        parts_b = [t for t in df.filter(pl.col("base_model_b") == model)["_text_b"].to_list() if t.strip()]
        parts   = parts_a + parts_b

        if len(parts) > max_docs:
            parts = random.sample(parts, max_docs)

        if parts:
            texts[model]       = parts
            full_corpus[model] = " ".join(parts)

    return texts, full_corpus


def load_joconde(path: str) -> pl.DataFrame:
    """Charge le fichier CSV Joconde (séparateur |) et sélectionne les colonnes utiles."""
    return pl.read_csv(
        path, separator="|", infer_schema_length=0,
        ignore_errors=True, encoding="utf8",
    ).select(["Auteur", "Domaine", "Periode_de_creation", "Region", "Nom_officiel_musee"])


def load_basilic(path: str) -> pl.DataFrame:
    """
    Charge le fichier CSV Basilic (séparateur ;) et sélectionne les colonnes utiles.

    Renomme les colonnes pour homogénéiser les accès dans le reste du pipeline.
    """
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


def _build_cultural_keywords_set() -> set[str]:
    """
    Construit un ensemble de tous les keywords culturels issus des dictionnaires
    Joconde et Basilic — utilisé pour filtrer les top termes TF-IDF.
    """
    cultural = set()
    for kw_dict in [PERIODES_KEYWORDS, DOMAINES_KEYWORDS, REGIONS_KEYWORDS, TYPES_KEYWORDS]:
        for kws in kw_dict.values():
            cultural.update(kws)
    return cultural


def tfidf_top_terms(
    texts_by_model: dict[str, list[str]], top_n: int = 30
) -> dict[str, dict]:
    """
    Calcule les top termes TF-IDF pour chaque modèle en deux catégories :

    - discriminants : termes culturels avec score TF-IDF élevé — ce qui distingue
      ce modèle des autres. Filtrés sur les keywords exacts des dictionnaires.
    - communs : termes culturels partagés par tous les modèles (score TF-IDF faible
      ou nul) — comme "paris" ou "impressionnisme" que tous les modèles utilisent
      de la même façon. Ces termes contribuent aux scores de biais mais ne discriminent pas.

    Paramètres TF-IDF :
    - max_features=10000 : limite le vocabulaire aux 10k mots les plus fréquents
    - ngram_range=(1,2)  : capture les expressions comme "art contemporain"
    - min_df=2           : ignore les termes présents dans moins de 2 corpus
    - max_df=0.85        : ignore les termes trop communs (présents dans 85%+ des corpus)

    Returns:
        {modèle: {"discriminants": [...], "communs": [...]}}
    """
    cultural_keywords = _build_cultural_keywords_set()
    sampled = [" ".join(v) for v in texts_by_model.values()]

    vec = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        strip_accents=None,
    )
    vec.fit(sampled)
    vocab      = np.array(vec.get_feature_names_out())
    vocab_set  = set(vocab.tolist())

    top_terms = {}
    for model in tqdm(texts_by_model.keys(), desc="Calcul TF-IDF"):
        doc       = " ".join(texts_by_model[model])
        tfidf_vec = vec.transform([doc]).toarray()[0]
        top_idx   = np.argsort(tfidf_vec)[::-1]

        discriminants = [
            vocab[i] for i in top_idx
            if tfidf_vec[i] > 0 and vocab[i] in cultural_keywords
        ][:top_n]

        discriminants_set = set(discriminants)
        communs = [
            kw for kw in cultural_keywords
            if kw in vocab_set
            and kw not in discriminants_set
            and tfidf_vec[list(vocab).index(kw)] == 0
        ][:20]

        top_terms[model] = {
            "discriminants": discriminants,
            "communs":       communs,
        }

    return top_terms


def compute_joconde_scores(
    full_corpus: dict[str, str],
    df_joconde: pl.DataFrame,
    periodes_patterns: dict,
    domaines_patterns: dict,
) -> dict[str, dict]:
    """
    Calcule les scores de biais Joconde pour chaque modèle.

    Métriques :
    - joconde_score     : MAE normalisée (périodes + domaines) × 1.2, plafonnée à 100
    - joconde_couverture: % des 1500 artistes Joconde les plus fréquents cités dans le corpus
    - joconde_periodes  : distribution comparée IA vs Joconde par période
    - joconde_domaines  : écarts IA - référence par domaine artistique

    Note : deux dictionnaires de keywords séparés — PERIODES_KEYWORDS_REF calqué sur
    le format Joconde ("10e siècle") et PERIODES_KEYWORDS calqué sur le registre IA
    ("XIXe siècle", "romantisme") — pour comparer des vocabulaires adaptés à chaque source.
    """
    periode_col    = col_to_list(df_joconde["Periode_de_creation"])
    ref_p_patterns = build_keyword_patterns(PERIODES_KEYWORDS_REF)
    ref_p = {
        p: sum(1 for v in periode_col if ref_p_patterns[p].search(v))
        for p in PERIODES_KEYWORDS
    }
    total_p   = sum(ref_p.values()) or 1
    ref_p_pct = {p: round(c / total_p * 100, 1) for p, c in ref_p.items()}

    domaine_col = col_to_list(df_joconde["Domaine"])
    ref_d = {
        d: sum(1 for v in domaine_col if count_in_corpus(v, domaines_patterns[d]) > 0)
        for d in DOMAINES_KEYWORDS
    }
    total_d   = sum(ref_d.values()) or 1
    ref_d_pct = {d: round(c / total_d * 100, 1) for d, c in ref_d.items()}

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

        ia_p_raw   = {p: count_in_corpus(corpus, periodes_patterns[p]) for p in PERIODES_KEYWORDS}
        total_ia_p = sum(ia_p_raw.values()) or 1
        ia_p_pct   = {p: round(c / total_ia_p * 100, 1) for p, c in ia_p_raw.items()}

        ia_d_raw   = {d: count_in_corpus(corpus, domaines_patterns[d]) for d in DOMAINES_KEYWORDS}
        total_ia_d = sum(ia_d_raw.values()) or 1
        ia_d_pct   = {d: round(c / total_ia_d * 100, 1) for d, c in ia_d_raw.items()}

        matched    = sum(1 for a in artistes_ref if a in corpus)
        couverture = round(min(matched / max(len(artistes_ref) * 0.01, 1), 100), 1)

        mae_p = float(np.mean([abs(ia_p_pct[p] - ref_p_pct[p]) for p in PERIODES_JOCONDE]))
        mae_d = float(np.mean([abs(ia_d_pct[d] - ref_d_pct[d]) for d in DOMAINES_JOCONDE]))
        score = round(min((mae_p + mae_d) * 1.2, 100), 1)

        results[model] = {
            "joconde_score":      score,
            "joconde_couverture": couverture,
            "joconde_periodes":   {p: {"ia": ia_p_pct[p], "ref": ref_p_pct[p]} for p in PERIODES_JOCONDE},
            "joconde_domaines":   {d: round(ia_d_pct[d] - ref_d_pct[d], 1) for d in DOMAINES_JOCONDE},
        }

    return results


def compute_basilic_scores(
    full_corpus: dict[str, str],
    df_basilic: pl.DataFrame,
    regions_patterns: dict,
    types_patterns: dict,
) -> dict[str, dict]:
    """
    Calcule les scores de biais Basilic pour chaque modèle.

    Métriques :
    - basilic_score     : MAE normalisée (régions + types) × 1.5, plafonnée à 100
    - basilic_couverture: % des 1500 lieux Basilic les plus fréquents cités dans le corpus
    - basilic_regions   : distribution comparée IA vs Basilic par région française
    - basilic_types     : distribution comparée IA vs Basilic par type d'équipement

    Note : la référence types utilise les valeurs exactes de la colonne Type_equipement
    de Basilic via TYPES_KEYWORDS_REF, tandis que le corpus IA utilise TYPES_KEYWORDS
    qui reflète le vocabulaire des conversations.
    """
    regions_col = col_to_list(df_basilic["Region"])
    ref_r = {
        r: sum(1 for v in regions_col if count_in_corpus(v, regions_patterns[r]) > 0)
        for r in REGIONS_KEYWORDS
    }
    total_r   = sum(ref_r.values()) or 1
    ref_r_pct = {r: round(c / total_r * 100, 1) for r, c in ref_r.items()}

    types_col      = col_to_list(df_basilic["Type_equipement"])
    ref_t_patterns = build_keyword_patterns(TYPES_KEYWORDS_REF)
    ref_t = {
        t: sum(1 for v in types_col if ref_t_patterns[t].search(v))
        for t in TYPES_KEYWORDS
    }
    total_t   = sum(ref_t.values()) or 1
    ref_t_pct = {t: round(c / total_t * 100, 1) for t, c in ref_t.items()}

    lieux_counter: Counter = Counter()
    for v in col_to_list(df_basilic["Nom"]):
        if len(v) > 6:
            lieux_counter[v] += 1
    lieux_ref = {l for l, _ in lieux_counter.most_common(1500)}

    results = {}
    for model in tqdm(full_corpus.keys(), desc="Scores Basilic"):
        corpus = full_corpus[model]

        ia_r_raw   = {r: count_in_corpus(corpus, regions_patterns[r]) for r in REGIONS_KEYWORDS}
        total_ia_r = sum(ia_r_raw.values()) or 1
        ia_r_pct   = {r: round(c / total_ia_r * 100, 1) for r, c in ia_r_raw.items()}

        ia_t_raw   = {t: count_in_corpus(corpus, types_patterns[t]) for t in TYPES_KEYWORDS}
        total_ia_t = sum(ia_t_raw.values()) or 1
        ia_t_pct   = {t: round(c / total_ia_t * 100, 1) for t, c in ia_t_raw.items()}

        matched    = sum(1 for l in lieux_ref if l in corpus)
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


def main():
    """
    Pipeline de calcul des scores de biais culturel — 5 étapes.

    1. Chargement : bias_conv.parquet + joconde.csv + basilic.csv
    2. Filtrage culturel + extraction des corpus texte par modèle
    3. TF-IDF : top 30 termes caractéristiques par modèle, stockés dans le parquet final
    4. Scores Joconde : périodes, domaines, couverture artistes
    5. Scores Basilic : régions, types d'équipements, couverture lieux
    Sortie : bias_scores.parquet avec tfidf_top_terms comme nouvelle colonne
    """
    print("[1/5] Chargement des fichiers...")
    df_bias    = pl.read_parquet(BIAS_CONV)
    df_joconde = load_joconde(JOCONDE_CSV)
    df_basilic = load_basilic(BASILIC_CSV)
    print(f"  bias_conv : {df_bias.height} lignes")
    print(f"  joconde   : {df_joconde.height} oeuvres")
    print(f"  basilic   : {df_basilic.height} équipements")

    print("\n[2/5] Filtrage et extraction des conversations culturelles...")
    texts_by_model, full_corpus = filter_and_extract_texts_by_model(df_bias)
    models = sorted(texts_by_model.keys())
    print(f"  {len(models)} modèles extraits")
    for m, txts in texts_by_model.items():
        print(f"    {m} : {len(txts)} docs")

    print("\n[3/5] TF-IDF — top 30 termes caractéristiques par modèle...")
    top_terms = tfidf_top_terms(texts_by_model, top_n=30)

    print("\n[4/5] Scores Joconde...")
    periodes_patterns = build_keyword_patterns(PERIODES_KEYWORDS)
    domaines_patterns = build_keyword_patterns(DOMAINES_KEYWORDS)
    scores_j = compute_joconde_scores(full_corpus, df_joconde, periodes_patterns, domaines_patterns)

    print("\n[5/5] Scores Basilic...")
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
            "joconde_score":      j.get("joconde_score",      0),
            "joconde_couverture": j.get("joconde_couverture", 0),
            "joconde_periodes":   json.dumps(j.get("joconde_periodes",  {}), ensure_ascii=False),
            "joconde_domaines":   json.dumps(j.get("joconde_domaines",  {}), ensure_ascii=False),
            "basilic_score":      b.get("basilic_score",      0),
            "basilic_couverture": b.get("basilic_couverture", 0),
            "basilic_regions":    json.dumps(b.get("basilic_regions",   {}), ensure_ascii=False),
            "basilic_types":      json.dumps(b.get("basilic_types",     {}), ensure_ascii=False),
            "tfidf_discriminants": json.dumps(top_terms.get(model, {}).get("discriminants", []), ensure_ascii=False),
            "tfidf_communs":       json.dumps(top_terms.get(model, {}).get("communs",        []), ensure_ascii=False),
        })

    df_out = pl.DataFrame(rows)
    df_out.write_parquet(OUTPUT)
    print(f"\nTerminé -> {OUTPUT}")
    print(df_out.select(["model", "joconde_score", "joconde_couverture", "basilic_score", "basilic_couverture"]))


if __name__ == "__main__":
    main()