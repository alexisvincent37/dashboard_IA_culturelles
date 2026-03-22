import streamlit as st
import polars as pl
import json
import os

st.set_page_config(page_title="IA Culturelles - Arena", layout="wide")

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
.stSelectbox label, .stMultiSelect label { font-family: 'DM Mono', monospace !important; font-size: 0.68rem !important; color: #475569 !important; text-transform: uppercase; letter-spacing: 0.08em; }

.card { background: #0F1117; border: 1px solid #2A3050; border-radius: 8px; padding: 14px 16px; margin-bottom: 10px; }
.card-q { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #CBD5E1; margin-bottom: 8px; line-height: 1.5; }
.card-meta { font-family: 'DM Mono', monospace; font-size: 0.62rem; color: #475569; }
.tag-win { background: rgba(52,211,153,0.12); color: #34D399; border: 1px solid rgba(52,211,153,0.25); padding: 1px 8px; border-radius: 3px; font-size: 0.62em; display: inline-block; }
.tag-b   { background: rgba(96,165,250,0.12);  color: #60A5FA; border: 1px solid rgba(96,165,250,0.25);  padding: 1px 8px; border-radius: 3px; font-size: 0.62em; display: inline-block; }
.tag-eq  { background: rgba(251,191,36,0.12);  color: #FBBF24; border: 1px solid rgba(251,191,36,0.25);  padding: 1px 8px; border-radius: 3px; font-size: 0.62em; display: inline-block; }

.winner-a { background: rgba(52,211,153,0.12); color: #34D399; border: 1px solid rgba(52,211,153,0.25); padding: 4px 12px; border-radius: 4px; font-size: 0.72em; font-family: 'DM Mono', monospace; display: inline-block; }
.winner-b { background: rgba(96,165,250,0.12); color: #60A5FA; border: 1px solid rgba(96,165,250,0.25); padding: 4px 12px; border-radius: 4px; font-size: 0.72em; font-family: 'DM Mono', monospace; display: inline-block; }
.winner-eq { background: rgba(251,191,36,0.12); color: #FBBF24; border: 1px solid rgba(251,191,36,0.25); padding: 4px 12px; border-radius: 4px; font-size: 0.72em; font-family: 'DM Mono', monospace; display: inline-block; }

.hdr-a { background: rgba(52,211,153,0.06); border: 1px solid rgba(52,211,153,0.15); border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; }
.hdr-b { background: rgba(96,165,250,0.06); border: 1px solid rgba(96,165,250,0.15); border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; }
.nm-a { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: #34D399; }
.nm-b { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: #60A5FA; }
.ver { font-family: 'DM Mono', monospace; font-size: 0.60rem; color: #475569; margin-top: 2px; }

.msg-u-lbl { font-size: 0.58rem; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; text-align: right; margin-bottom: 3px; }
.msg-u { background: #1E2130; border-radius: 12px 12px 2px 12px; padding: 10px 14px; font-size: 0.78rem; color: #94A3B8; line-height: 1.6; margin-bottom: 8px; }
.msg-ai-lbl { font-size: 0.58rem; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 3px; }

.fb-g { background: rgba(52,211,153,0.08); color: #34D399; border: 1px solid rgba(52,211,153,0.2); padding: 2px 8px; border-radius: 4px; font-size: 0.65em; display: inline-block; margin: 2px; }
.fb-r { background: rgba(248,113,113,0.08); color: #F87171; border: 1px solid rgba(248,113,113,0.2); padding: 2px 8px; border-radius: 4px; font-size: 0.65em; display: inline-block; margin: 2px; }
.cmt { background: #0A0C12; border: 1px solid #2A3050; border-radius: 6px; padding: 10px 14px; font-size: 0.72rem; color: #94A3B8; font-style: italic; margin-top: 8px; line-height: 1.6; }
.slabel { font-family: 'DM Mono', monospace; font-size: 0.60rem; color: #475569; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px; }
.conv-count { font-family: 'DM Mono', monospace; font-size: 0.62rem; color: #475569; margin-bottom: 12px; }
.prompt-box { background: #1E2130; border-radius: 8px; padding: 12px 16px; font-size: 0.82rem; color: #94A3B8; line-height: 1.6; margin-bottom: 4px; }
.ma-col { color: #34D399; font-family: 'DM Mono', monospace; font-size: 0.62rem; }
.mb-col { color: #60A5FA; font-family: 'DM Mono', monospace; font-size: 0.62rem; }
</style>
""", unsafe_allow_html=True)

ARENA_PATH = "data/cleaned/arena.parquet"

@st.cache_data
def load_arena():
    if not os.path.exists(ARENA_PATH):
        return None
    return pl.read_parquet(ARENA_PATH)

def parse_conv(raw) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [{"role": m.get("role", ""), "content": m.get("content", "")} for m in raw if isinstance(m, dict)]
    try:
        msgs = json.loads(str(raw).replace("'", '"'))
        if isinstance(msgs, list):
            return [{"role": m.get("role", ""), "content": m.get("content", "")} for m in msgs if isinstance(m, dict)]
    except Exception:
        pass
    return []

def get_feedback(row, side):
    tags = []
    if row.get(f"conv_useful_{side}"): tags.append(("g", "Utile"))
    if row.get(f"conv_creative_{side}"): tags.append(("g", "Créatif"))
    if row.get(f"conv_clear_formatting_{side}"): tags.append(("g", "Clair"))
    if row.get(f"conv_incorrect_{side}"): tags.append(("r", "Incorrect"))
    if row.get(f"conv_superficial_{side}"): tags.append(("r", "Superficiel"))
    if row.get(f"conv_instructions_not_followed_{side}"): tags.append(("r", "Instructions non suivies"))
    comment = str(row.get(f"conv_comments_{side}") or "").strip()
    return tags, comment

df = load_arena()
if df is None:
    st.error("arena.parquet introuvable — relance data_cleaning.py.")
    st.stop()

all_models = sorted(set(
    df["base_model_a"].drop_nulls().unique().to_list() +
    df["base_model_b"].drop_nulls().unique().to_list()
))
all_cats = sorted([c for c in df["selected_category"].drop_nulls().unique().to_list() if c])

def get_versions(model):
    if model == "Tous":
        return ["Tous"]
    va = df.filter(pl.col("base_model_a") == model)["version_a"].drop_nulls().unique().to_list()
    vb = df.filter(pl.col("base_model_b") == model)["version_b"].drop_nulls().unique().to_list()
    versions = sorted(set(v for v in va + vb if v and str(v).strip()))
    return ["Tous"] + versions

if "sel_conv" not in st.session_state:
    st.session_state.sel_conv = None

with st.sidebar:
    st.markdown('<div class="slabel">Filtres</div>', unsafe_allow_html=True)

    sel_m1  = st.selectbox("Modèle 1", ["Tous"] + all_models, key="m1")
    ver1_opts = get_versions(sel_m1)
    sel_v1  = st.selectbox("Version 1", ver1_opts, key="v1")

    sel_m2  = st.selectbox("Modèle 2", ["Tous"] + all_models, key="m2")
    ver2_opts = get_versions(sel_m2)
    sel_v2  = st.selectbox("Version 2", ver2_opts, key="v2")

    sel_cats   = st.multiselect("Catégorie", all_cats, placeholder="Toutes")
    sel_winner = st.selectbox("Résultat", ["Tous", "M1 gagne", "M2 gagne", "Égalité"])
    sel_turns  = st.selectbox("Longueur", ["Toutes", "1 tour", "2–3 tours", "4+ tours"])

filt = df.clone()

if sel_m1 != "Tous" and sel_m2 != "Tous":
    filt = filt.filter(
        ((pl.col("base_model_a") == sel_m1) & (pl.col("base_model_b") == sel_m2)) |
        ((pl.col("base_model_a") == sel_m2) & (pl.col("base_model_b") == sel_m1))
    )
elif sel_m1 != "Tous":
    filt = filt.filter(pl.col("base_model_a").is_in([sel_m1]) | pl.col("base_model_b").is_in([sel_m1]))
elif sel_m2 != "Tous":
    filt = filt.filter(pl.col("base_model_a").is_in([sel_m2]) | pl.col("base_model_b").is_in([sel_m2]))

if sel_m1 != "Tous" and sel_v1 != "Tous":
    filt = filt.filter(
        ((pl.col("base_model_a") == sel_m1) & (pl.col("version_a") == sel_v1)) |
        ((pl.col("base_model_b") == sel_m1) & (pl.col("version_b") == sel_v1))
    )
if sel_m2 != "Tous" and sel_v2 != "Tous":
    filt = filt.filter(
        ((pl.col("base_model_a") == sel_m2) & (pl.col("version_a") == sel_v2)) |
        ((pl.col("base_model_b") == sel_m2) & (pl.col("version_b") == sel_v2))
    )

if sel_cats:
    filt = filt.filter(pl.col("selected_category").is_in(sel_cats))

if sel_winner == "M1 gagne" and sel_m1 != "Tous":
    filt = filt.filter((pl.col("chosen_base_model") == sel_m1) & (pl.col("both_equal") == False))
elif sel_winner == "M2 gagne" and sel_m2 != "Tous":
    filt = filt.filter((pl.col("chosen_base_model") == sel_m2) & (pl.col("both_equal") == False))
elif sel_winner == "Égalité":
    filt = filt.filter(pl.col("both_equal") == True)

if sel_turns == "1 tour":
    filt = filt.filter(pl.col("conv_turns") == 1)
elif sel_turns == "2–3 tours":
    filt = filt.filter((pl.col("conv_turns") >= 2) & (pl.col("conv_turns") <= 3))
elif sel_turns == "4+ tours":
    filt = filt.filter(pl.col("conv_turns") >= 4)

if st.session_state.sel_conv is not None:
    cur_ids = filt.select("arena_id").to_series().to_list()
    if st.session_state.sel_conv not in cur_ids:
        st.session_state.sel_conv = None

if st.session_state.sel_conv is None:
    st.title("Arena")
    st.markdown(
        '<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#475569;margin-bottom:2px;">'
        'EXPLORATEUR DE CONVERSATIONS · Votes Compar:IA</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(f'<div class="conv-count">{filt.height} conversations · affichage des 200 premières</div>', unsafe_allow_html=True)

    rows = filt.head(200).to_dicts()
    for row in rows:
        aid  = row["arena_id"]
        q    = str(row.get("opening_msg") or "").strip() or "—"
        ma   = row.get("base_model_a", "?")
        mb   = row.get("base_model_b", "?")
        va   = row.get("version_a", "") or ""
        vb   = row.get("version_b", "") or ""
        t    = row.get("conv_turns", 1)
        cat  = row.get("selected_category") or ""
        eq   = row.get("both_equal", False)
        w    = row.get("chosen_base_model", "")
        ts   = str(row.get("timestamp") or "")[:10]

        if eq:
            verdict_tag = '<span class="tag-eq">⚖ Égalité</span>'
        elif w == ma:
            verdict_tag = f'<span class="tag-win">🏆 {ma}</span>'
        elif w == mb:
            verdict_tag = f'<span class="tag-b">🏆 {mb}</span>'
        else:
            verdict_tag = ""

        ma_str = ma + (f" · {va}" if va else "")
        mb_str = mb + (f" · {vb}" if vb else "")
        meta   = f'<span class="ma-col">{ma_str}</span> <span style="color:#2A3050">vs</span> <span class="mb-col">{mb_str}</span>'
        meta  += f' &nbsp;·&nbsp; {t} tour{"s" if t > 1 else ""}'
        if cat: meta += f' &nbsp;·&nbsp; {cat}'
        if ts:  meta += f' &nbsp;·&nbsp; {ts}'
        meta  += f' &nbsp;&nbsp;{verdict_tag}'

        st.markdown(f"""
        <div class="card">
            <div class="card-q">{q[:130]}{"…" if len(q) > 130 else ""}</div>
            <div class="card-meta">{meta}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Voir la conversation →", key=f"b_{aid}"):
            st.session_state.sel_conv = aid
            st.rerun()

else:
    match = filt.filter(pl.col("arena_id") == st.session_state.sel_conv)
    if match.is_empty():
        st.session_state.sel_conv = None
        st.rerun()

    row      = match.to_dicts()[0]
    ma       = row.get("base_model_a", "?")
    mb       = row.get("base_model_b", "?")
    va       = row.get("version_a", "") or ""
    vb       = row.get("version_b", "") or ""
    turns    = row.get("conv_turns", 1)
    cat      = row.get("selected_category") or "—"
    eq       = row.get("both_equal", False)
    w        = row.get("chosen_base_model", "")
    ts       = str(row.get("timestamp") or "")[:10]
    open_msg = str(row.get("opening_msg") or "").strip()
    conv_a   = parse_conv(row.get("conversation_a"))
    conv_b   = parse_conv(row.get("conversation_b"))
    tags_a, cmt_a = get_feedback(row, "a")
    tags_b, cmt_b = get_feedback(row, "b")

    if eq:
        verdict = '<span class="winner-eq">⚖️ Égalité</span>'
    elif w == ma:
        verdict = f'<span class="winner-a">🏆 {ma} gagne</span>'
    elif w == mb:
        verdict = f'<span class="winner-b">🏆 {mb} gagne</span>'
    else:
        verdict = '<span class="winner-eq">— Sans verdict</span>'

    st.title(f"{ma} vs {mb}")
    st.markdown(
        f'<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#475569;margin-bottom:8px;">'
        f'{cat} &nbsp;·&nbsp; {turns} tour{"s" if turns > 1 else ""} &nbsp;·&nbsp; {ts}</div>',
        unsafe_allow_html=True
    )
    st.markdown(verdict, unsafe_allow_html=True)

    if open_msg:
        st.markdown("---")
        st.markdown('<div class="slabel">Prompt</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prompt-box">{open_msg}</div>', unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    user_msgs = [m for m in conv_a if m["role"] == "user"]
    ia_msgs_a = [m for m in conv_a if m["role"] == "assistant"]
    ia_msgs_b = [m for m in conv_b if m["role"] == "assistant"]
    wi_a = " 🏆" if (not eq and w == ma) else (" ⚖️" if eq else "")
    wi_b = " 🏆" if (not eq and w == mb) else (" ⚖️" if eq else "")

    with col_a:
        st.markdown(f'<div class="hdr-a"><div class="nm-a">{ma}{wi_a}</div><div class="ver">{va}</div></div>', unsafe_allow_html=True)
        for i, (u, r) in enumerate(zip(user_msgs, ia_msgs_a)):
            if i > 0:
                st.markdown(f'<div class="msg-u-lbl">Utilisateur — tour {i+1}</div><div class="msg-u">{u["content"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="msg-ai-lbl">{ma}</div>', unsafe_allow_html=True)
            st.markdown(r["content"])
        if tags_a or cmt_a:
            st.markdown('<div class="slabel" style="margin-top:14px;">Retours utilisateur</div>', unsafe_allow_html=True)
            st.markdown("".join(f'<span class="fb-{t}">{txt}</span>' for t, txt in tags_a), unsafe_allow_html=True)
            if cmt_a:
                st.markdown(f'<div class="cmt">💬 {cmt_a}</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown(f'<div class="hdr-b"><div class="nm-b">{mb}{wi_b}</div><div class="ver">{vb}</div></div>', unsafe_allow_html=True)
        for i, (u, r) in enumerate(zip(user_msgs, ia_msgs_b)):
            if i > 0:
                st.markdown(f'<div class="msg-u-lbl">Utilisateur — tour {i+1}</div><div class="msg-u">{u["content"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="msg-ai-lbl">{mb}</div>', unsafe_allow_html=True)
            st.markdown(r["content"])
        if tags_b or cmt_b:
            st.markdown('<div class="slabel" style="margin-top:14px;">Retours utilisateur</div>', unsafe_allow_html=True)
            st.markdown("".join(f'<span class="fb-{t}">{txt}</span>' for t, txt in tags_b), unsafe_allow_html=True)
            if cmt_b:
                st.markdown(f'<div class="cmt">💬 {cmt_b}</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("← Retour aux conversations"):
        st.session_state.sel_conv = None
        st.rerun()