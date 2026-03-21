import polars as pl
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'cleaned')
os.makedirs(output_dir, exist_ok=True)

params_regex = (
    r"(?i)("
    r"\bv?\d+(\.\d+)*(-[a-z0-9]+)?\b|"
    r"\b\d+[x]\d+[bB]\b|\b\d+[bB]\b|"
    r"\b[a-z]\d+[a-z]?\b|"
    r"\b(o\d|r\d|p\d|v\d)\b|"
    r"\d+o|"
    r"\b\d{4}-\d{2}(-\d{2})?\b|"
    r"\b\d{4,6}\b|"
    r"\b(instruct|chat|it|dpo|sonnet|haiku|pro|flash|mini|nano|preview|think|thinking|coder|distill|latest|plus|max|large|medium|small|tiny|base|xl|ext|expanse|scout|maverick|nemotron|oss|all-minilm|saba|turbo|light|a|b)\b|"
    r"|[-_]\d+([-_]\d+)*\b"
    r")"
)

corrections = {
    r"^ge$": "gemini", r"^stral$": "mistral", r"^qwq$": "qwen",
    r"^o\d$": "gpt", r"^gpt.*": "gpt", r"^claude.*": "claude",
    r"^gemini.*": "gemini", r".*llama.*": "llama", r"^mistral.*": "mistral",
    r"^mixtral.*": "mixtral", r"^qwen.*": "qwen", r"^grok.*": "grok",
    r"^phi.*": "phi", r"^gemma.*": "gemma", r"^glm.*": "glm",
    r"^yi.*": "yi", r"^deepseek.*": "deepseek", r"^hermes.*": "hermes",
    r"^lfm.*": "lfm", r"^command.*": "command", r"^jamba.*": "jamba"
}

def apply_model_cleaning(df):
    if "conv_turns" in df.columns:
        df = df.filter(pl.col("conv_turns") > 0)
    df = df.with_columns([
        pl.col("model_a_name").str.extract_all(params_regex).list.eval(pl.element().filter(pl.element() != "")).list.slice(0, 2).list.join("-").alias("version_a"),
        pl.col("model_b_name").str.extract_all(params_regex).list.eval(pl.element().filter(pl.element() != "")).list.slice(0, 2).list.join("-").alias("version_b"),
        pl.col("model_a_name").str.replace_all(params_regex, "").str.to_lowercase().str.replace_all(r"[-._]+", "-").str.strip_chars("-.").alias("base_model_a"),
        pl.col("model_b_name").str.replace_all(params_regex, "").str.to_lowercase().str.replace_all(r"[-._]+", "-").str.strip_chars("-.").alias("base_model_b"),
    ])
    for wrong, right in corrections.items():
        df = df.with_columns([
            pl.col("base_model_a").str.replace(wrong, right),
            pl.col("base_model_b").str.replace(wrong, right),
        ])
    df = df.with_columns([
        pl.col("base_model_a").str.strip_chars("-"),
        pl.col("base_model_b").str.strip_chars("-"),
        pl.col("version_a").str.strip_chars("-"),
        pl.col("version_b").str.strip_chars("-"),
    ])
    return df.filter(
        (pl.col("base_model_a") != "") & (pl.col("base_model_b") != "") &
        pl.col("base_model_a").is_not_null() & pl.col("base_model_b").is_not_null()
    )

def apply_winner_cleaning(df):
    df = df.with_columns([
        pl.col("chosen_model_name").str.extract_all(params_regex).list.eval(pl.element().filter(pl.element() != "")).list.slice(0, 2).list.join("-").alias("chosen_version"),
        pl.col("chosen_model_name").str.replace_all(params_regex, "").str.to_lowercase().str.replace_all(r"[-._]+", "-").str.strip_chars("-.").alias("chosen_base_model"),
    ])
    for wrong, right in corrections.items():
        df = df.with_columns([pl.col("chosen_base_model").str.replace(wrong, right)])
    return df

path_react_in = os.path.join(script_dir, 'reactions.parquet')
if os.path.exists(path_react_in):
    df_react = pl.read_parquet(path_react_in)
    df_react = apply_model_cleaning(df_react)
    df_react = df_react.with_columns([
        pl.when(pl.col("model_pos") == "a").then(pl.col("base_model_a")).when(pl.col("model_pos") == "b").then(pl.col("base_model_b")).otherwise(None).alias("model"),
        pl.when(pl.col("model_pos") == "a").then(pl.col("version_a")).when(pl.col("model_pos") == "b").then(pl.col("version_b")).otherwise(None).alias("version"),
    ])
    drop_react = ["conversation_a", "conversation_b", "system_prompt", "question_id",
                  "comment", "refers_to_conv_id", "response_content", "question_content",
                  "timestamp", "msg_index", "opening_msg", "conv_turns", "session_hash", "msg_rank"]
    df_react = df_react.drop([c for c in drop_react if c in df_react.columns])
    df_react = df_react.select([col for col in df_react.columns if 'id' not in col or col == 'id' or col in ['model', 'version']])
    df_react.write_parquet(os.path.join(output_dir, 'react.parquet'))

path_conv_in = os.path.join(script_dir, 'conversations.parquet')
if os.path.exists(path_conv_in):
    df_conv = pl.read_parquet(path_conv_in)
    bad_matches = ["none", "w1yz2kxp7d2u4nxrw3x444c2otumic64ysqfi37", "wnnzokjptdouqnjri3j4q4y2atgm4cs4kscf53u"]
    df_conv = df_conv.filter(
        ~pl.col("model_a_name").str.starts_with("javascript") & ~pl.col("model_a_name").str.starts_with("'") &
        ~pl.col("model_a_name").str.starts_with("<") & ~pl.col("model_a_name").is_in(bad_matches) &
        ~pl.col("model_b_name").str.starts_with("javascript") & ~pl.col("model_b_name").str.starts_with("'") &
        ~pl.col("model_b_name").str.starts_with("<") & ~pl.col("model_b_name").is_in(bad_matches)
    )
    df_conv = df_conv.with_columns([
        pl.col("visitor_id").replace("", None).fill_null(pl.col("session_hash")).alias("visitor_id")
    ])
    df_conv = apply_model_cleaning(df_conv)
    drop_conv = ["conversation_a", "conversation_b", "conversation_pair_id", "conv_a_id", "conv_b_id", "session_hash", "model_pair_name"]
    df_conv = df_conv.drop([c for c in drop_conv if c in df_conv.columns])
    df_conv.write_parquet(os.path.join(output_dir, 'conv.parquet'))

path_votes_in = os.path.join(script_dir, 'votes.parquet')
if os.path.exists(path_votes_in):
    df_votes = pl.read_parquet(path_votes_in)
    df_votes = apply_model_cleaning(df_votes)
    df_votes = apply_winner_cleaning(df_votes)
    drop_votes = ["conversation_a", "conversation_b"]
    df_votes = df_votes.drop([c for c in drop_votes if c in df_votes.columns])
    df_votes = df_votes.select([col for col in df_votes.columns if 'id' not in col or col == 'id' or 'chosen' in col or 'base_model' in col or 'version' in col])
    df_votes.write_parquet(os.path.join(output_dir, 'vot.parquet'))

    df_arena = pl.read_parquet(path_votes_in)
    df_arena = df_arena.filter(
        ~pl.col("model_a_name").str.starts_with("javascript") & ~pl.col("model_a_name").str.starts_with("'") &
        ~pl.col("model_a_name").str.starts_with("<") &
        ~pl.col("model_b_name").str.starts_with("javascript") & ~pl.col("model_b_name").str.starts_with("'") &
        ~pl.col("model_b_name").str.starts_with("<")
    )
    df_arena = apply_model_cleaning(df_arena)
    df_arena = apply_winner_cleaning(df_arena)
    arena_cols = [
        "base_model_a", "base_model_b", "version_a", "version_b",
        "chosen_base_model", "chosen_version", "both_equal",
        "conversation_a", "conversation_b", "opening_msg",
        "conv_turns", "selected_category", "timestamp",
        "conv_comments_a", "conv_comments_b",
        "conv_useful_a", "conv_useful_b",
        "conv_creative_a", "conv_creative_b",
        "conv_clear_formatting_a", "conv_clear_formatting_b",
        "conv_incorrect_a", "conv_incorrect_b",
        "conv_superficial_a", "conv_superficial_b",
        "conv_instructions_not_followed_a", "conv_instructions_not_followed_b",
    ]
    df_arena = df_arena.select([c for c in arena_cols if c in df_arena.columns])
    df_arena = df_arena.with_row_index("arena_id")
    df_arena.write_parquet(os.path.join(output_dir, 'arena.parquet'))

bias_dir = os.path.join(script_dir, 'bias_analysis')
os.makedirs(bias_dir, exist_ok=True)

if os.path.exists(path_conv_in):
    df_bias = pl.read_parquet(path_conv_in)
    df_bias = df_bias.filter(
        ~pl.col("model_a_name").str.starts_with("javascript") & ~pl.col("model_a_name").str.starts_with("'") &
        ~pl.col("model_a_name").str.starts_with("<") &
        ~pl.col("model_b_name").str.starts_with("javascript") & ~pl.col("model_b_name").str.starts_with("'") &
        ~pl.col("model_b_name").str.starts_with("<")
    )
    df_bias = apply_model_cleaning(df_bias)
    bias_cols = [
        "base_model_a", "base_model_b",
        "conversation_a", "conversation_b",
        "opening_msg", "short_summary", "keywords", "categories", "languages",
        "visitor_id", "timestamp",
    ]
    df_bias = df_bias.select([c for c in bias_cols if c in df_bias.columns])
    df_bias = df_bias.filter(pl.col("languages").list.contains("fr"))
    df_bias.write_parquet(os.path.join(bias_dir, 'bias_conv.parquet'))