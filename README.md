# univariate_scan2345

# UNIVARIATE + SPECIAL-DECILE TOOLKIT
# Paste in a notebook cell. Uses pandas, numpy, scipy, sklearn.
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, ks_2samp, pointbiserialr
from sklearn.metrics import roc_auc_score

# -------------------------
# Special quantile function
# -------------------------
import numpy as np
import pandas as pd

def special_quantiles(series, x_percent=5, min_bin_pct=0.01):
    """
    Robust special-quantiles binning:
      - Separates null (-2/"null"), negative (-1/"neg"), zero (0/"zero")
      - Bins positives into ~x_percent bins using quantile-derived edges (np.quantile -> pd.cut)
      - Returns: (numeric_labels, string_labels, bin_edges_list)
    Works even with ties by collapsing duplicate edges; if positives have too-few uniques,
    will label by unique values and still return edges representing those unique values.
    """
    s = pd.Series(series)  # copy, preserve index
    idx = s.index
    numeric_labels = pd.Series(index=idx, dtype="float64")
    string_labels = pd.Series(index=idx, dtype="object")

    # Mark special categories
    is_null = s.isna()
    numeric_labels[is_null] = -2
    string_labels[is_null] = "null"

    is_neg = (~is_null) & (s < 0)
    numeric_labels[is_neg] = -1
    string_labels[is_neg] = "neg"

    is_zero = (~is_null) & (~is_neg) & (s == 0)
    numeric_labels[is_zero] = 0
    string_labels[is_zero] = "zero"

    # Positives
    is_pos = (~is_null) & (~is_neg) & (~is_zero) & (s > 0)
    pos = s[is_pos].astype(float)
    n = len(s)
    n_pos = len(pos)

    if n_pos == 0:
        return numeric_labels, string_labels, []

    # target number of bins based on remaining pct
    remaining_pct = n_pos / n if n>0 else 0
    if x_percent <= 0 or x_percent >= 100:
        n_bins_target = 1
    else:
        n_bins_target = int(round(remaining_pct * 100 / x_percent))
        n_bins_target = max(1, n_bins_target)

    unique_pos_vals = np.unique(pos)
    unique_pos_count = len(unique_pos_vals)

    # If too few unique positive values, label by unique values but still produce edges
    if unique_pos_count <= max(1, n_bins_target):
        # label by rank of unique value
        rank_map = {v: i+1 for i, v in enumerate(sorted(unique_pos_vals))}
        numeric_labels[is_pos] = pos.map(rank_map)
        string_labels[is_pos] = pos.map(lambda v: str(v))
        # build edges as the midpoints between unique values (or exact values)
        if unique_pos_count == 1:
            edges = [float(unique_pos_vals[0]), float(unique_pos_vals[0])]
        else:
            # create edges that capture exact points (use quantiles of unique values)
            edges = np.concatenate(([unique_pos_vals[0]], (unique_pos_vals[:-1] + unique_pos_vals[1:]) / 2.0, [unique_pos_vals[-1]]))
            edges = edges.tolist()
        return numeric_labels, string_labels, edges

    # Compute quantile-based edges robustly (use np.quantile)
    try:
        q = np.linspace(0.0, 1.0, n_bins_target + 1)
        raw_edges = np.quantile(pos, q)
        # collapse duplicates (ties) to unique sorted edges
        edges = np.unique(raw_edges)
        if len(edges) <= 1:
            # everything identical after quantiles - fallback
            numeric_labels[is_pos] = 1
            string_labels[is_pos] = f"{float(pos.min()):.6g}-{float(pos.max()):.6g}"
            return numeric_labels, string_labels, []
        # ensure monotonic increasing edges; if edges length > 1 use pd.cut
        # pad a tiny epsilon to rightmost edge to include max value in last bin (if needed)
        # but pd.cut is inclusive on right by default for bins, so this is okay.
        # Use labels 1..(len(edges)-1)
        try:
            cats = pd.cut(pos, bins=edges, include_lowest=True, duplicates='drop', labels=False)
        except TypeError:
            # older pandas may not accept duplicates arg here — fallback to pd.cut without duplicates arg
            cats = pd.cut(pos, bins=edges, include_lowest=True, labels=False)
        # pd.cut labels start at 0; convert to 1-based
        cats1 = cats + 1
        numeric_labels.loc[is_pos] = cats1
        # create human readable ranges for string labels from edges
        bin_ranges = []
        for i in range(len(edges)-1):
            left = float(edges[i])
            right = float(edges[i+1])
            bin_ranges.append(f"{round(left,6)}-{round(right,6)}")
        string_labels.loc[is_pos] = cats1.map(lambda q: bin_ranges[int(q)-1])
        return numeric_labels, string_labels, edges.tolist()
    except Exception:
        # final fallback: single bin
        numeric_labels.loc[is_pos] = 1
        string_labels.loc[is_pos] = f"{float(pos.min()):.6g}-{float(pos.max()):.6g}"
        return numeric_labels, string_labels, []


# -------------------------
# IV / WOE function (optional)
# -------------------------
def calc_iv_woe_from_crosstab(tab):
    """
    Input: pd.DataFrame contingency table: rows=categories, cols=[0,1] counts
    Output: (iv, woe_dict)
    """
    # ensure columns 0 and 1 exist
    if 1 not in tab.columns or 0 not in tab.columns:
        return np.nan, {}
    # add small constant to avoid division by zero
    eps = 1e-9
    event = tab[1].astype(float)
    non_event = tab[0].astype(float)
    distr_event = event / (event.sum() + eps)
    distr_non_event = non_event / (non_event.sum() + eps)
    woe = np.log((distr_event + eps) / (distr_non_event + eps))
    iv = ((distr_event - distr_non_event) * woe).sum()
    return float(iv), woe.to_dict()

# -------------------------
# Categorical metrics
# -------------------------
def categorical_metrics(series, target_series, is_rare=False, top_k_pct=0.1):
    """
    Returns dict of stats for a categorical-like series.
    series: pd.Series (can be numeric-coded or string)
    target_series: pd.Series of 0/1
    """
    s = series.fillna("__MISSING__").astype("object")
    tab = pd.crosstab(s, target_series)
    # convert to counts
    counts = tab.sum(axis=1).sort_values(ascending=False)
    n_total = len(s)
    overall_rate = float(target_series.mean()) if n_total>0 else np.nan

    # select top category
    top_cat = counts.index[0] if len(counts)>0 else None
    top_cat_count = int(counts.iloc[0]) if len(counts)>0 else 0
    top_cat_rate = float(tab.loc[top_cat,1]) / top_cat_count if (len(counts)>0 and 1 in tab.columns and top_cat_count>0) else np.nan

    # build result
    result = {
        "n_missing": int((series == "__MISSING__").sum()),
        "n_unique": int(s.nunique()),
        "overall_rate": overall_rate,
        "top_category": str(top_cat),
        "top_category_count": top_cat_count,
        "top_category_rate": top_cat_rate
    }

    # association test: use Fisher for small cells, else chi2
    try:
        if tab.size == 0:
            result["chi2_pvalue"] = np.nan
        elif tab.values.min() < 5 and tab.shape == (2,2):
            _, p = fisher_exact(tab.values)
            result["chi2_pvalue"] = float(p)
        else:
            _, p, _, _ = chi2_contingency(tab)
            result["chi2_pvalue"] = float(p)
    except Exception:
        result["chi2_pvalue"] = np.nan

    # IV & WOE
    try:
        iv, woe_map = calc_iv_woe_from_crosstab(tab)
        result["iv"] = iv
    except Exception:
        result["iv"] = np.nan

    # Lift & capture for top bin (top category)
    try:
        if overall_rate > 0 and top_cat_count>0:
            result["lift_top_category"] = result["top_category_rate"] / overall_rate
            result["capture_top_category_pct_of_events"] = (tab.loc[top_cat,1] / (tab[1].sum() + 1e-12)) if (1 in tab.columns) else np.nan
        else:
            result["lift_top_category"] = np.nan
            result["capture_top_category_pct_of_events"] = np.nan
    except Exception:
        result["lift_top_category"] = np.nan
        result["capture_top_category_pct_of_events"] = np.nan

    # top-k cumulative capture (if ordered by score, but for pure categorical we sort by category event rate)
    try:
        # compute event rate per category, sort descending, accumulate counts until top_k_pct
        rates = (tab[1] / (tab.sum(axis=1) + 1e-12)).sort_values(ascending=False)
        sizes = tab.sum(axis=1)[rates.index]
        cutoff = int(max(1, np.round(n_total * top_k_pct)))
        cum = sizes.cumsum()
        selected = rates.index[cum <= cutoff]
        if len(selected)==0:
            selected = [rates.index[0]]
        captured = tab.loc[selected,1].sum() if 1 in tab.columns else 0
        total_events = tab[1].sum() if 1 in tab.columns else 0
        result["capture_top_{}pct_events".format(int(top_k_pct*100))] = float(captured) / (float(total_events)+1e-12) if total_events>0 else np.nan
    except Exception:
        result["capture_top_{}pct_events".format(int(top_k_pct*100))] = np.nan

    return result

# -------------------------
# Numeric metrics
# -------------------------
def numeric_metrics(series, target_series, is_rare=False):
    s = series
    n = len(s)
    n_missing = int(s.isna().sum())
    n_non_missing = n - n_missing
    res = {
        "n_missing": n_missing,
        "n_unique": int(s.nunique(dropna=True)),
        "mean": float(s.mean()) if n_non_missing>0 else np.nan,
        "median": float(s.median()) if n_non_missing>0 else np.nan,
        "std": float(s.std()) if n_non_missing>0 else np.nan,
        "min": float(s.min()) if n_non_missing>0 else np.nan,
        "max": float(s.max()) if n_non_missing>0 else np.nan
    }
    # correlations & stats
    try:
        # Pearson correlation with binary target
        res["pearson_corr_with_target"] = float(s.corr(target_series))
    except Exception:
        res["pearson_corr_with_target"] = np.nan
    try:
        # point biserial (same idea)
        if s.dropna().nunique() > 1:
            res["pointbiserial_corr"] = float(pointbiserialr(target_series.loc[s.dropna().index], s.dropna())[0])
        else:
            res["pointbiserial_corr"] = np.nan
    except Exception:
        res["pointbiserial_corr"] = np.nan
    # KS between values for events vs non-events
    try:
        ev = s[target_series==1].dropna()
        ne = s[target_series==0].dropna()
        if len(ev)>0 and len(ne)>0:
            res["ks_statistic"] = float(ks_2samp(ev, ne).statistic)
        else:
            res["ks_statistic"] = np.nan
    except Exception:
        res["ks_statistic"] = np.nan
    # AUC using raw numeric as score (works if numeric meaningful)
    try:
        mask = ~s.isna() & ~target_series.isna()
        if mask.sum()>0 and target_series[mask].nunique()>1:
            res["auc_using_numeric_as_score"] = float(roc_auc_score(target_series[mask], s[mask]))
        else:
            res["auc_using_numeric_as_score"] = np.nan
    except Exception:
        res["auc_using_numeric_as_score"] = np.nan

    # top-decile lift (helpful for rare targets). We'll compute top 10% by numeric value
    try:
        df_tmp = pd.DataFrame({"score": s, "target": target_series})
        df_tmp = df_tmp.dropna(subset=["score"])
        cutoff = max(1, int(np.ceil(len(df_tmp) * 0.10)))
        topk = df_tmp.sort_values("score", ascending=False).head(cutoff)
        top_rate = float(topk["target"].mean()) if cutoff>0 else np.nan
        overall_rate = float(target_series.mean())
        res["top10pct_rate"] = top_rate
        res["top10pct_lift"] = (top_rate / overall_rate) if overall_rate>0 else np.nan
    except Exception:
        res["top10pct_rate"] = np.nan
        res["top10pct_lift"] = np.nan

    return res

# -------------------------
# Main variable summary loop
# -------------------------
def variable_summary(df, target_col, x_percent=5, rare_threshold=0.02, top_k_for_capture=0.1):
    """
    df: pandas DataFrame
    target_col: name of binary 0/1 target column
    x_percent: desired percent per quantile bin for the positive part in special_quantiles
    rare_threshold: target prevalence under which we compute/flag rare-specific metrics
    """
    y = df[target_col]
    overall_rate = float(y.mean())
    is_rare = overall_rate < rare_threshold
    results = []

    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]

        # Quick skip: if column entirely unique ID-like except small cardinality detection
        try:
            # detect numeric vs categorical
            if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > max(500, len(df)*0.5):
                # huge unique numeric - likely ID - record minimal info and skip heavy stats
                row = {
                    "variable": col, "treatment": "id_like_skip", "var_type": "numeric_id_like",
                    "n_missing": int(series.isna().sum()),
                    "n_unique": int(series.nunique(dropna=True)),
                    "overall_rate": overall_rate
                }
                results.append(row)
                continue
        except Exception:
            pass

        # Handle numeric
        if pd.api.types.is_numeric_dtype(series):
            # numeric raw
            nm = numeric_metrics(series, y, is_rare=is_rare)
            row = {"variable": col, "treatment": "numeric", "var_type": "numeric"}
            row.update(nm)
            row.update({"n_missing": int(series.isna().sum()), "n_unique": int(series.nunique(dropna=True)), "overall_rate": overall_rate})
            results.append(row)

            # numeric deciles using special_quantiles: produce two label series
            num_labels, str_labels, bins = special_quantiles(series, x_percent=x_percent)
            # treat numeric_labels as categorical (but numeric-coded). compute categorical metrics
            cat_metrics = categorical_metrics(num_labels, y, is_rare=is_rare, top_k_pct=top_k_for_capture)
            cat_row = {"variable": col, "treatment": "numeric_deciles", "var_type": "numeric_as_categorical"}
            cat_row.update(cat_metrics)
            cat_row.update({"n_missing": int(series.isna().sum()), "n_unique": int(series.nunique(dropna=True)), "overall_rate": overall_rate})
            # additional decile-specific stats: top bin index and range info
            try:
                if len(bins) > 0:
                    cat_row["decile_bins_count"] = len(bins)-1
                    cat_row["decile_bin_edges"] = bins.tolist()
                else:
                    cat_row["decile_bins_count"] = np.nan
                    cat_row["decile_bin_edges"] = np.nan
            except Exception:
                cat_row["decile_bins_count"] = np.nan
                cat_row["decile_bin_edges"] = np.nan
            results.append(cat_row)

        else:
            # treat as categorical (string-like)
            cat_metrics = categorical_metrics(series, y, is_rare=is_rare, top_k_pct=top_k_for_capture)
            row = {"variable": col, "treatment": "categorical", "var_type": "categorical"}
            row.update(cat_metrics)
            row.update({"n_missing": int(series.isna().sum()), "n_unique": int(series.nunique(dropna=True)), "overall_rate": overall_rate})
            results.append(row)

    summary_df = pd.DataFrame(results)
    # reorder columns for readability
    cols_order = ["variable","treatment","var_type","n_missing","n_unique","overall_rate",
                  "mean","median","std","min","max",
                  "pearson_corr_with_target","pointbiserial_corr","ks_statistic","auc_using_numeric_as_score",
                  "top10pct_rate","top10pct_lift",
                  "chi2_pvalue","iv","top_category","top_category_count","top_category_rate",
                  "lift_top_category","capture_top_category_pct_of_events","capture_top_10pct_events",
                  "decile_bins_count","decile_bin_edges"]
    existing = [c for c in cols_order if c in summary_df.columns]
    # put remaining cols after
    remaining = [c for c in summary_df.columns if c not in existing]
    summary_df = summary_df[existing + remaining]
    return summary_df

# -------------------------  
# Example usage:
# -------------------------
# 1) create df (use your generator or load your dataset)
# 2) run:
# summary = variable_summary(df, "target_rare", x_percent=5, rare_threshold=0.02)
# summary.head(40)
# summary.to_excel("univariate_with_special_deciles.xlsx", index=False)
