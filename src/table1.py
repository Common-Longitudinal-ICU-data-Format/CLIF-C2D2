"""
Table 1 generator: produce standard descriptive statistics for the C2D2 dataset.

Generates a CSV with one row per variable (grouped by clinical domain),
showing median [Q1, Q3] for continuous variables, n (%) for binary/categorical.
"""

from collections import OrderedDict
from pathlib import Path

import polars as pl

# Ordered mapping: domain name -> list of column prefixes/exact names.
# A column is assigned to the FIRST domain whose pattern matches.
DOMAIN_COLUMNS: OrderedDict[str, list[str]] = OrderedDict(
    [
        (
            "Anthropometrics & Demographics",
            [
                "C2D2_race",
                "C2D2_ethnicity",
                "C2D2_sex",
                "C2D2_language",
                "C2D2_age_",
                "C2D2_zipcode",
                "first_height_",
                "first_weight_",
            ],
        ),
        (
            "Advanced Directives",
            [
                "icu_24hr_code_status",
            ],
        ),
        (
            "Comorbidities",
            [
                "cci_",
            ],
        ),
        (
            "Diagnoses",
            [
                "icu_any_dx_",
            ],
        ),
        (
            "Diagnostic Tests",
            [
                "icu_24hr_creatinine_",
                "icu_24hr_bilirubin_",
                "icu_24hr_platelet_",
                "icu_24hr_hemoglobin_",
                "icu_24hr_wbc_",
                "icu_24hr_sodium_",
                "icu_24hr_potassium_",
                "icu_24hr_bicarbonate_",
                "icu_24hr_chloride_",
                "icu_24hr_bun_",
                "icu_24hr_glucose_",
                "icu_24hr_calcium_",
                "icu_24hr_lactate_",
                "icu_24hr_albumin_",
                "icu_24hr_ph_",
                "icu_24hr_pco2_",
                "icu_24hr_po2_",
                "icu_24hr_inr_",
                "icu_24hr_troponin_",
                "icu_24hr_procalcitonin_",
                "icu_24hr_crp_",
                "icu_24hr_ast_",
                "icu_24hr_alt_",
            ],
        ),
        (
            "Interventions",
            [
                "icu_any_imv",
                "icu_any_nippv",
                "icu_any_hfnc",
                "icu_any_tracheostomy",
                "icu_any_crrt",
                "icu_any_ecmo",
            ],
        ),
        (
            "Medications",
            [
                "icu_24hr_any_",
                "icu_24hr_max_",
            ],
        ),
        (
            "Objective Assessments",
            [
                "icu_24hr_temp_",
                "icu_24hr_heart_rate_",
                "icu_24hr_sbp_",
                "icu_24hr_dbp_",
                "icu_24hr_map_",
                "icu_24hr_spo2_",
                "icu_24hr_respiratory_rate_",
                "icu_24hr_gcs_",
                "icu_24hr_rass_",
                "icu_24hr_cam_",
                "icu_24hr_braden_",
                "icu_24hr_pf_ratio_",
            ],
        ),
        (
            "Outcomes",
            [
                "C2D2_admission_",
                "C2D2_discharge_",
                "C2D2_icu_",
                "hospital_",
                "icu_los_",
            ],
        ),
    ]
)

# Columns to always skip (identifiers and datetimes handled separately)
SKIP_COLUMNS = {"patient_id", "hospitalization_id", "icu_stay_id"}


def _match_domain(col_name: str) -> str | None:
    """Return the domain name for a column, or None if no match."""
    for domain, patterns in DOMAIN_COLUMNS.items():
        for pattern in patterns:
            if col_name == pattern or col_name.startswith(pattern):
                return domain
    return None


def _classify_column(
    series: pl.Series,
) -> str:
    """Classify a column as 'binary', 'continuous', or 'categorical'."""
    dtype = series.dtype

    if dtype in (pl.String, pl.Utf8, pl.Categorical):
        return "categorical"

    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32):
        unique_vals = series.drop_nulls().unique()
        if len(unique_vals) <= 2 and unique_vals.to_list() <= [0, 1]:
            # Check all values are in {0, 1}
            vals_set = set(unique_vals.to_list())
            if vals_set <= {0, 1}:
                return "binary"
        return "continuous"

    if dtype in (
        pl.Float32,
        pl.Float64,
        pl.Int64,
        pl.UInt64,
    ):
        return "continuous"

    if dtype in (pl.Date, pl.Datetime, pl.Time, pl.Duration):
        return "skip"

    return "continuous"


def _fmt_pct(n: int, total: int) -> str:
    """Format count and percentage: '1234 (56.7%)'."""
    if total == 0:
        return f"{n} (0.0%)"
    pct = 100.0 * n / total
    return f"{n} ({pct:.1f}%)"


def _summarize_continuous(series: pl.Series) -> tuple[str, int, str]:
    """Return (summary_str, n_nonmissing, pct_missing_str) for continuous."""
    non_null = series.drop_nulls()
    n_nonmissing = len(non_null)
    total = len(series)
    pct_missing = f"{100.0 * (total - n_nonmissing) / total:.1f}%" if total > 0 else "0.0%"

    if n_nonmissing == 0:
        return "NA", 0, pct_missing

    # Cast to float for quantile computation
    vals = non_null.cast(pl.Float64)
    median = vals.median()
    q1 = vals.quantile(0.25, interpolation="nearest")
    q3 = vals.quantile(0.75, interpolation="nearest")

    # Format: remove unnecessary trailing zeros for clean display
    def _fmt(v: float | None) -> str:
        if v is None:
            return "NA"
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.2f}"

    summary = f"{_fmt(median)} [{_fmt(q1)}, {_fmt(q3)}]"
    return summary, n_nonmissing, pct_missing


def _summarize_binary(series: pl.Series, total_n: int) -> tuple[str, int, str]:
    """Return (summary_str, n_nonmissing, pct_missing_str) for binary."""
    non_null = series.drop_nulls()
    n_nonmissing = len(non_null)
    total = len(series)
    pct_missing = f"{100.0 * (total - n_nonmissing) / total:.1f}%" if total > 0 else "0.0%"

    n_positive = int(non_null.sum()) if n_nonmissing > 0 else 0
    summary = _fmt_pct(n_positive, total_n)
    return summary, n_nonmissing, pct_missing


def _summarize_categorical(
    series: pl.Series, total_n: int
) -> tuple[int, str, list[tuple[str, str]]]:
    """Return (n_nonmissing, pct_missing, [(level, summary_str), ...]) for categorical."""
    non_null = series.drop_nulls()
    n_nonmissing = len(non_null)
    total = len(series)
    pct_missing = f"{100.0 * (total - n_nonmissing) / total:.1f}%" if total > 0 else "0.0%"

    # Count per level, sorted by frequency descending
    counts = non_null.value_counts().sort("count", descending=True)

    levels: list[tuple[str, str]] = []
    for row in counts.iter_rows():
        level_name = str(row[0])
        level_count = int(row[1])
        levels.append((level_name, _fmt_pct(level_count, total_n)))

    return n_nonmissing, pct_missing, levels


def generate_table1(
    df: pl.DataFrame, output_dir: Path, site_name: str
) -> None:
    """
    Generate a Table 1 summary statistics CSV from the final C2D2 DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The super-wide C2D2 output (1 row per ICU stay).
    output_dir : Path
        Directory to write the CSV into.
    site_name : str
        Site identifier for the output filename.
    """
    total_n = len(df)

    # Assign columns to domains
    domain_cols: dict[str, list[str]] = {d: [] for d in DOMAIN_COLUMNS}
    domain_cols["Other"] = []

    analysis_cols = [
        c
        for c in df.columns
        if c not in SKIP_COLUMNS
        and _classify_column(df[c]) != "skip"
    ]

    for col in analysis_cols:
        domain = _match_domain(col)
        if domain is not None:
            domain_cols[domain].append(col)
        else:
            domain_cols["Other"].append(col)

    # Build rows for the output table
    rows: list[dict[str, str]] = []

    # Header row with total N
    rows.append(
        {
            "domain": "",
            "variable": "N (ICU stays)",
            "summary": str(total_n),
            "n_nonmissing": "",
            "pct_missing": "",
        }
    )

    # Iterate domains in order
    all_domains = list(DOMAIN_COLUMNS.keys()) + (
        ["Other"] if domain_cols.get("Other") else []
    )

    for domain in all_domains:
        cols = domain_cols.get(domain, [])
        if not cols:
            continue

        # Domain header row
        rows.append(
            {
                "domain": domain,
                "variable": "",
                "summary": "",
                "n_nonmissing": "",
                "pct_missing": "",
            }
        )

        for col in sorted(cols):
            series = df[col]
            col_type = _classify_column(series)

            if col_type == "continuous":
                summary, n_nm, pct_m = _summarize_continuous(series)
                rows.append(
                    {
                        "domain": "",
                        "variable": f"  {col}",
                        "summary": summary,
                        "n_nonmissing": str(n_nm),
                        "pct_missing": pct_m,
                    }
                )

            elif col_type == "binary":
                summary, n_nm, pct_m = _summarize_binary(series, total_n)
                rows.append(
                    {
                        "domain": "",
                        "variable": f"  {col}",
                        "summary": summary,
                        "n_nonmissing": str(n_nm),
                        "pct_missing": pct_m,
                    }
                )

            elif col_type == "categorical":
                n_nm, pct_m, levels = _summarize_categorical(series, total_n)
                # Variable row (no summary — levels listed below)
                rows.append(
                    {
                        "domain": "",
                        "variable": f"  {col}",
                        "summary": "",
                        "n_nonmissing": str(n_nm),
                        "pct_missing": pct_m,
                    }
                )
                # Level sub-rows
                for level_name, level_summary in levels:
                    rows.append(
                        {
                            "domain": "",
                            "variable": f"    {level_name}",
                            "summary": level_summary,
                            "n_nonmissing": "",
                            "pct_missing": "",
                        }
                    )

    # Write output
    out_df = pl.DataFrame(rows)
    out_path = output_dir / f"{site_name}_table1.csv"
    out_df.write_csv(str(out_path))
    print(f"  Table 1 written: {out_path}")
