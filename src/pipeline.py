"""
Pipeline orchestrator: load config -> build cohort -> run domains -> merge -> write.
"""

from pathlib import Path
from typing import Any

import polars as pl
import yaml
from tqdm import tqdm

from .cohort import build_icu_cohort
from .extractors.advanced_directives import AdvancedDirectivesExtractor
from .extractors.anthropometrics import AnthropometricsExtractor
from .extractors.comorbidities import ComorbiditiesExtractor
from .extractors.diagnoses import DiagnosesExtractor
from .extractors.diagnostic_tests import DiagnosticTestsExtractor
from .extractors.interventions import InterventionsExtractor
from .extractors.medications import MedicationsExtractor
from .extractors.objective_assessments import ObjectiveAssessmentsExtractor
from .extractors.outcomes import OutcomesExtractor

DOMAIN_EXTRACTORS = {
    "advanced_directives": AdvancedDirectivesExtractor,
    "anthropometrics": AnthropometricsExtractor,
    "comorbidities": ComorbiditiesExtractor,
    "diagnoses": DiagnosesExtractor,
    "diagnostic_tests": DiagnosticTestsExtractor,
    "interventions": InterventionsExtractor,
    "medications": MedicationsExtractor,
    "objective_assessments": ObjectiveAssessmentsExtractor,
    "outcomes": OutcomesExtractor,
}


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate the pipeline config YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    required = ["site_name", "clif_directory", "filetype", "output_directory"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    return config


def apply_crosswalks(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, str], list[dict]]:
    """
    Rename pipeline columns to CDE names and keep only active (mapped) CDEs.

    The domain crosswalk YAMLs list all adult C2D2 CDEs; entries with
    source_type 'unmapped' are deactivated and produce no output column.
    Returns the final DataFrame, a {cde_name: domain} map and all crosswalk
    entries (each tagged with its domain).
    """
    crosswalk_dir = Path(__file__).parent / "crosswalks"
    domains: dict[str, str] = {}
    renames: dict[str, str] = {}
    entries: list[dict] = []
    for domain in DOMAIN_EXTRACTORS:
        with open(crosswalk_dir / f"{domain}.yaml") as f:
            for entry in yaml.safe_load(f) or []:
                name = entry["c2d2_name"]
                entries.append({**entry, "domain": domain})
                if name in domains:
                    raise ValueError(f"Duplicate CDE in crosswalks: {name}")
                if entry["source_type"] == "unmapped":
                    continue
                domains[name] = domain
                pcol = entry.get("pipeline_col")
                if pcol and pcol != name:
                    renames[pcol] = name

    df = df.rename({k: v for k, v in renames.items() if k in df.columns})

    missing = [c for c in domains if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} mapped CDEs not produced (domain disabled or failed): {missing}")
    keep = [c for c in domains if c in df.columns]
    return df.select("patient_id", "hospitalization_id", "icu_stay_id", *keep), {c: domains[c] for c in keep}, entries


def cde_status(df: pl.DataFrame, entries: list[dict], clif_dir: str, filetype: str) -> pl.DataFrame:
    """
    One row per CDE saying whether it was produced at this site and why not.

    Unmapped CDEs carry their CLIF-level clif_status from the crosswalk. Mapped
    CDEs are 'site_table_missing' (a source table file is absent),
    'site_no_data' (column all null) or 'available'.
    """
    rows = []
    for e in entries:
        name = e["c2d2_name"]
        n = df[name].count() if name in df.columns else 0
        if e["source_type"] == "unmapped":
            status = e["clif_status"]
        else:
            tables = e.get("clif_tables") or [e["clif_table"]]
            missing = [t for t in tables if not (Path(clif_dir) / f"clif_{t}.{filetype}").exists()]
            status = "site_table_missing" if missing else "site_no_data" if n == 0 else "available"
        rows.append({
            "c2d2_name": name,
            "domain": e["domain"],
            "source_type": e["source_type"],
            "status": status,
            "reason": e.get("reason"),
            "n_nonnull": n,
            "pct_nonnull": round(100 * n / df.height, 1) if df.height else 0.0,
        })
    return pl.DataFrame(rows)


def run_pipeline(config_path: str) -> None:
    """
    Run the full CLIF-to-C2D2 ETL pipeline.

    1. Load config
    2. Build ICU cohort
    3. Run each enabled domain extractor
    4. Merge all domain results into super-wide DataFrame
    5. Write output files
    """
    config = load_config(config_path)

    clif_dir = config["clif_directory"]
    filetype = config["filetype"]
    timezone = config.get("timezone")
    site_name = config["site_name"]
    output_dir = Path(config["output_directory"])
    output_formats = config.get("output_formats", ["parquet"])
    domains_config = config.get("domains", {})

    # Step 1: Build ICU cohort
    print(f"Building ICU cohort from {clif_dir}...")
    cohort_df = build_icu_cohort(clif_dir, filetype, timezone)
    n_stays = len(cohort_df)
    n_patients = cohort_df["patient_id"].n_unique()
    print(f"  Found {n_stays} ICU stays across {n_patients} patients")

    if cohort_df.is_empty():
        print("No ICU stays found. Exiting.")
        return

    # Step 2: Run enabled domain extractors
    # Start with cohort identifiers
    result_df = cohort_df.select(
        "patient_id", "hospitalization_id", "icu_stay_id"
    )

    enabled_domains = [
        name for name, enabled in domains_config.items()
        if enabled and name in DOMAIN_EXTRACTORS
    ]

    for domain_name in tqdm(enabled_domains, desc="Extracting domains"):
        extractor_cls = DOMAIN_EXTRACTORS[domain_name]
        extractor = extractor_cls(
            cohort_df=cohort_df,
            clif_directory=clif_dir,
            filetype=filetype,
            timezone=timezone,
        )

        try:
            domain_df = extractor.extract()
        except Exception as e:
            print(f"  WARNING: Domain '{domain_name}' failed: {e}")
            continue

        # Left-join on icu_stay_id
        # Drop icu_stay_id duplicates before joining
        new_cols = [c for c in domain_df.columns if c != "icu_stay_id"]
        # Drop any columns that already exist in result to avoid duplication
        existing_cols = set(result_df.columns)
        cols_to_keep = ["icu_stay_id"] + [c for c in new_cols if c not in existing_cols]
        domain_df = domain_df.select(cols_to_keep)

        result_df = result_df.join(domain_df, on="icu_stay_id", how="left")

    # Step 3: Apply crosswalks (rename to CDE names, keep active CDEs only)
    result_df, cde_domains, entries = apply_crosswalks(result_df)

    # Step 4: Write output
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in output_formats:
        if fmt == "parquet":
            out_path = output_dir / f"{site_name}_c2d2.parquet"
            result_df.write_parquet(str(out_path))
            print(f"  Written: {out_path}")
        elif fmt == "csv":
            out_path = output_dir / f"{site_name}_c2d2.csv"
            result_df.write_csv(str(out_path))
            print(f"  Written: {out_path}")

    status = cde_status(result_df, entries, clif_dir, filetype)
    status.write_csv(str(output_dir / f"{site_name}_c2d2_cde_status.csv"))
    counts = status["status"].value_counts(sort=True).iter_rows()
    print("  CDE status: " + ", ".join(f"{c} {s}" for s, c in counts))

    # Step 5: Generate Table 1 if requested
    if config.get("generate_table1", False):
        from .table1 import generate_table1

        generate_table1(result_df, output_dir, site_name, cde_domains)

    print(f"Done. Output shape: {result_df.shape} "
          f"({result_df.shape[0]} rows, {result_df.shape[1]} columns)")
