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

    # Step 3: Write output
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

    # Step 4: Generate Table 1 if requested
    if config.get("generate_table1", False):
        from .table1 import generate_table1

        generate_table1(result_df, output_dir, site_name)

    print(f"Done. Output shape: {result_df.shape} "
          f"({result_df.shape[0]} rows, {result_df.shape[1]} columns)")
