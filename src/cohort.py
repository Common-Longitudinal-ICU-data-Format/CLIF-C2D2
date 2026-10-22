"""
ICU stay identification from CLIF ADT table.

Builds the cohort DataFrame that all domain extractors join against.
Each row represents one ICU stay with time boundaries.
"""

import polars as pl
from datetime import timedelta
from clifpy.utils.io_polars import load_data_polars


ICU_GAP_HOURS = 1  # Gap threshold between consecutive ICU records
MIN_ICU_HOURS = 24  # Keep only ICU stays at least this long
MIN_AGE = 18  # Adults only (hospitalization.age_at_admission)


def build_icu_cohort(
    clif_directory: str,
    filetype: str,
    timezone: str | None = None,
) -> pl.DataFrame:
    """
    Identify adult ICU stays of at least MIN_ICU_HOURS from ADT data.

    Algorithm:
    1. Load ADT table, filter location_category == 'icu'
    2. Sort by hospitalization_id, in_dttm
    3. Detect gaps > ICU_GAP_HOURS between consecutive ICU records -> new stay
    4. Assign icu_stay_id = {hospitalization_id}_icu_{seq}
    5. Compute icu_in_dttm, icu_out_dttm, icu_24h_end_dttm per stay
    6. Keep stays lasting >= MIN_ICU_HOURS (missing icu_out_dttm -> dropped)
    7. Join with hospitalization for patient_id; keep age_at_admission >= MIN_AGE

    Returns
    -------
    pl.DataFrame
        Columns: patient_id, hospitalization_id, icu_stay_id,
                 icu_in_dttm, icu_out_dttm, icu_24h_end_dttm
    """
    # Load ADT filtered to ICU locations
    adt = load_data_polars(
        table_name="adt",
        table_path=clif_directory,
        table_format_type=filetype,
        filters={"location_category": "icu"},
        site_tz=timezone,
        lazy=False,
    )

    if adt.is_empty():
        return pl.DataFrame(
            schema={
                "patient_id": pl.Utf8,
                "hospitalization_id": pl.Utf8,
                "icu_stay_id": pl.Utf8,
                "icu_in_dttm": pl.Datetime,
                "icu_out_dttm": pl.Datetime,
                "icu_24h_end_dttm": pl.Datetime,
            }
        )

    # Sort by hospitalization and time
    adt = adt.sort(["hospitalization_id", "in_dttm"])

    # Detect ICU stay boundaries: gap > threshold between consecutive records
    adt = adt.with_columns(
        pl.col("in_dttm")
        .diff()
        .over("hospitalization_id")
        .alias("_gap"),
        pl.col("hospitalization_id")
        .shift(1)
        .alias("_prev_hosp"),
    )

    # Also look at gap between current in_dttm and previous out_dttm
    adt = adt.with_columns(
        (
            pl.col("in_dttm")
            - pl.col("out_dttm").shift(1).over("hospitalization_id")
        ).alias("_transfer_gap")
    )

    # New stay when: first record for a hospitalization, or gap > threshold
    gap_threshold = timedelta(hours=ICU_GAP_HOURS)
    adt = adt.with_columns(
        pl.when(
            pl.col("_transfer_gap").is_null()
            | (pl.col("_transfer_gap") > gap_threshold)
        )
        .then(1)
        .otherwise(0)
        .alias("_new_stay")
    )

    # Assign sequential ICU stay number within each hospitalization
    adt = adt.with_columns(
        pl.col("_new_stay")
        .cum_sum()
        .over("hospitalization_id")
        .alias("_stay_seq")
    )

    # Build icu_stay_id
    adt = adt.with_columns(
        (
            pl.col("hospitalization_id")
            + pl.lit("_icu_")
            + pl.col("_stay_seq").cast(pl.Utf8)
        ).alias("icu_stay_id")
    )

    # Aggregate per ICU stay
    cohort = adt.group_by(["hospitalization_id", "icu_stay_id"]).agg(
        pl.col("in_dttm").min().alias("icu_in_dttm"),
        pl.col("out_dttm").max().alias("icu_out_dttm"),
    )

    # Keep only stays of at least MIN_ICU_HOURS
    cohort = cohort.filter(
        pl.col("icu_out_dttm") - pl.col("icu_in_dttm") >= timedelta(hours=MIN_ICU_HOURS)
    )

    # Add 24h end time
    cohort = cohort.with_columns(
        (pl.col("icu_in_dttm") + timedelta(hours=24)).alias("icu_24h_end_dttm")
    )

    # Join with hospitalization for patient_id, adults only
    hosp = load_data_polars(
        table_name="hospitalization",
        table_path=clif_directory,
        table_format_type=filetype,
        columns=["patient_id", "hospitalization_id", "age_at_admission"],
        site_tz=timezone,
        lazy=False,
    ).filter(pl.col("age_at_admission") >= MIN_AGE)

    cohort = cohort.join(hosp, on="hospitalization_id", how="inner")

    # Select final columns in order
    cohort = cohort.select(
        "patient_id",
        "hospitalization_id",
        "icu_stay_id",
        "icu_in_dttm",
        "icu_out_dttm",
        "icu_24h_end_dttm",
    )

    return cohort.sort("icu_stay_id")
