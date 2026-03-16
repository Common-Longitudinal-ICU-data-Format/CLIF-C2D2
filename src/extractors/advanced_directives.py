"""Domain 1: Advanced Directives - most recent code status in first 24h."""

import polars as pl
from .base import BaseExtractor


class AdvancedDirectivesExtractor(BaseExtractor):
    crosswalk_file = "advanced_directives.yaml"

    def extract(self) -> pl.DataFrame:
        # code_status uses patient_id, not hospitalization_id
        try:
            cs = self._load_clif_table("code_status")
        except FileNotFoundError:
            return self.cohort_df.select("icu_stay_id").with_columns(
                pl.lit(None).cast(pl.Utf8).alias("icu_24hr_code_status")
            )

        # Join with cohort via patient_id to get time boundaries
        cohort_cols = self.cohort_df.select(
            "patient_id", "icu_stay_id", "icu_in_dttm", "icu_24h_end_dttm"
        )
        joined = cs.join(cohort_cols, on="patient_id", how="inner")

        # Filter to first 24h
        filtered = joined.filter(
            (pl.col("start_dttm") >= pl.col("icu_in_dttm"))
            & (pl.col("start_dttm") <= pl.col("icu_24h_end_dttm"))
        )

        # Take last (most recent) code status per ICU stay
        result = (
            filtered.sort("start_dttm")
            .group_by("icu_stay_id")
            .agg(pl.col("code_status_category").last().alias("icu_24hr_code_status"))
        )

        # Join back to ensure all ICU stays are represented
        return self.cohort_df.select("icu_stay_id").join(
            result, on="icu_stay_id", how="left"
        )
