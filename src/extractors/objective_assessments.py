"""Domain 8: Objective Assessments - vitals, GCS, P/F ratio, RASS."""

import polars as pl
from .base import BaseExtractor


class ObjectiveAssessmentsExtractor(BaseExtractor):
    crosswalk_file = "objective_assessments.yaml"

    def extract(self) -> pl.DataFrame:
        derived = self._extract_derived()

        # Add P/F ratio: PaO2 / FiO2 (worst in first 24h)
        pf_ratio = self._compute_pf_ratio()
        result = derived.join(pf_ratio, on="icu_stay_id", how="left")

        return result

    def _compute_pf_ratio(self) -> pl.DataFrame:
        """
        Compute worst (min) P/F ratio in first 24h.

        P/F = PaO2 (from labs, po2_arterial) / FiO2 (from respiratory_support, fio2_set)
        Uses nearest-prior FiO2 for each PaO2 measurement.
        """
        base = self.cohort_df.select("icu_stay_id")

        try:
            labs = self._load_clif_table(
                "labs",
                filters={"lab_category": "po2_arterial"},
            )
            resp = self._load_clif_table("respiratory_support")
        except FileNotFoundError:
            return base.with_columns(
                pl.lit(None).cast(pl.Float64).alias("icu_24hr_pf_ratio_min")
            )

        # Filter labs to first 24h
        labs_filtered = self._filter_to_time_window(labs, "lab_result_dttm", "icu_first_24h")
        if labs_filtered.is_empty():
            return base.with_columns(
                pl.lit(None).cast(pl.Float64).alias("icu_24hr_pf_ratio_min")
            )

        # Get FiO2 values within the ICU stay
        resp_filtered = self._filter_to_time_window(resp, "recorded_dttm", "icu_first_24h")

        # For each PaO2, find the nearest prior FiO2 using asof join
        pao2 = labs_filtered.select(
            "icu_stay_id", "lab_result_dttm", "lab_value_numeric"
        ).rename({"lab_value_numeric": "pao2", "lab_result_dttm": "time"}).sort("time")

        fio2_cols = ["icu_stay_id", "recorded_dttm"]
        if "fio2_set" in resp_filtered.columns:
            fio2_cols.append("fio2_set")
        else:
            return base.with_columns(
                pl.lit(None).cast(pl.Float64).alias("icu_24hr_pf_ratio_min")
            )

        fio2 = resp_filtered.select(fio2_cols).rename(
            {"recorded_dttm": "time"}
        ).filter(pl.col("fio2_set").is_not_null()).sort("time")

        if fio2.is_empty():
            return base.with_columns(
                pl.lit(None).cast(pl.Float64).alias("icu_24hr_pf_ratio_min")
            )

        # Asof join: for each PaO2, get nearest prior FiO2
        joined = pao2.join_asof(
            fio2,
            on="time",
            by="icu_stay_id",
            strategy="backward",
        )

        # Compute P/F ratio (ensure FiO2 is as fraction, not percentage)
        joined = joined.with_columns(
            pl.when(pl.col("fio2_set") > 1.0)
            .then(pl.col("fio2_set") / 100.0)
            .otherwise(pl.col("fio2_set"))
            .alias("fio2_fraction")
        )

        joined = joined.with_columns(
            (pl.col("pao2") / pl.col("fio2_fraction")).alias("pf_ratio")
        )

        # Get worst (min) P/F ratio per ICU stay
        result = joined.group_by("icu_stay_id").agg(
            pl.col("pf_ratio").min().alias("icu_24hr_pf_ratio_min")
        )

        return base.join(result, on="icu_stay_id", how="left")
