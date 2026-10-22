"""Domain 9: Outcomes - LOS, admission/discharge, mortality."""

import polars as pl
from datetime import timedelta

from .base import BaseExtractor

PRIOR_HOSP_HOURS = 6  # Prior hospitalization discharged within this window counts as ICU admit source


class OutcomesExtractor(BaseExtractor):
    crosswalk_file = "outcomes.yaml"

    def extract(self) -> pl.DataFrame:
        direct = self._extract_direct()

        # Compute LOS columns from cohort
        los = self.cohort_df.select(
            "icu_stay_id",
            "icu_in_dttm",
            "icu_out_dttm",
        )

        # ICU LOS in days
        los = los.with_columns(
            (
                (pl.col("icu_out_dttm") - pl.col("icu_in_dttm")).dt.total_hours() / 24.0
            ).alias("icu_los_days")
        )

        # Hospital LOS - need hospitalization data
        try:
            hosp = self._load_clif_table(
                "hospitalization",
                columns=["patient_id", "hospitalization_id", "admission_dttm", "discharge_dttm"],
            )
            cohort_hosp = self.cohort_df.select("icu_stay_id", "hospitalization_id")
            hosp_los = cohort_hosp.join(hosp, on="hospitalization_id", how="left")
            hosp_los = hosp_los.with_columns(
                (
                    (pl.col("discharge_dttm") - pl.col("admission_dttm")).dt.total_hours() / 24.0
                ).alias("hospital_los_days")
            ).select("icu_stay_id", "hospital_los_days")
        except FileNotFoundError:
            hosp = None
            hosp_los = self.cohort_df.select("icu_stay_id").with_columns(
                pl.lit(None).cast(pl.Float64).alias("hospital_los_days")
            )

        admit_source = self._icu_admit_source(hosp)

        result = direct
        for df in [los.select("icu_stay_id", "icu_los_days"), hosp_los, admit_source]:
            result = result.join(df, on="icu_stay_id", how="left")

        return result

    def _icu_admit_source(self, hosp: pl.DataFrame | None) -> pl.DataFrame:
        """
        Location before ICU entry: last adt location earlier in the same
        hospitalization; else the last adt location of the patient's prior
        hospitalization discharged <= PRIOR_HOSP_HOURS before ICU entry;
        else 'direct_icu_admit'.
        """
        stays = self.cohort_df.select("icu_stay_id", "patient_id", "hospitalization_id", "icu_in_dttm")
        try:
            adt = self._load_clif_table(
                "adt", columns=["hospitalization_id", "in_dttm", "location_category"]
            ).sort("in_dttm")
        except FileNotFoundError:
            return stays.select("icu_stay_id", pl.lit(None, dtype=pl.Utf8).alias("icu_admit_source"))

        same_hosp = (
            stays.join(adt, on="hospitalization_id")
            .filter(pl.col("in_dttm") < pl.col("icu_in_dttm"))
            .group_by("icu_stay_id")
            .agg(pl.col("location_category").last().alias("_same"))
        )
        stays = stays.join(same_hosp, on="icu_stay_id", how="left")

        if hosp is not None:
            last_loc = adt.group_by("hospitalization_id").agg(pl.col("location_category").last().alias("_prior"))
            prior = (
                stays.join(hosp.select("patient_id", "hospitalization_id", "discharge_dttm"),
                           on="patient_id", suffix="_prev")
                .filter(
                    (pl.col("hospitalization_id_prev") != pl.col("hospitalization_id"))
                    & (pl.col("discharge_dttm") <= pl.col("icu_in_dttm"))
                    & (pl.col("discharge_dttm") >= pl.col("icu_in_dttm") - timedelta(hours=PRIOR_HOSP_HOURS))
                )
                .sort("discharge_dttm")
                .group_by("icu_stay_id")
                .agg(pl.col("hospitalization_id_prev").last())
                .join(last_loc, left_on="hospitalization_id_prev", right_on="hospitalization_id")
                .select("icu_stay_id", "_prior")
            )
            stays = stays.join(prior, on="icu_stay_id", how="left")
        else:
            stays = stays.with_columns(pl.lit(None, dtype=pl.Utf8).alias("_prior"))

        return stays.select(
            "icu_stay_id",
            pl.coalesce("_same", "_prior", pl.lit("direct_icu_admit")).alias("icu_admit_source"),
        )
