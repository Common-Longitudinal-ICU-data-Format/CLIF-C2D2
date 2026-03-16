"""Domain 9: Outcomes - LOS, admission/discharge, mortality."""

import polars as pl
from .base import BaseExtractor


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
                columns=["hospitalization_id", "admission_dttm", "discharge_dttm"],
            )
            cohort_hosp = self.cohort_df.select("icu_stay_id", "hospitalization_id")
            hosp_los = cohort_hosp.join(hosp, on="hospitalization_id", how="left")
            hosp_los = hosp_los.with_columns(
                (
                    (pl.col("discharge_dttm") - pl.col("admission_dttm")).dt.total_hours() / 24.0
                ).alias("hospital_los_days")
            ).select("icu_stay_id", "hospital_los_days")
        except FileNotFoundError:
            hosp_los = self.cohort_df.select("icu_stay_id").with_columns(
                pl.lit(None).cast(pl.Float64).alias("hospital_los_days")
            )

        # Hospital mortality flag
        try:
            hosp_data = self._load_clif_table(
                "hospitalization",
                columns=["hospitalization_id", "discharge_category"],
            )
            cohort_hosp = self.cohort_df.select("icu_stay_id", "hospitalization_id")
            mort = cohort_hosp.join(hosp_data, on="hospitalization_id", how="left")
            mort = mort.with_columns(
                pl.when(pl.col("discharge_category") == "Expired")
                .then(1)
                .otherwise(0)
                .alias("hospital_mortality")
            ).select("icu_stay_id", "hospital_mortality")
        except FileNotFoundError:
            mort = self.cohort_df.select("icu_stay_id").with_columns(
                pl.lit(None).cast(pl.Int32).alias("hospital_mortality")
            )

        # ICU in/out times
        icu_times = self.cohort_df.select(
            "icu_stay_id",
            pl.col("icu_in_dttm").alias("C2D2_icu_admission_dttm"),
            pl.col("icu_out_dttm").alias("C2D2_icu_discharge_dttm"),
        )

        result = direct
        for df in [los.select("icu_stay_id", "icu_los_days"), hosp_los, mort, icu_times]:
            result = result.join(df, on="icu_stay_id", how="left")

        return result
