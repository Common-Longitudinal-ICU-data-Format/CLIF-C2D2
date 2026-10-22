"""Domain 3: Comorbidities - 17 CCI conditions (CDE-aligned)."""

import polars as pl
from .base import BaseExtractor
from clifpy.utils.comorbidity import calculate_cci


class ComorbiditiesExtractor(BaseExtractor):
    crosswalk_file = "comorbidities.yaml"

    def extract(self) -> pl.DataFrame:
        try:
            dx = self._load_clif_table("hospital_diagnosis")
        except FileNotFoundError:
            return self.cohort_df.select("icu_stay_id")

        # calculate_cci returns a pandas DataFrame
        cci_pd = calculate_cci(dx)
        cci = pl.from_pandas(cci_pd)

        # Ensure hospitalization_id is string for join
        if "hospitalization_id" in cci.columns:
            cci = cci.with_columns(
                pl.col("hospitalization_id").cast(pl.Utf8)
            )

        # Map from hospitalization_id to icu_stay_id
        cohort_map = self.cohort_df.select("hospitalization_id", "icu_stay_id")
        result = cohort_map.join(cci, on="hospitalization_id", how="left")

        # Prefix CCI condition columns with "cci_" to namespace them
        cci_condition_cols = [c for c in result.columns if c not in ("hospitalization_id", "icu_stay_id", "cci_score")]
        rename_map = {c: f"cci_{c}" for c in cci_condition_cols}
        result = result.rename(rename_map)

        result = result.drop("hospitalization_id", "cci_score")
        return result.join(self._extract_pass_through_ontology(), on="icu_stay_id", how="left")
