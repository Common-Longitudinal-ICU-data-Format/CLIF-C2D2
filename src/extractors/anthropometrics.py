"""Domain 2: Anthropometrics & Demographics - 8 elements."""

import polars as pl
from .base import BaseExtractor


class AnthropometricsExtractor(BaseExtractor):
    crosswalk_file = "anthropometrics.yaml"

    def extract(self) -> pl.DataFrame:
        direct = self._extract_direct()
        derived = self._extract_derived()

        result = direct.join(derived, on="icu_stay_id", how="left")
        return result
