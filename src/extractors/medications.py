"""Domain 7: Medications - admin flags and max doses in first 24h."""

import polars as pl
from .base import BaseExtractor


class MedicationsExtractor(BaseExtractor):
    crosswalk_file = "medications.yaml"

    def extract(self) -> pl.DataFrame:
        return self._extract_derived()
