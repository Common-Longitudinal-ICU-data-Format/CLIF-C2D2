"""Domain 6: Interventions - respiratory support, ECMO, CRRT flags."""

import polars as pl
from .base import BaseExtractor


class InterventionsExtractor(BaseExtractor):
    crosswalk_file = "interventions.yaml"

    def extract(self) -> pl.DataFrame:
        return self._extract_derived()
