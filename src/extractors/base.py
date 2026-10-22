"""
Abstract base extractor for C2D2 domain extraction.

All domain extractors inherit from BaseExtractor and implement extract().
Provides shared helpers for loading CLIF tables and time-window filtering.
"""

import abc
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from clifpy.utils.io_polars import load_data_polars


class BaseExtractor(abc.ABC):
    """Base class for all C2D2 domain extractors."""

    # Subclasses set this to their crosswalk filename (e.g., "outcomes.yaml")
    crosswalk_file: str = ""

    def __init__(
        self,
        cohort_df: pl.DataFrame,
        clif_directory: str,
        filetype: str,
        timezone: str | None = None,
    ):
        self.cohort_df = cohort_df
        self.clif_directory = clif_directory
        self.filetype = filetype
        self.timezone = timezone
        self.crosswalk = self._load_crosswalk()

    def _load_crosswalk(self) -> list[dict[str, Any]]:
        """Load the crosswalk YAML for this domain."""
        if not self.crosswalk_file:
            return []
        path = Path(__file__).parent.parent / "crosswalks" / self.crosswalk_file
        if not path.exists():
            return []
        with open(path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, list) else []

    def _load_clif_table(
        self,
        table_name: str,
        columns: list[str] | None = None,
        filters: dict | None = None,
    ) -> pl.DataFrame:
        """Load a CLIF table eagerly."""
        return load_data_polars(
            table_name=table_name,
            table_path=self.clif_directory,
            table_format_type=self.filetype,
            columns=columns,
            filters=filters,
            site_tz=self.timezone,
            lazy=False,
        )

    def _filter_to_time_window(
        self,
        df: pl.DataFrame,
        time_col: str,
        window: str,
    ) -> pl.DataFrame:
        """
        Join df with cohort and filter to the specified time window.

        Parameters
        ----------
        df : pl.DataFrame
            Must have hospitalization_id and a datetime column.
        time_col : str
            Name of the datetime column to filter on.
        window : str
            'icu_first_24h' or 'icu_full_stay'

        Returns
        -------
        pl.DataFrame
            Filtered df with icu_stay_id added.
        """
        joined = df.join(
            self.cohort_df.select(
                "hospitalization_id",
                "icu_stay_id",
                "icu_in_dttm",
                "icu_out_dttm",
                "icu_24h_end_dttm",
            ),
            on="hospitalization_id",
            how="inner",
        )

        if window == "icu_first_24h":
            return joined.filter(
                (pl.col(time_col) >= pl.col("icu_in_dttm"))
                & (pl.col(time_col) <= pl.col("icu_24h_end_dttm"))
            )
        elif window == "icu_full_stay":
            return joined.filter(
                (pl.col(time_col) >= pl.col("icu_in_dttm"))
                & (pl.col(time_col) <= pl.col("icu_out_dttm"))
            )
        else:
            raise ValueError(f"Unknown time window: {window}")

    def _extract_derived(self) -> pl.DataFrame:
        """
        Generic extraction for crosswalk entries with source_type == 'derived'.

        Handles time-windowed aggregation (max, min, first, last, any, sum).
        """
        derived = [e for e in self.crosswalk if e.get("source_type") == "derived"]
        if not derived:
            return self.cohort_df.select("icu_stay_id")

        result = self.cohort_df.select("icu_stay_id")

        # Group by clif_table + time_window + category_column to batch-load
        # This prevents mixing types (e.g., string device_category with int tracheostomy)
        batches: dict[tuple[str, str, str | None], list[dict]] = {}
        for entry in derived:
            cat_col = entry.get("clif_category_column") or None
            key = (entry["clif_table"], entry.get("time_window", "icu_first_24h"), cat_col)
            batches.setdefault(key, []).append(entry)

        for (table_name, window, cat_col), entries in batches.items():
            # Determine which category values we need (an entry may list several)
            cat_values = list({
                v
                for e in entries if e.get("clif_category_value") is not None
                for v in (e["clif_category_value"] if isinstance(e["clif_category_value"], list) else [e["clif_category_value"]])
            })
            time_col = entries[0].get("clif_time_column", "recorded_dttm")
            val_col = entries[0].get("clif_value_column", "lab_value_numeric")

            # Load table with category filter if applicable
            # Only use predicate pushdown for string filters; non-string values
            # (int/bool) can cause type mismatches and are filtered per-entry instead
            filters = {}
            if cat_col and cat_values and all(isinstance(v, str) for v in cat_values):
                filters[cat_col] = cat_values

            try:
                df = self._load_clif_table(table_name, filters=filters)
            except FileNotFoundError:
                # Table doesn't exist at this site - produce nulls
                print(f"  {table_name}: local table not available, "
                      f"{[e['c2d2_name'] for e in entries]} set to null")
                for entry in entries:
                    result = result.with_columns(pl.lit(None).alias(entry["c2d2_name"]))
                continue

            # Filter to time window
            df = self._filter_to_time_window(df, time_col, window)

            # For each entry, compute the aggregation
            for entry in entries:
                c2d2_name = entry["c2d2_name"]
                agg = entry.get("aggregation", "max")
                entry_val_col = entry.get("clif_value_column", val_col)

                # Filter to specific category value
                subset = df
                if cat_col and entry.get("clif_category_value") is not None:
                    val = entry["clif_category_value"]
                    if isinstance(val, list):
                        subset = df.filter(pl.col(cat_col).is_in(val))
                    else:
                        # Cast filter value to match column dtype
                        if cat_col in df.columns:
                            col_dtype = df[cat_col].dtype
                            if col_dtype == pl.Boolean:
                                val = bool(val)
                            elif col_dtype in (pl.Int32, pl.Int64):
                                val = int(val)
                        subset = df.filter(pl.col(cat_col) == val)

                agg_expr = self._build_agg_expr(entry_val_col, agg, c2d2_name)
                agged = subset.group_by("icu_stay_id").agg(agg_expr)

                result = result.join(agged, on="icu_stay_id", how="left")
                # "any" flags: no rows in the window means 0, not unknown
                if agg == "any":
                    result = result.with_columns(pl.col(c2d2_name).fill_null(0))

        return result

    def _extract_direct(self) -> pl.DataFrame:
        """
        Generic extraction for crosswalk entries with source_type == 'direct'.

        Copies a column directly from a CLIF table.
        """
        direct = [e for e in self.crosswalk if e.get("source_type") == "direct"]
        if not direct:
            return self.cohort_df.select("icu_stay_id")

        result = self.cohort_df.select("icu_stay_id")

        # Group by clif_table
        batches: dict[str, list[dict]] = {}
        for entry in direct:
            batches.setdefault(entry["clif_table"], []).append(entry)

        for table_name, entries in batches.items():
            cols_needed = ["hospitalization_id"] + [e["clif_column"] for e in entries]
            if table_name == "patient":
                cols_needed = ["patient_id"] + [e["clif_column"] for e in entries]

            try:
                df = self._load_clif_table(table_name, columns=list(set(cols_needed)))
            except FileNotFoundError:
                for entry in entries:
                    result = result.with_columns(pl.lit(None).alias(entry["c2d2_name"]))
                continue

            # Join based on table key
            if table_name == "patient":
                # Join through cohort's patient_id
                cohort_with_patient = self.cohort_df.select(
                    "icu_stay_id", "patient_id"
                )
                joined = cohort_with_patient.join(df, on="patient_id", how="left")
            else:
                cohort_with_hosp = self.cohort_df.select(
                    "icu_stay_id", "hospitalization_id"
                )
                joined = cohort_with_hosp.join(
                    df, on="hospitalization_id", how="left"
                )

            # Rename columns
            for entry in entries:
                col = entry["clif_column"]
                c2d2_name = entry["c2d2_name"]
                if col in joined.columns:
                    rename_map = {col: c2d2_name}
                    joined = joined.rename(rename_map)

            # Deduplicate - take first per icu_stay_id
            joined = joined.unique(subset=["icu_stay_id"], keep="first")

            select_cols = ["icu_stay_id"] + [e["c2d2_name"] for e in entries]
            select_cols = [c for c in select_cols if c in joined.columns]
            joined = joined.select(select_cols)

            result = result.join(joined, on="icu_stay_id", how="left")

        return result

    def _extract_pass_through_ontology(self) -> pl.DataFrame:
        """
        Generic extraction for crosswalk entries with source_type == 'pass_through_ontology'.

        Matches ICD10/CPT code prefixes to produce binary flags.
        """
        onto = [e for e in self.crosswalk if e.get("source_type") == "pass_through_ontology"]
        if not onto:
            return self.cohort_df.select("icu_stay_id")

        result = self.cohort_df.select("icu_stay_id")

        # Group by clif_table
        batches: dict[str, list[dict]] = {}
        for entry in onto:
            batches.setdefault(entry["clif_table"], []).append(entry)

        for table_name, entries in batches.items():
            try:
                df = self._load_clif_table(table_name)
            except FileNotFoundError:
                for entry in entries:
                    result = result.with_columns(pl.lit(None).cast(pl.Int32).alias(entry["c2d2_name"]))
                continue

            # Join with cohort to get icu_stay_id
            df = df.join(
                self.cohort_df.select("hospitalization_id", "icu_stay_id"),
                on="hospitalization_id",
                how="inner",
            )

            # Determine the code column
            code_col = "diagnosis_code"
            if "procedure" in table_name:
                code_col = "procedure_code"

            # Clean codes: uppercase, remove decimals
            if code_col in df.columns:
                df = df.with_columns(
                    pl.col(code_col)
                    .str.to_uppercase()
                    .str.replace_all(".", "", literal=True)
                    .alias("_code_clean")
                )
            else:
                for entry in entries:
                    result = result.with_columns(pl.lit(None).cast(pl.Int32).alias(entry["c2d2_name"]))
                continue

            # Optionally filter by code_format
            for entry in entries:
                c2d2_name = entry["c2d2_name"]
                prefixes = entry.get("code_prefixes", [])
                code_format = entry.get("code_format")

                subset = df
                if code_format and "diagnosis_code_format" in df.columns:
                    subset = df.filter(pl.col("diagnosis_code_format") == code_format)

                # Build prefix match expression
                if prefixes:
                    match_expr = pl.lit(False)
                    for prefix in prefixes:
                        match_expr = match_expr | pl.col("_code_clean").str.starts_with(prefix)

                    matched = (
                        subset.filter(match_expr)
                        .group_by("icu_stay_id")
                        .agg(pl.lit(1).alias(c2d2_name))
                    )
                else:
                    matched = pl.DataFrame(
                        schema={"icu_stay_id": pl.Utf8, c2d2_name: pl.Int32}
                    )

                result = result.join(matched, on="icu_stay_id", how="left")
                result = result.with_columns(
                    pl.col(c2d2_name).fill_null(0).cast(pl.Int32)
                )

        return result

    @staticmethod
    def _build_agg_expr(col: str, agg: str, alias: str) -> pl.Expr:
        """Build a Polars aggregation expression."""
        if agg == "max":
            return pl.col(col).max().alias(alias)
        elif agg == "min":
            return pl.col(col).min().alias(alias)
        elif agg == "first":
            return pl.col(col).first().alias(alias)
        elif agg == "last":
            return pl.col(col).last().alias(alias)
        elif agg == "any":
            return pl.col(col).is_not_null().any().cast(pl.Int32).alias(alias)
        elif agg == "sum":
            return pl.col(col).sum().alias(alias)
        elif agg == "mean":
            return pl.col(col).mean().alias(alias)
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

    @abc.abstractmethod
    def extract(self) -> pl.DataFrame:
        """
        Extract C2D2 elements for this domain.

        Returns
        -------
        pl.DataFrame
            Keyed by icu_stay_id with one column per C2D2 element.
        """
        ...
