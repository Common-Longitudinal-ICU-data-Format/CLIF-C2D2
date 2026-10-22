# CLIF-C2D2

Extracts the adult C2D2 common data elements (CDEs) from CLIF tables.

```bash
.venv/bin/python main.py --config config.yaml
```

Each run writes `{site}_c2d2.parquet` and `{site}_c2d2.csv`, which hold the ID columns plus one column per mapped CDE. It also writes `{site}_c2d2_cde_status.csv`, which has one row for each of the 190 CDEs.

## CDE labels

Every CDE in `src/crosswalks/*.yaml` has a method label (`source_type`). The status file also gives each CDE an availability label (`status`). The required keys are enforced by `scripts/check_crosswalk_coverage.py`.

### How a CDE is computed (`source_type`)

| source_type | meaning | required keys |
|---|---|---|
| `direct` | one CLIF column copied as is | `clif_table`, `clif_column` |
| `derived` | YAML only: filter a CLIF table by mCIDE category, restrict to a time window, then aggregate | `clif_table`, `clif_category_column`, `clif_category_value`, `clif_value_column`, `clif_time_column`, `aggregation`, `time_window` |
| `pass_through_ontology` | diagnosis/procedure code prefix match | `clif_table`, `code_format`, `code_prefixes` |
| `custom` | Python logic (multiple steps or tables) | `clif_tables`, `implementation` |
| `unmapped` | not computed, so there is no output column | `clif_status`, `reason` |

Every mapped entry also needs `output_type`.

### Why a CDE is or isn't available (`status`)

| status | where it is set | meaning | whose issue |
|---|---|---|---|
| `available` | at run time | the column was produced and has data | none |
| `site_table_missing` | at run time | the site lacks a CLIF table the CDE needs, so the column is null | site |
| `site_no_data` | at run time | the table exists but holds no matching data, so the column is null | site |
| `not_yet_mapped` | `clif_status` in the YAML | CLIF has the data, but the mapping isn't written yet | this repo |
| `clif_partial` | `clif_status` in the YAML | the CLIF dictionary (mCIDE) covers only part of the concept | CLIF |
| `clif_gap` | `clif_status` in the YAML | not in the CLIF dictionary (mCIDE) | CLIF |
| `external_data` | `clif_status` in the YAML | needs a non-CLIF data source | external |
