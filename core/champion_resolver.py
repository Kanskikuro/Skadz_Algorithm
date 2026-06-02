import pandas as pd


class ChampionResolver:
    """
    Resolves champions between:
      - LCU/Data Dragon champion IDs
      - Riot internal keys, e.g. MonkeyKing, DrMundo, Kaisa
      - normalized filenames, e.g. dr_mundo, lee_sin
      - display names, e.g. Wukong, Dr. Mundo, Kai'Sa
    """

    def __init__(
        self,
        champions_df: pd.DataFrame,
        app_champion_list: list[str] | None = None,
    ):
        self._df = champions_df.copy()
        self._normalize_columns()
        self._fill_missing_columns()
        self._validate()

        self._id_to_display: dict[int, str] = {}
        self._display_to_riot_key: dict[str, str] = {}
        self._display_to_normalized_name: dict[str, str] = {}
        self._key_to_display: dict[str, str] = {}
        self._normalized_to_display: dict[str, str] = {}

        for _, row in self._df.iterrows():
            try:
                champion_id = int(row["champion_id"])
            except (TypeError, ValueError):
                continue

            display_name = self._clean_cell(row["display_name"])
            normalized_name = self._clean_cell(row["normalized_name"])
            riot_key = self._clean_cell(row["riot_key"])

            if not display_name:
                continue

            if not normalized_name:
                normalized_name = self._filename_normalize(display_name)

            if not riot_key:
                riot_key = display_name

            self._id_to_display[champion_id] = display_name

            display_key = self._normalize(display_name)

            self._display_to_riot_key[display_key] = riot_key
            self._display_to_normalized_name[display_key] = normalized_name

            self._key_to_display[self._normalize(riot_key)] = display_name
            self._normalized_to_display[self._normalize(normalized_name)] = display_name
            self._normalized_to_display[self._normalize(display_name)] = display_name

        self._app_name_lookup: dict[str, str] = {}

        if app_champion_list:
            self._app_name_lookup = {
                self._normalize(name): name
                for name in app_champion_list
                if self._clean_cell(name)
            }

    @classmethod
    def from_csv(
        cls,
        path: str,
        app_champion_list: list[str] | None = None,
    ) -> "ChampionResolver":
        df = pd.read_csv(path)

        lowered_columns = {str(c).strip().lower() for c in df.columns}

        known_header_names = {
            "id",
            "champion_id",
            "name",
            "display_name",
            "normalized_name",
            "normalized",
            "riot_key",
            "riotkey",
            "key",
        }

        has_known_header = bool(lowered_columns & known_header_names)

        if not has_known_header and df.shape[1] >= 4:
            df = pd.read_csv(
                path,
                header=None,
                names=[
                    "champion_id",
                    "display_name",
                    "normalized_name",
                    "riot_key",
                ],
            )

        return cls(df, app_champion_list=app_champion_list)

    def champions(self) -> list[str]:
        return sorted(
            {
                name
                for name in self._id_to_display.values()
                if name and name.lower() != "nan"
            }
        )

    def resolve_id(self, champion_id: int | None) -> str | None:
        if champion_id is None:
            return None

        try:
            champion_id = int(champion_id)
        except (TypeError, ValueError):
            return None

        if champion_id <= 0:
            return None

        display_name = self._id_to_display.get(champion_id)

        if not display_name:
            return None

        return self._to_app_name(display_name)

    def resolve_name(self, name: str | None) -> str | None:
        if not name:
            return None

        key = self._normalize(name)

        display_name = (
            self._normalized_to_display.get(key)
            or self._key_to_display.get(key)
        )

        if not display_name:
            return None

        return self._to_app_name(display_name)

    def riot_key_for_name(self, name: str | None) -> str | None:
        resolved = self.resolve_name(name)

        if not resolved:
            return None

        return self._display_to_riot_key.get(self._normalize(resolved))

    def normalized_name_for_name(self, name: str | None) -> str | None:
        resolved = self.resolve_name(name)

        if not resolved:
            return None

        return self._display_to_normalized_name.get(self._normalize(resolved))

    def icon_name_candidates(self, name: str | None) -> list[str]:
        """
        Returns possible icon filenames without extension.

        Priority:
          1. normalized_name from champions.csv, e.g. lee_sin
          2. Riot key, e.g. LeeSin
          3. display name variants
        """
        resolved = self.resolve_name(name)

        if not resolved:
            return []

        normalized_name = self.normalized_name_for_name(resolved)
        riot_key = self.riot_key_for_name(resolved)

        compact_display = (
            resolved
            .replace(" ", "")
            .replace(".", "")
            .replace("'", "")
            .replace("&", "and")
            .replace("-", "")
        )

        candidates = []

        if normalized_name:
            candidates.extend([
                normalized_name,
                normalized_name.lower(),
            ])

        if riot_key:
            candidates.extend([
                riot_key,
                riot_key.lower(),
            ])

        candidates.extend([
            resolved,
            resolved.lower(),
            compact_display,
            compact_display.lower(),
        ])

        result: list[str] = []
        seen: set[str] = set()

        for candidate in candidates:
            candidate = self._clean_cell(candidate)

            if not candidate:
                continue

            key = candidate.lower()

            if key in seen:
                continue

            seen.add(key)
            result.append(candidate)

        return result

    def _to_app_name(self, display_name: str) -> str:
        if not self._app_name_lookup:
            return display_name

        return self._app_name_lookup.get(
            self._normalize(display_name),
            display_name,
        )

    def _normalize_columns(self) -> None:
        self._df.columns = [
            str(column).strip().lower()
            for column in self._df.columns
        ]

        rename_map = {
            "id": "champion_id",
            "championid": "champion_id",
            "champion_id": "champion_id",

            "name": "display_name",
            "champion_name": "display_name",
            "display": "display_name",
            "display_name": "display_name",

            "normalized": "normalized_name",
            "normalizedname": "normalized_name",
            "normalized_name": "normalized_name",

            "key": "riot_key",
            "riotkey": "riot_key",
            "riot_key": "riot_key",
        }

        self._df = self._df.rename(
            columns={
                column: rename_map.get(column, column)
                for column in self._df.columns
            }
        )

    def _fill_missing_columns(self) -> None:
        if "display_name" not in self._df.columns and "champion_id" in self._df.columns:
            self._df["display_name"] = ""

        if "normalized_name" not in self._df.columns:
            if "display_name" in self._df.columns:
                self._df["normalized_name"] = (
                    self._df["display_name"]
                    .astype(str)
                    .map(self._filename_normalize)
                )
            else:
                self._df["normalized_name"] = ""

        if "riot_key" not in self._df.columns:
            if "display_name" in self._df.columns:
                self._df["riot_key"] = self._df["display_name"].astype(str)
            else:
                self._df["riot_key"] = ""

    def _validate(self) -> None:
        required = {
            "champion_id",
            "display_name",
            "normalized_name",
            "riot_key",
        }

        missing = required - set(self._df.columns)

        if missing:
            raise ValueError(
                f"Missing champion resolver columns: {missing}. "
                f"Found columns: {list(self._df.columns)}"
            )

    @staticmethod
    def _clean_cell(value) -> str:
        if pd.isna(value):
            return ""

        text = str(value).strip()

        if text.lower() == "nan":
            return ""

        return text

    @staticmethod
    def _normalize(name: str) -> str:
        return (
            str(name)
            .lower()
            .strip()
            .replace("'", "")
            .replace(".", "")
            .replace(" ", "")
            .replace("&", "and")
            .replace("-", "")
            .replace("_", "")
        )

    @staticmethod
    def _filename_normalize(name: str) -> str:
        return (
            str(name)
            .lower()
            .strip()
            .replace("'", "")
            .replace(".", "")
            .replace(" ", "_")
            .replace("&", "")
            .replace("-", "_")
        )