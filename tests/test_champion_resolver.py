import pandas as pd
import pytest

from core.champion_resolver import ChampionResolver


# Mirrors the real data/champions.csv schema, produced by
# scripts/download_champions_and_icons.py: champion_id, display_name,
# sanitized_name, alias. A few champions where the Riot API key and the
# sanitized filename both diverge from the display name are included
# deliberately, since those are exactly the cases a naive resolver gets
# wrong.
def _champions_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"champion_id": 266, "display_name": "Aatrox", "sanitized_name": "aatrox", "alias": "Aatrox"},
            {"champion_id": 103, "display_name": "Ahri", "sanitized_name": "ahri", "alias": "Ahri"},
            {"champion_id": 62, "display_name": "Wukong", "sanitized_name": "wukong", "alias": "MonkeyKing"},
            {"champion_id": 36, "display_name": "Dr. Mundo", "sanitized_name": "dr_mundo", "alias": "DrMundo"},
            {"champion_id": 20, "display_name": "Nunu & Willump", "sanitized_name": "nunu__willump", "alias": "Nunu"},
            {"champion_id": 121, "display_name": "Kha'Zix", "sanitized_name": "khazix", "alias": "Khazix"},
        ]
    )


@pytest.fixture
def resolver() -> ChampionResolver:
    return ChampionResolver(_champions_df())


def test_resolve_id_returns_display_name(resolver):
    assert resolver.resolve_id(266) == "Aatrox"
    assert resolver.resolve_id("62") == "Wukong"


@pytest.mark.parametrize("bad_id", [None, 0, -1, "not-a-number", 999999])
def test_resolve_id_returns_none_for_invalid_or_unknown_ids(resolver, bad_id):
    assert resolver.resolve_id(bad_id) is None


def test_resolve_name_is_punctuation_and_case_insensitive(resolver):
    assert resolver.resolve_name("khazix") == "Kha'Zix"
    assert resolver.resolve_name("KHA'ZIX") == "Kha'Zix"
    assert resolver.resolve_name(" dr mundo ") == "Dr. Mundo"


def test_resolve_name_matches_by_riot_key(resolver):
    assert resolver.resolve_name("MonkeyKing") == "Wukong"
    assert resolver.resolve_name("monkeyking") == "Wukong"


def test_resolve_name_returns_none_for_unknown_or_empty(resolver):
    assert resolver.resolve_name("NotAChampion") is None
    assert resolver.resolve_name("") is None
    assert resolver.resolve_name(None) is None


def test_riot_key_for_name_reads_the_alias_column(resolver):
    # Regression guard: the resolver used to only recognize a "riot_key"
    # header, so it silently ignored the CSV's real "alias" column and
    # fell back to using display_name as the key for every champion.
    assert resolver.riot_key_for_name("Wukong") == "MonkeyKing"
    assert resolver.riot_key_for_name("Nunu & Willump") == "Nunu"
    assert resolver.riot_key_for_name("Dr. Mundo") == "DrMundo"
    assert resolver.riot_key_for_name("Aatrox") == "Aatrox"


def test_normalized_name_for_name_reads_the_sanitized_name_column(resolver):
    # Same regression guard as above, for the "sanitized_name" header.
    assert resolver.normalized_name_for_name("Nunu & Willump") == "nunu__willump"
    assert resolver.normalized_name_for_name("Dr. Mundo") == "dr_mundo"


def test_icon_name_candidates_priority_and_dedup(resolver):
    candidates = resolver.icon_name_candidates("Wukong")

    # normalized_name first, then riot_key. Every other variant
    # (normalized_name.lower(), riot_key.lower(), display name, compact
    # display name, and their lowercased forms) case-insensitively
    # collapses onto one of these two for this champion.
    assert candidates == ["wukong", "MonkeyKing"]

    # No case-insensitive duplicates.
    assert len(candidates) == len(set(c.lower() for c in candidates))


def test_icon_name_candidates_returns_empty_list_for_unknown_champion(resolver):
    assert resolver.icon_name_candidates("NotAChampion") == []


def test_champions_returns_sorted_unique_display_names(resolver):
    assert resolver.champions() == sorted(
        ["Aatrox", "Ahri", "Wukong", "Dr. Mundo", "Nunu & Willump", "Kha'Zix"]
    )


def test_rows_with_missing_display_name_or_bad_champion_id_are_skipped():
    df = pd.DataFrame(
        [
            {"champion_id": 1, "display_name": "Aatrox", "sanitized_name": "aatrox", "alias": "Aatrox"},
            {"champion_id": "not-a-number", "display_name": "Ahri", "sanitized_name": "ahri", "alias": "Ahri"},
            {"champion_id": 3, "display_name": "", "sanitized_name": "", "alias": ""},
        ]
    )

    resolver = ChampionResolver(df)

    assert resolver.champions() == ["Aatrox"]
    assert resolver.resolve_id(1) == "Aatrox"
    assert resolver.resolve_id(3) is None


def test_from_csv_handles_headerless_positional_columns(tmp_path):
    csv_path = tmp_path / "champions.csv"
    csv_path.write_text(
        "266,Aatrox,aatrox,Aatrox\n103,Ahri,ahri,Ahri\n",
        encoding="utf-8",
    )

    resolver = ChampionResolver.from_csv(str(csv_path))

    assert resolver.resolve_id(266) == "Aatrox"
    assert resolver.resolve_name("Ahri") == "Ahri"


def test_app_champion_list_remaps_to_the_apps_own_casing():
    df = pd.DataFrame(
        [{"champion_id": 121, "display_name": "Khazix", "sanitized_name": "khazix", "alias": "Khazix"}]
    )

    resolver = ChampionResolver(df, app_champion_list=["Kha'Zix"])

    assert resolver.resolve_id(121) == "Kha'Zix"
    assert resolver.resolve_name("khazix") == "Kha'Zix"


def test_missing_champion_id_column_raises():
    # display_name/normalized_name/riot_key can all be derived or defaulted,
    # but there's no champion_id to derive - this must fail loudly rather
    # than silently produce an unusable resolver.
    df = pd.DataFrame([{"display_name": "Aatrox"}])

    with pytest.raises(ValueError):
        ChampionResolver(df)
