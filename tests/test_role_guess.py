import pandas as pd
import pytest

from core.repo import PriorsRepository
from core.role_guess import (
    build_priors_lookup,
    guess_enemy_roles,
    resolve_ally_role_assignments,
)


@pytest.fixture
def priors_repo() -> PriorsRepository:
    # Each champion has one clearly dominant role, so the Hungarian guesser
    # has an unambiguous, deterministic answer to assert against.
    df = pd.DataFrame(
        [
            {"champion_name": "Aatrox", "top": 90, "jungle": 5, "middle": 5, "bottom": 0, "support": 0},
            {"champion_name": "LeeSin", "top": 5, "jungle": 90, "middle": 5, "bottom": 0, "support": 0},
            {"champion_name": "Ahri", "top": 0, "jungle": 0, "middle": 95, "bottom": 5, "support": 0},
            {"champion_name": "Jinx", "top": 0, "jungle": 0, "middle": 0, "bottom": 95, "support": 5},
            {"champion_name": "Thresh", "top": 0, "jungle": 0, "middle": 0, "bottom": 5, "support": 95},
        ]
    )

    return PriorsRepository(df)


def test_known_lanes_are_kept_and_remaining_champs_are_guessed(priors_repo):
    ally_champs = ["Aatrox", "LeeSin", "Ahri", "Jinx", "Thresh"]
    ally_champs_by_role = {"top": "Aatrox", "jungle": "LeeSin"}

    assignments = resolve_ally_role_assignments(
        ally_champs,
        ally_champs_by_role,
        priors_repo,
    )

    assert assignments == {
        "top": "Aatrox",
        "jungle": "LeeSin",
        "middle": "Ahri",
        "bottom": "Jinx",
        "support": "Thresh",
    }


def test_fully_known_lanes_are_returned_as_is_without_guessing(priors_repo):
    ally_champs = ["Aatrox", "LeeSin", "Ahri", "Jinx", "Thresh"]
    ally_champs_by_role = {
        "top": "Aatrox",
        "jungle": "LeeSin",
        "middle": "Ahri",
        "bottom": "Jinx",
        "support": "Thresh",
    }

    assignments = resolve_ally_role_assignments(
        ally_champs,
        ally_champs_by_role,
        priors_repo,
    )

    assert assignments == ally_champs_by_role


def test_no_known_lanes_matches_a_plain_role_guess(priors_repo):
    ally_champs = ["Aatrox", "LeeSin", "Ahri", "Jinx", "Thresh"]

    assignments = resolve_ally_role_assignments(
        ally_champs,
        {},
        priors_repo,
    )

    assert assignments == guess_enemy_roles(ally_champs, priors_repo)


def test_a_known_lane_wins_even_against_the_champions_own_dominant_role():
    # Aatrox is overwhelmingly a top-lane pick by prior, but the LCU says
    # this particular Aatrox locked in jungle. The known lane must be kept,
    # and the guesser must place LeeSin into a *different* remaining role
    # rather than also being pulled toward "top".
    df = pd.DataFrame(
        [
            {"champion_name": "Aatrox", "top": 90, "jungle": 5, "middle": 5, "bottom": 0, "support": 0},
            {"champion_name": "LeeSin", "top": 50, "jungle": 90, "middle": 0, "bottom": 0, "support": 0},
        ]
    )
    repo = PriorsRepository(df)

    ally_champs = ["Aatrox", "LeeSin"]
    ally_champs_by_role = {"jungle": "Aatrox"}

    assignments = resolve_ally_role_assignments(
        ally_champs,
        ally_champs_by_role,
        repo,
    )

    assert assignments == {"jungle": "Aatrox", "top": "LeeSin"}


def test_build_priors_lookup_is_cached_on_the_dataframe(priors_repo):
    priors_df = priors_repo.get_df()

    first = build_priors_lookup(priors_df)
    second = build_priors_lookup(priors_df)

    assert first is second
    assert first == {
        "aatrox": {"top": 0.9, "jungle": 0.05, "middle": 0.05, "bottom": 0.0, "support": 0.0},
        "leesin": {"top": 0.05, "jungle": 0.9, "middle": 0.05, "bottom": 0.0, "support": 0.0},
        "ahri": {"top": 0.0, "jungle": 0.0, "middle": 0.95, "bottom": 0.05, "support": 0.0},
        "jinx": {"top": 0.0, "jungle": 0.0, "middle": 0.0, "bottom": 0.95, "support": 0.05},
        "thresh": {"top": 0.0, "jungle": 0.0, "middle": 0.0, "bottom": 0.05, "support": 0.95},
    }


def test_build_priors_lookup_cache_does_not_leak_across_dataframes():
    df_a = pd.DataFrame(
        [{"champion_name": "Aatrox", "top": 90, "jungle": 0, "middle": 0, "bottom": 0, "support": 0}]
    )
    df_b = pd.DataFrame(
        [{"champion_name": "Ahri", "top": 0, "jungle": 0, "middle": 90, "bottom": 0, "support": 0}]
    )

    lookup_a = build_priors_lookup(df_a)
    lookup_b = build_priors_lookup(df_b)

    assert lookup_a is not lookup_b
    assert "aatrox" in lookup_a and "aatrox" not in lookup_b
    assert "ahri" in lookup_b and "ahri" not in lookup_a
