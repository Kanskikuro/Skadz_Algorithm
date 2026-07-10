import pytest

from core.recommend import WorstEnemyResponse, get_champion_scores_for_role


class FakeMatchupRepo:
    """
    In-memory matchup_repo stand-in.

    pair_scores keys are exactly the (champ1, role1, relation_type, champ2,
    role2) tuples core.recommend's forward lookup direction queries with -
    lookup_pair_values() tries this exact direction first, then the reversed
    pair, so tests only need to define the direction they care about.
    """

    def __init__(self, pair_scores, roles_by_champion, pair_deltas=None):
        self._pair_scores = pair_scores
        self._roles_by_champion = roles_by_champion
        self._pair_deltas = pair_deltas or {}

    def get_pair_score(self, champ1, role1, relation_type, champ2, role2, method="Bayesian", default=0.0):
        return self._pair_scores.get((champ1, role1, relation_type, champ2, role2), default)

    def get_pair_delta(self, champ1, role1, relation_type, champ2, role2, method="Bayesian", default=0.0):
        return self._pair_deltas.get((champ1, role1, relation_type, champ2, role2), default)

    def roles_for_champion(self, champion):
        return self._roles_by_champion.get(champion, set())

    def champion_roles(self):
        return {
            (champ, role)
            for champ, roles in self._roles_by_champion.items()
            for role in roles
        }


def _scores_by_champ(repo, pick_strategy, ally_team=None, enemy_team=None):
    scores = get_champion_scores_for_role(
        matchup_repo=repo,
        method="Bayesian",
        role_to_fill="jungle",
        ally_team=ally_team or {},
        enemy_team=enemy_team or {"support": "Thresh"},
        pick_strategy=pick_strategy,
        champion_pool=["LeeSin", "Nunu"],
        enemy_candidate_pool=["LeeSin", "Nunu", "Malphite", "Senna"],
    )

    return {champ: (log_odds, delta, worst) for champ, log_odds, delta, worst in scores}


def _trap_scenario_repo() -> FakeMatchupRepo:
    # Malphite (top) is a devastating but poorly-synergistic counter to
    # LeeSin - exactly the "trap" MinimaxAllRoles is meant to surface: the
    # enemy's strongest response to our jungle pick is a bad fit for their
    # own team.
    return FakeMatchupRepo(
        pair_scores={
            ("Malphite", "top", "Counter", "LeeSin", "jungle"): 5.0,
            # No Thresh synergy defined for Malphite -> synergy defaults to 0.
        },
        roles_by_champion={
            "LeeSin": {"jungle"},
            "Nunu": {"jungle"},
            "Malphite": {"top"},
        },
    )


def _genuine_threat_scenario_repo() -> FakeMatchupRepo:
    # Senna (bottom) both synergizes well with Thresh AND counters LeeSin -
    # a real threat with no dilemma for the enemy, unlike the trap scenario.
    return FakeMatchupRepo(
        pair_scores={
            ("Senna", "bottom", "Synergy", "Thresh", "support"): 1.0,
            ("Senna", "bottom", "Counter", "LeeSin", "jungle"): 2.0,
        },
        roles_by_champion={
            "LeeSin": {"jungle"},
            "Nunu": {"jungle"},
            "Senna": {"bottom"},
        },
    )


def test_maximize_never_reports_a_worst_response():
    repo = _trap_scenario_repo()

    scores = _scores_by_champ(repo, "Maximize")

    assert scores["LeeSin"][2] is None
    assert scores["Nunu"][2] is None


def test_minimax_identifies_the_strongest_response_and_its_team_fit():
    repo = FakeMatchupRepo(
        pair_scores={
            ("Malphite", "top", "Counter", "LeeSin", "jungle"): 5.0,
        },
        roles_by_champion={"LeeSin": {"jungle"}, "Nunu": {"jungle"}, "Malphite": {"top"}},
    )

    scores = _scores_by_champ(repo, "MinimaxAllRoles")
    _log_odds, _delta, worst = scores["LeeSin"]

    assert worst == WorstEnemyResponse(
        champion="Malphite",
        role="top",
        synergy_log_odds=0.0,
        counter_log_odds=5.0,
    )


def test_minimax_worst_response_is_consistent_with_the_final_score():
    repo = _trap_scenario_repo()

    scores = _scores_by_champ(repo, "MinimaxAllRoles")
    log_odds, _delta, worst = scores["LeeSin"]

    # final_log_odds = base_log_odds (0, no ally/known-enemy data here) minus
    # the worst response's own combined synergy + counter contribution.
    assert log_odds == pytest.approx(-(worst.synergy_log_odds + worst.counter_log_odds))


def test_minimax_worst_response_can_be_a_genuine_threat_not_just_a_trap():
    repo = _genuine_threat_scenario_repo()

    scores = _scores_by_champ(repo, "MinimaxAllRoles")
    _log_odds, _delta, worst = scores["LeeSin"]

    assert worst.champion == "Senna"
    assert worst.synergy_log_odds > 0  # good fit for the enemy team - not a trap


def test_minimax_reports_no_worst_response_when_no_threat_exceeds_zero():
    # Nunu has no counter or synergy data defined against it anywhere in
    # this fixture, so every candidate enemy response nets to 0 - there is
    # no meaningful "worst case" to report, even though MinimaxAllRoles ran.
    repo = FakeMatchupRepo(
        pair_scores={},
        roles_by_champion={"LeeSin": {"jungle"}, "Nunu": {"jungle"}, "Malphite": {"top"}},
    )

    scores = _scores_by_champ(repo, "MinimaxAllRoles")
    log_odds, _delta, worst = scores["Nunu"]

    assert worst is None
    assert log_odds == pytest.approx(0.0)


def test_minimax_delta_comes_from_the_same_candidate_as_worst_response():
    # Malphite has the strongest combined (log_odds) threat but a tiny
    # delta; Senna has a weaker combined threat but a huge delta. final_delta
    # must be driven by whichever candidate actually won the worst-case
    # search (Malphite), not independently re-maximized over delta alone
    # (which would silently pull in Senna's unrelated delta instead).
    repo = FakeMatchupRepo(
        pair_scores={
            ("Malphite", "top", "Counter", "LeeSin", "jungle"): 5.0,
            ("Senna", "bottom", "Counter", "LeeSin", "jungle"): 2.0,
        },
        pair_deltas={
            ("Malphite", "top", "Counter", "LeeSin", "jungle"): 0.1,
            ("Senna", "bottom", "Counter", "LeeSin", "jungle"): 9.0,
        },
        roles_by_champion={
            "LeeSin": {"jungle"},
            "Nunu": {"jungle"},
            "Malphite": {"top"},
            "Senna": {"bottom"},
        },
    )

    scores = _scores_by_champ(repo, "MinimaxAllRoles")
    _log_odds, delta, worst = scores["LeeSin"]

    assert worst.champion == "Malphite"
    assert delta == pytest.approx(-0.1)


def test_hybrid_reports_a_worst_response_only_for_unknown_roles():
    repo = _trap_scenario_repo()

    # jungle is unfilled for the enemy -> Hybrid behaves like MinimaxAllRoles.
    unknown_role_scores = _scores_by_champ(repo, "Hybrid", enemy_team={"support": "Thresh"})
    assert unknown_role_scores["LeeSin"][2] is not None

    # jungle is already known for the enemy -> Hybrid behaves like Maximize.
    known_role_scores = _scores_by_champ(
        repo, "Hybrid", enemy_team={"support": "Thresh", "jungle": "Nunu"}
    )
    assert known_role_scores["LeeSin"][2] is None
