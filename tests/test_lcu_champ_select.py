from core.lcu_champ_select import LcuChampSelectService


class FakeChampionResolver:
    """Minimal stand-in for ChampionResolver, keyed by fake champion ids."""

    _NAMES = {
        1: "Aatrox",
        2: "LeeSin",
        3: "Ahri",
        4: "Jinx",
        5: "Thresh",
        6: "Yasuo",
    }

    def resolve_id(self, champion_id):
        return self._NAMES.get(int(champion_id))


class FakeLcuClient:
    def __init__(self, session):
        self._session = session

    def get_champ_select_session(self):
        return self._session


def make_service(session):
    return LcuChampSelectService(FakeLcuClient(session), FakeChampionResolver())


def test_ally_champs_are_placed_in_their_assigned_lane():
    session = {
        "myTeam": [
            {"cellId": 0, "championId": 1, "assignedPosition": "top"},
            {"cellId": 1, "championId": 2, "assignedPosition": "jungle"},
            {"cellId": 2, "championId": 3, "assignedPosition": "middle"},
            {"cellId": 3, "championId": 4, "assignedPosition": "bottom"},
            {"cellId": 4, "championId": 5, "assignedPosition": "utility"},
        ],
        "theirTeam": [],
        "bans": {"myTeamBans": [], "theirTeamBans": []},
        "actions": [],
    }

    state = make_service(session).read_state()

    assert state.ally_champs_by_role == {
        "top": "Aatrox",
        "jungle": "LeeSin",
        "middle": "Ahri",
        "bottom": "Jinx",
        "support": "Thresh",
    }


def test_lane_assignment_survives_pick_order_not_matching_lane_order():
    # cellId order (pick order) is deliberately scrambled relative to lanes,
    # to guard against regressing to "first picked -> first ally slot".
    session = {
        "myTeam": [
            {"cellId": 0, "championId": 5, "assignedPosition": "utility"},
            {"cellId": 1, "championId": 4, "assignedPosition": "bottom"},
            {"cellId": 2, "championId": 1, "assignedPosition": "top"},
            {"cellId": 3, "championId": 3, "assignedPosition": "middle"},
            {"cellId": 4, "championId": 2, "assignedPosition": "jungle"},
        ],
        "theirTeam": [],
        "bans": {"myTeamBans": [], "theirTeamBans": []},
        "actions": [],
    }

    state = make_service(session).read_state()

    assert state.ally_champs_by_role == {
        "support": "Thresh",
        "bottom": "Jinx",
        "top": "Aatrox",
        "middle": "Ahri",
        "jungle": "LeeSin",
    }


def test_hovered_pick_via_actions_is_placed_in_its_assigned_lane():
    # championId is still 0 on the myTeam entry (not locked in yet); the pick
    # is only visible via the actions list, as happens while a player hovers.
    session = {
        "myTeam": [
            {"cellId": 0, "championId": 0, "assignedPosition": "top"},
        ],
        "theirTeam": [],
        "bans": {"myTeamBans": [], "theirTeamBans": []},
        "actions": [
            [{"type": "pick", "actorCellId": 0, "championId": 1, "completed": False}],
        ],
    }

    state = make_service(session).read_state()

    assert state.ally_champs_by_role == {"top": "Aatrox"}
    assert state.ally_champs == ["Aatrox"]


def test_champ_without_known_lane_is_omitted_from_role_mapping():
    session = {
        "myTeam": [
            {"cellId": 0, "championId": 1, "assignedPosition": ""},
        ],
        "theirTeam": [],
        "bans": {"myTeamBans": [], "theirTeamBans": []},
        "actions": [],
    }

    state = make_service(session).read_state()

    assert state.ally_champs_by_role == {}
    assert state.ally_champs == ["Aatrox"]


def test_duplicate_assigned_position_is_left_unresolved():
    # Two allies mid role-swap can briefly both report "top". Neither should
    # be placed, so both fall back to the role guesser until it settles.
    session = {
        "myTeam": [
            {"cellId": 0, "championId": 1, "assignedPosition": "top"},
            {"cellId": 1, "championId": 2, "assignedPosition": "top"},
            {"cellId": 2, "championId": 3, "assignedPosition": "middle"},
        ],
        "theirTeam": [],
        "bans": {"myTeamBans": [], "theirTeamBans": []},
        "actions": [],
    }

    state = make_service(session).read_state()

    assert state.ally_champs_by_role == {"middle": "Ahri"}
    assert sorted(state.ally_champs) == sorted(["Aatrox", "LeeSin", "Ahri"])


def test_enemy_champs_have_no_lane_mapping():
    # The LCU does not expose theirTeam assignedPosition, so enemy champions
    # must never end up in ally_champs_by_role.
    session = {
        "myTeam": [],
        "theirTeam": [
            {"cellId": 5, "championId": 6, "assignedPosition": "middle"},
        ],
        "bans": {"myTeamBans": [], "theirTeamBans": []},
        "actions": [],
    }

    state = make_service(session).read_state()

    assert state.ally_champs_by_role == {}
    assert state.enemy_champs == ["Yasuo"]


def test_bans_from_both_teams_and_actions_are_merged():
    session = {
        "myTeam": [],
        "theirTeam": [],
        "bans": {"myTeamBans": [2], "theirTeamBans": [3]},
        "actions": [
            [{"type": "ban", "actorCellId": 0, "championId": 4, "completed": True}],
        ],
    }

    state = make_service(session).read_state()

    assert sorted(state.banned_champs) == sorted(["LeeSin", "Ahri", "Jinx"])


def test_read_state_returns_none_without_a_session():
    state = make_service(None).read_state()

    assert state is None
