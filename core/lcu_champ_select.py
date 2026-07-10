from dataclasses import dataclass

from core.champion_resolver import ChampionResolver
from core.lcu_client import LcuClient

# LCU reports the selected lane as "assignedPosition" on each myTeam member.
# It uses "utility" where the app's Role enum uses "support".
_LCU_POSITION_TO_ROLE = {
    "top": "top",
    "jungle": "jungle",
    "middle": "middle",
    "bottom": "bottom",
    "utility": "support",
}


@dataclass(frozen=True)
class LcuChampSelectState:
    ally_champs: list[str]
    enemy_champs: list[str]
    banned_champs: list[str]
    ally_champs_by_role: dict[str, str]


class LcuChampSelectService:
    """
    Reads current champion select state from LCU.

    Includes:
      - locked/selected ally champions
      - locked/selected enemy champions
      - hovered champions
      - banned champions
    """

    def __init__(
        self,
        lcu_client: LcuClient,
        champion_resolver: ChampionResolver,
    ):
        self._lcu_client = lcu_client
        self._champion_resolver = champion_resolver

    def read_state(self) -> LcuChampSelectState | None:
        session = self._lcu_client.get_champ_select_session()

        if not session:
            return None

        ally_cell_ids = {
            member.get("cellId")
            for member in session.get("myTeam", [])
        }

        enemy_cell_ids = {
            member.get("cellId")
            for member in session.get("theirTeam", [])
        }

        ally_ids: dict[int, int] = {}
        enemy_ids: dict[int, int] = {}
        banned_ids: set[int] = set()
        ally_positions: dict[int, str] = {}

        for member in session.get("myTeam", []):
            cell_id = member.get("cellId")
            champion_id = member.get("championId")

            if cell_id is not None and champion_id and champion_id > 0:
                ally_ids[int(cell_id)] = int(champion_id)

            role = _LCU_POSITION_TO_ROLE.get(
                str(member.get("assignedPosition", "")).lower()
            )

            if cell_id is not None and role:
                ally_positions[int(cell_id)] = role

        for member in session.get("theirTeam", []):
            cell_id = member.get("cellId")
            champion_id = member.get("championId")

            if cell_id is not None and champion_id and champion_id > 0:
                enemy_ids[int(cell_id)] = int(champion_id)

        bans = session.get("bans", {})

        for champion_id in bans.get("myTeamBans", []):
            if champion_id and champion_id > 0:
                banned_ids.add(int(champion_id))

        for champion_id in bans.get("theirTeamBans", []):
            if champion_id and champion_id > 0:
                banned_ids.add(int(champion_id))

        for action_group in session.get("actions", []):
            for action in action_group:
                action_type = action.get("type")
                champion_id = action.get("championId")
                actor_cell_id = action.get("actorCellId")

                if not champion_id or champion_id <= 0:
                    continue

                champion_id = int(champion_id)

                if action_type == "pick":
                    if actor_cell_id in ally_cell_ids:
                        ally_ids[int(actor_cell_id)] = champion_id
                    elif actor_cell_id in enemy_cell_ids:
                        enemy_ids[int(actor_cell_id)] = champion_id

                elif action_type == "ban":
                    banned_ids.add(champion_id)

        # Mid role-swap, two allies can briefly report the same
        # assignedPosition. Treat that role as unresolved rather than
        # picking one arbitrarily, so both champs fall back to the role
        # guesser until the swap settles and positions are unambiguous again.
        role_claim_counts: dict[str, int] = {}

        for role in ally_positions.values():
            role_claim_counts[role] = role_claim_counts.get(role, 0) + 1

        ally_champs_by_role: dict[str, str] = {}

        for cell_id, champion_id in ally_ids.items():
            role = ally_positions.get(cell_id)
            name = self._champion_resolver.resolve_id(champion_id)

            if role and name and role_claim_counts.get(role, 0) == 1:
                ally_champs_by_role[role] = name

        return LcuChampSelectState(
            ally_champs=self._ids_to_names(ally_ids.values()),
            enemy_champs=self._ids_to_names(enemy_ids.values()),
            banned_champs=self._ids_to_names(banned_ids),
            ally_champs_by_role=ally_champs_by_role,
        )

    def _ids_to_names(self, champion_ids) -> list[str]:
        names: list[str] = []

        for champion_id in champion_ids:
            name = self._champion_resolver.resolve_id(int(champion_id))

            if name and name not in names:
                names.append(name)

        return names