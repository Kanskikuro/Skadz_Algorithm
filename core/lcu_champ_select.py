from dataclasses import dataclass

from core.champion_resolver import ChampionResolver
from core.lcu_client import LcuClient


@dataclass(frozen=True)
class LcuChampSelectState:
    ally_champs: list[str]
    enemy_champs: list[str]
    banned_champs: list[str]


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

        for member in session.get("myTeam", []):
            cell_id = member.get("cellId")
            champion_id = member.get("championId")

            if cell_id is not None and champion_id and champion_id > 0:
                ally_ids[int(cell_id)] = int(champion_id)

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

        return LcuChampSelectState(
            ally_champs=self._ids_to_names(ally_ids.values()),
            enemy_champs=self._ids_to_names(enemy_ids.values()),
            banned_champs=self._ids_to_names(banned_ids),
        )

    def _ids_to_names(self, champion_ids) -> list[str]:
        names: list[str] = []

        for champion_id in champion_ids:
            name = self._champion_resolver.resolve_id(int(champion_id))

            if name and name not in names:
                names.append(name)

        return names