import base64
import os
from dataclasses import dataclass
from typing import Any

import requests
from requests import Session
from requests.exceptions import RequestException


@dataclass(frozen=True)
class LcuConnectionInfo:
    port: int
    password: str
    protocol: str


class LcuClient:
    """
    Read-only LCU client.

    Reads the League Client lockfile and calls local LCU endpoints.
    """

    DEFAULT_LOCKFILE_PATHS = [
        os.path.join("C:\\", "Riot Games", "League of Legends", "lockfile"),
        os.path.join("D:\\", "Riot Games", "League of Legends", "lockfile"),
    ]

    def __init__(self, lockfile_path: str | None = None):
        self.lockfile_path = lockfile_path
        self._session = Session()
        self._connection_info: LcuConnectionInfo | None = None

        # LCU uses a local self-signed certificate.
        self._session.verify = False

        # Suppress self-signed certificate warnings.
        requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

    def is_available(self) -> bool:
        try:
            self._ensure_connection()
            return True
        except Exception:
            return False

    def get_champ_select_session(self) -> dict[str, Any] | None:
        try:
            return self.get("/lol-champ-select/v1/session")
        except Exception:
            return None

    def get(self, endpoint: str) -> dict[str, Any]:
        self._ensure_connection()

        if self._connection_info is None:
            raise RuntimeError("LCU connection info is missing.")

        url = self._url(endpoint)

        try:
            response = self._session.get(url, timeout=1.0)
        except RequestException as exc:
            self._connection_info = None
            raise RuntimeError(f"LCU request failed: {exc}") from exc

        if response.status_code == 404:
            raise RuntimeError(f"LCU endpoint not available: {endpoint}")

        if response.status_code in {401, 403}:
            self._connection_info = None
            raise RuntimeError("LCU authentication failed.")

        response.raise_for_status()
        return response.json()

    def _ensure_connection(self) -> None:
        if self._connection_info is not None:
            return

        info = self._read_lockfile()
        self._connection_info = info

        token = base64.b64encode(f"riot:{info.password}".encode("utf-8")).decode("ascii")

        self._session.headers.update({
            "Authorization": f"Basic {token}",
            "Accept": "application/json",
        })

    def _read_lockfile(self) -> LcuConnectionInfo:
        path = self._find_lockfile_path()

        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        parts = content.split(":")

        if len(parts) != 5:
            raise RuntimeError(f"Invalid LCU lockfile format: {path}")

        _name, _pid, port, password, protocol = parts

        return LcuConnectionInfo(
            port=int(port),
            password=password,
            protocol=protocol,
        )

    def _find_lockfile_path(self) -> str:
        if self.lockfile_path and os.path.exists(self.lockfile_path):
            return self.lockfile_path

        # Prefer the standard install directory if it exists.
        for path in self.DEFAULT_LOCKFILE_PATHS:
            if path and os.path.exists(path):
                return path

        # Fallback: search common Riot locations under drives C/D.
        search_roots = [
            os.path.join("C:\\", "Riot Games"),
            os.path.join("D:\\", "Riot Games"),
        ]

        for root in search_roots:
            if not os.path.exists(root):
                continue

            for dirpath, _dirnames, filenames in os.walk(root):
                if "lockfile" in filenames:
                    return os.path.join(dirpath, "lockfile")

        raise FileNotFoundError(
            "Could not find League Client lockfile. "
            "Make sure the League client is running."
        )

    def _url(self, endpoint: str) -> str:
        if self._connection_info is None:
            raise RuntimeError("LCU connection info is missing.")

        endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"

        return (
            f"{self._connection_info.protocol}://"
            f"127.0.0.1:{self._connection_info.port}"
            f"{endpoint}"
        )
