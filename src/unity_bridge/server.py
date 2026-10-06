"""Bounded newline-JSON protocol over a loopback TCP connection."""

from __future__ import annotations

import json
import socket
import socketserver
from typing import Any, Callable

from .session import CaveSession

MAX_PACKET = 4096


class BridgeHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: BridgeServer = self.server  # type: ignore[assignment]
        session = server.session_factory()
        self.connection.settimeout(15)
        try:
            while True:
                packet = self.rfile.readline(MAX_PACKET + 1)
                if not packet:
                    return
                if len(packet) > MAX_PACKET or not packet.endswith(b"\n"):
                    self.reply({"protocol": 1, "error": "request exceeds packet limit"})
                    return
                try:
                    response = session.handle(json.loads(packet))
                except (ValueError, TypeError, UnicodeDecodeError) as error:
                    response = {"protocol": 1, "error": str(error)}
                self.reply(response)
        except (ConnectionError, OSError):
            return
        finally:
            session.game.close()

    def reply(self, response: dict[str, Any]) -> None:
        self.wfile.write(
            (json.dumps(response, separators=(",", ":"), allow_nan=False) + "\n").encode()
        )
        self.wfile.flush()


class BridgeServer(socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(
        self, address: tuple[str, int], session_factory: Callable[[], CaveSession]
    ) -> None:
        if address[0] != "127.0.0.1":
            raise ValueError("the Unity pilot bridge binds only to 127.0.0.1")
        self.session_factory = session_factory
        super().__init__(address, BridgeHandler)

    def get_request(self) -> tuple[socket.socket, Any]:
        connection, address = super().get_request()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return connection, address
