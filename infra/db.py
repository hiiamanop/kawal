from collections.abc import Callable
from typing import TypeVar

import psycopg
from psycopg import Connection


T = TypeVar("T")


class TransactionRunner:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def run(
        self,
        operation: Callable[[Connection], T],
        after_commit: Callable[[], None] | None = None,
    ) -> T:
        with psycopg.connect(self._dsn) as connection:
            with connection.transaction():
                result = operation(connection)
        if after_commit is not None:
            after_commit()
        return result
