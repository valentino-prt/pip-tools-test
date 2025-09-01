import asyncio
import logging
from typing import Callable, Optional, Awaitable, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DisconnectedError(RuntimeError):
    """Raised when the session is disconnected."""


# --- Simulated SDK -----------------------------------------------------------
class Session:
    def __init__(self, url: str, app_id: str):
        self.url = url
        self.app_id = app_id
        self._closed = True

    async def connect(self) -> "Session":
        await asyncio.sleep(0)
        self._closed = False
        return self

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
        await asyncio.sleep(0)

    async def check_connection(self) -> bool:
        # Replace with your real health-check (ping, noop, etc.)
        await asyncio.sleep(0)
        return not self._closed

    async def get_sql(self, sql: str):
        # Replace with your real SQL call
        await asyncio.sleep(0)
        return {"ok": True, "sql": sql}


# --- Minimal Client ----------------------------------------------------------
class TotoClient:
    def __init__(self, app_id: str, url: str):
        self.app_id = app_id
        self.url = url
        self.session: Optional[Session] = None
        self.reconnection = 0

    async def connect(self) -> None:
        if self.session is not None:
            return
        app_id = f"{self.app_id}_{self.reconnection}" if self.reconnection > 0 else self.app_id
        self.session = await Session(self.url, app_id).connect()
        logger.info("Connected (app_id=%s)", app_id)

    async def disconnect(self) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None
            self.reconnection = 0

    async def safe_execute(
        self,
        coro_func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs,
    ):
        """
        Safely execute a coroutine function of the session.
        - Uses session.check_connection()
        - Pops 'timeout' from kwargs (default 60s)
        - Returns None on timeout or error
        """
        if not self.session:
            await self.connect()

        assert self.session is not None

        ok = await self.session.check_connection()
        if not ok:
            raise DisconnectedError("Session not connected")

        timeout = kwargs.pop("timeout", 60)

        func_name = getattr(coro_func, "__name__", str(coro_func))
        arg_preview = args[0] if args else ""
        if isinstance(arg_preview, str) and len(arg_preview) > 80:
            arg_preview = arg_preview[:77] + "..."

        logger.debug(
            "safe_execute → %s(%s...) [timeout=%s]",
            func_name,
            arg_preview,
            timeout,
        )

        op_coro = coro_func(*args, **kwargs)
        try:
            result = await asyncio.wait_for(op_coro, timeout=timeout)
            logger.debug("safe_execute success ← %s", func_name)
            return result
        except asyncio.TimeoutError:
            logger.warning(
                "safe_execute TIMEOUT after %.1fs on %s(%s...)",
                timeout,
                func_name,
                arg_preview,
            )
            return None
        except Exception as e:
            logger.exception(
                "safe_execute ERROR on %s(%s...): %s",
                func_name,
                arg_preview,
                e,
            )
            return None

    async def execute_query(self, sql: str, timeout: Optional[float] = None):
        if not self.session:
            await self.connect()
        return await self.safe_execute(self.session.get_sql, sql, timeout=timeout)


# --- Demo --------------------------------------------------------------------
async def main():
    client = TotoClient(app_id="my-app", url="https://toto.com")
    try:
        while True:
            try:
                await client.connect()
                res = await client.execute_query("SELECT 1", timeout=2.0)
                logger.info("Result: %s", res)
                await asyncio.sleep(1)
            except DisconnectedError:
                logger.warning("Reconnecting after disconnection...")
                client.reconnection += 1
                continue
            except asyncio.TimeoutError:
                logger.warning("Query timeout, retrying...")
                continue
            except Exception:
                raise
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
