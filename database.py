# from contextlib import asynccontextmanager
# import asyncio
# import os
# from typing import Optional

# from asyncpg import pool
# from dotenv import load_dotenv

# from logger import logger

# load_dotenv()

# class AsyncDatabaseConnection():
#     def __init__(self, min_conn=1, max_conn=10):
#         self._min_conn = min_conn
#         self._max_conn = max_conn
#         self.pool: Optional[pool.Pool] = None
#         self.connection_string = (
#             f'postgresql://{os.getenv("DB_USERNAME")}:'
#             f'{os.getenv("DB_PWORD")}@'
#             f'{os.getenv("DB_URL")}{os.getenv("DB_NAME")}?'
#             f'sslmode={os.getenv("DB_SSL")}'        
#         )

#     async def create_connection_pool(self):
#         self.pool = await pool.create_pool(
#             min_size=self._min_conn,
#             max_size=self._max_conn,
#             dsn=self.connection_string
#         )
#         logger.info("Connection pool created")        

#     @asynccontextmanager
#     async def connection_from_pool(self):
#         if not self.pool:
#             logger.error("Connection pool not initialised")
#             logger.info("Create the connection pool")
#             await self.create_connection_pool()
#         assert self.pool is not None
#         async with self.pool.acquire() as conn:
#             yield conn
    
#     async def close_all(self):
#         if self.pool:
#             await self.pool.close()
#             logger.info("All async database connections closed")
        
#     def _sync_close_all(self):
#         """A synchronous wrapper for graceful shutdown when the app exits."""
#         # asyncio.run() cannot be called from a running event loop
#         try:
#             loop = asyncio.get_running_loop()
#             # Use a background task or warn
#             logger.warning("Cannot close pool from running event loop. Please call close_all() explicitly.")
#         except RuntimeError:
#             asyncio.run(self.close_all())


# if __name__ == "__main__":
#     async def main():
#         async_db = AsyncDatabaseConnection()
#         await async_db.create_connection_pool()

#     asyncio.run(main())
