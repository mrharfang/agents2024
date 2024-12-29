from __future__ import annotations as _annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, TypeVar

import fastapi
from fastapi import Depends, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from typing_extensions import LiteralString, ParamSpec, TypedDict

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart
)

from dotenv import load_dotenv
load_dotenv()

agent = Agent('openai:gpt-4o')
THIS_DIR = Path(__file__).parent
DIST_DIR = THIS_DIR.parent / 'dist'
PUBLIC_DIR = THIS_DIR.parent / 'public'

@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    # Startup Phase: Initialize database
    async with Database.connect() as db:
        # Running Phase: Make db available to all routes
        yield {'db': db}
        # Shutdown Phase: Database connection auto-closes

app = fastapi.FastAPI(lifespan=lifespan)

@app.get('/')
async def index() -> FileResponse:
    return FileResponse((PUBLIC_DIR / 'chat_app.html'), media_type='text/html')

@app.get('/dist/{file_path:path}')
async def serve_static(file_path: str) -> FileResponse:
    return FileResponse((DIST_DIR / file_path))

async def get_db(request: Request) -> Database:
    return request.state.db 

@app.get('/chat')
async def get_chat(database: Database = Depends(get_db)) -> Response:
    msgs = await database.get_messages()
    return Response(
        b'\n'.join(json.dumps(to_chat_message(msg)).encode('utf-8') for msg in msgs),
        media_type='text/plain'
    )


class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            return {
                'role': 'user',
                'timestamp': first_part.timestamp.isoformat(),
                'content': first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                'role': 'model',
                'timestamp': m.timestamp.isoformat(),
                'content': first_part.content,
            }
    raise UnexpectedModelBehavior(f'Unexpected message type for chat app: {m}')


@app.post('/chat')
async def post_chat(
    prompt: Annotated[str, fastapi.Form()], database: Database = Depends(get_db)
) -> StreamingResponse:
    async def stream_messages():
        """Streams new line delimited JSON 'Message's to the client."""
        # stream the user prompt so that it can be displayed straight away
        yield (
            json.dumps(
                {
                    'role': 'user',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'content': prompt
                }
            ).encode('utf-8') + b'\n'
        )
        # get the chat history so far to pass as context to the agent
        messages = await database.get_messages()

        #run the agent with the user prompt and the chat history
        async with agent.run_stream( prompt, message_history=messages) as result:
            async for text in result.stream(debounce_by=0.01):
                # text here is a 'str' and the frontend wants
                # JSON encoded ModelResponse, so we create one
                m = ModelResponse.from_text(content=text, timestamp=result.timestamp())
                yield json.dumps(to_chat_message(m)).encode('utf-8') + b'\n'

        # add new messages (e.g., the user prompt and the agent response in this case) to the database
        await database.add_message(result.new_messages_json())

    return StreamingResponse(stream_messages(), media_type='text/plain')            

P = ParamSpec('P')
R = TypeVar('R')

@dataclass
class Database:
    """
    Rudimentary database to store chat messages in SQLite.

    The SQLite standard library package is synchronous, so we
    use a thread pool executor to run queries asynchronously.
    """

    con: sqlite3.Connection
    _loop: asyncio.AbstractEventLoop
    _executor: ThreadPoolExecutor

    @classmethod
    @asynccontextmanager
    async def connect(
        cls, file: Path = THIS_DIR / 'chat_app.messages.sqlite'
    ) -> AsyncIterator[Database]:
        loop = asyncio.get_event_loop()  # Remove 'with' statement
        executor = ThreadPoolExecutor(max_workers=1)
        con = await loop.run_in_executor(executor, cls._connect, str(file))
        slf = cls(con, loop, executor)
        try:
            yield slf
        finally:
            await slf._asyncify(con.close)

    @staticmethod
    def _connect(file: Path) -> sqlite3.Connection:
        con = sqlite3.connect(file)
        cur = con.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INT PRIMARY KEY,
                message_list TEXT);
        ''')
        con.commit()
        return con

    async def add_message(self, messages:bytes):
        await self._asyncify(
            self._execute,
            'INSERT INTO messages (message_list) VALUES (?);',
            messages,
            commit=True,
        )
        await self._asyncify(self.con.commit)

    async def get_messages(self) -> list[ModelMessage]:
        c = await self._asyncify(
            self._execute,
            'SELECT message_list FROM messages ORDER BY id DESC;'
        )
        rows = await self._asyncify(c.fetchall)
        messages: list[ModelMessage] = []
        for row in rows:
            messages.extend(ModelMessagesTypeAdapter.validate_json(row[0]))
        return messages

    def _execute(
        self, sql: LiteralString, *args: Any, commit: bool = False
    ) -> sqlite3.Cursor:
        cur = self.con.cursor()
        cur.execute(sql, args)
        if commit:
            self.con.commit()  # Fix the missing dot
        return cur

    async def _asyncify(
        self,
        func: Callable[P, R],  # func that takes parameters P and returns type R
        *args: P.args,         # args matching the function's parameter types
        **kwargs: P.kwargs     # kwargs matching the function's parameter types
    ) -> R:                    # method returns whatever type the original function returns
        return await self._loop.run_in_executor(
            self._executor, 
            partial(func, **kwargs),
            *args
        )



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'chat_app:app',  # Use import string instead of app instance
        host='0.0.0.0',
        port=8000,
        reload=True,
        reload_dirs=[str(THIS_DIR)]
    )