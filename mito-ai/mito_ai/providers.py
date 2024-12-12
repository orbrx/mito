import os
import traceback
from typing import AsyncGenerator, Optional, Union

from openai import AsyncOpenAI
from openai._streaming import AsyncStream
from openai.types.chat import ChatCompletionChunk
from traitlets import Unicode, default
from traitlets.config import LoggingConfigurable

from .models import (
    CompletionError,
    CompletionItem,
    CompletionItemError,
    CompletionReply,
    CompletionRequest,
    CompletionStreamChunk,
)

__all__ = ["OpenAIProvider"]


class OpenAIProvider(LoggingConfigurable):
    """Provide AI feature through OpenAI services."""

    # Internally it uses jinja2 template to render prompt messages.

    api_key = Unicode(
        config=True,
        help="OpenAI API key. Default value is read from the OPENAI_API_KEY environment variable.",
    )

    # FIXME add validate function to check if the model is valid
    model = Unicode(
        "gpt-4o-mini", config=True, help="OpenAI model to use for completions"
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client: Optional[AsyncOpenAI] = None

    @default("api_key")
    def _api_key_default(self):
        return os.environ.get("OPENAI_API_KEY", "")

    @property
    def can_stream(self) -> bool:
        """Whether the provider supports streaming completions.

        Streaming is only supported if an OpenAI API key is provided.
        """
        return bool(self.api_key)

    @property
    def client(self) -> AsyncOpenAI:
        """Get the asynchronous OpenAI client."""
        if not self._client or self._client.is_closed():
            self._client = AsyncOpenAI(api_key=self.api_key)

        return self._client

    async def _check_authentication(self) -> None:
        # # TODO implement this
        # async for _ in client.models.list():
        #     logging.getLogger("ServerApp").info("%s", _)
        #     break
        ...

    async def request_completions(self, request: CompletionRequest) -> CompletionReply:
        """Get a completion from the OpenAI API.

        Args:
            request: The completion request description.
        Returns:
            The completion
        """
        completion = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=100,
            messages=request.messages,
        )

        if len(completion.choices) == 0:
            return CompletionReply(
                items=[],
                parent_id=request.message_id,
                error=CompletionError(
                    type="NoCompletion",
                    title="No completion returned from the OpenAI API.",
                    traceback="",
                ),
            )

        else:
            try:
                item = CompletionItem(
                    insertText=completion.choices[0].message.content or "",
                    isIncomplete=False,
                )
                return CompletionReply(
                    items=[item],
                    parent_id=request.message_id,
                )
            except BaseException as e:
                return CompletionReply(
                    items=[],
                    parent_id=request.message_id,
                    error=CompletionError(
                        type=e.__class__.__name__,
                        title=e.args[0] if e.args else "Exception",
                        traceback=traceback.format_exc(),
                    ),
                )

    async def stream_completions(
        self, request: CompletionRequest
    ) -> AsyncGenerator[Union[CompletionReply, CompletionStreamChunk], None]:
        """Stream completions from the OpenAI API.

        Args:
            request: The completion request description.
        Returns:
            An async generator yielding first an acknowledge completion reply without
            completion and then completion chunks from the third-party provider.
        """
        # The streaming completion has two steps:
        # Step 1: Acknowledge the request
        # Step 2: Stream the completion chunks coming from the OpenAI API

        # Acknowledge the request
        yield CompletionReply(
            items=[
                CompletionItem(
                    insertText="", isIncomplete=True, token=request.message_id
                )
            ],
            parent_id=request.message_id,
        )

        # Send the completion request to the OpenAI API and returns a stream of completion chunks
        stream: AsyncStream[
            ChatCompletionChunk
        ] = await self.client.chat.completions.create(
            model=self.model,
            stream=True,
            max_tokens=100,
            messages=request.messages,
        )
        async for chunk in stream:
            try:
                is_finished = chunk.choices[0].finish_reason is not None
                yield CompletionStreamChunk(
                    parent_id=request.message_id,
                    chunk=CompletionItem(
                        insertText=chunk.choices[0].delta.content or "",
                        isIncomplete=True,
                        token=request.message_id,
                    ),
                    done=is_finished,
                )
            except BaseException as e:
                yield CompletionStreamChunk(
                    parent_id=request.message_id,
                    chunk=CompletionItem(
                        insertText="",
                        isIncomplete=True,
                        error=CompletionItemError(
                            message=f"Failed to parse chunk completion: {e!r}"
                        ),
                        token=request.message_id,
                    ),
                    done=True,
                    error=CompletionError(
                        type=e.__class__.__name__,
                        title=e.args[0] if e.args else "Exception",
                        traceback=traceback.format_exc(),
                    ),
                )
                break
