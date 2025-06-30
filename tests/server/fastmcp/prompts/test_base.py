import pytest
from pydantic import FileUrl

from mcp.server.fastmcp.prompts.base import (
    AssistantMessage,
    Message,
    Prompt,
    TextContent,
    UserMessage,
)
from mcp.types import EmbeddedResource, TextResourceContents


class TestRenderPrompt:
    @pytest.mark.anyio
    async def test_basic_fn(self):
        def fn() -> str:
            return "Hello, world!"

        prompt = Prompt.from_function(fn)
        assert await prompt.render() == [UserMessage(content=TextContent(type="text", text="Hello, world!"))]

    @pytest.mark.anyio
    async def test_async_fn(self):
        async def fn() -> str:
            return "Hello, world!"

        prompt = Prompt.from_function(fn)
        assert await prompt.render() == [UserMessage(content=TextContent(type="text", text="Hello, world!"))]

    @pytest.mark.anyio
    async def test_fn_with_args(self):
        async def fn(name: str, age: int = 30) -> str:
            return f"Hello, {name}! You're {age} years old."

        prompt = Prompt.from_function(fn)
        assert await prompt.render(arguments={"name": "World"}) == [
            UserMessage(content=TextContent(type="text", text="Hello, World! You're 30 years old."))
        ]

    @pytest.mark.anyio
    async def test_fn_with_invalid_kwargs(self):
        async def fn(name: str, age: int = 30) -> str:
            return f"Hello, {name}! You're {age} years old."

        prompt = Prompt.from_function(fn)
        with pytest.raises(ValueError):
            await prompt.render(arguments={"age": 40})

    @pytest.mark.anyio
    async def test_fn_returns_message(self):
        async def fn() -> UserMessage:
            return UserMessage(content="Hello, world!")

        prompt = Prompt.from_function(fn)
        assert await prompt.render() == [UserMessage(content=TextContent(type="text", text="Hello, world!"))]

    @pytest.mark.anyio
    async def test_fn_returns_assistant_message(self):
        async def fn() -> AssistantMessage:
            return AssistantMessage(content=TextContent(type="text", text="Hello, world!"))

        prompt = Prompt.from_function(fn)
        assert await prompt.render() == [AssistantMessage(content=TextContent(type="text", text="Hello, world!"))]

    @pytest.mark.anyio
    async def test_fn_returns_multiple_messages(self):
        expected = [
            UserMessage("Hello, world!"),
            AssistantMessage("How can I help you today?"),
            UserMessage("I'm looking for a restaurant in the center of town."),
        ]

        async def fn() -> list[Message]:
            return expected

        prompt = Prompt.from_function(fn)
        assert await prompt.render() == expected

    @pytest.mark.anyio
    async def test_fn_returns_list_of_strings(self):
        expected = [
            "Hello, world!",
            "I'm looking for a restaurant in the center of town.",
        ]

        async def fn() -> list[str]:
            return expected

        prompt = Prompt.from_function(fn)
        assert await prompt.render() == [UserMessage(t) for t in expected]

    @pytest.mark.anyio
    async def test_fn_returns_resource_content(self):
        """Test returning a message with resource content."""

        async def fn() -> UserMessage:
            return UserMessage(
                content=EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=FileUrl("file://file.txt"),
                        text="File contents",
                        mimeType="text/plain",
                    ),
                )
            )

        prompt = Prompt.from_function(fn)
        assert await prompt.render() == [
            UserMessage(
                content=EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=FileUrl("file://file.txt"),
                        text="File contents",
                        mimeType="text/plain",
                    ),
                )
            )
        ]

    @pytest.mark.anyio
    async def test_fn_returns_mixed_content(self):
        """Test returning messages with mixed content types."""

        async def fn() -> list[Message]:
            return [
                UserMessage(content="Please analyze this file:"),
                UserMessage(
                    content=EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri=FileUrl("file://file.txt"),
                            text="File contents",
                            mimeType="text/plain",
                        ),
                    )
                ),
                AssistantMessage(content="I'll help analyze that file."),
            ]

        prompt = Prompt.from_function(fn)
        assert await prompt.render() == [
            UserMessage(content=TextContent(type="text", text="Please analyze this file:")),
            UserMessage(
                content=EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=FileUrl("file://file.txt"),
                        text="File contents",
                        mimeType="text/plain",
                    ),
                )
            ),
            AssistantMessage(content=TextContent(type="text", text="I'll help analyze that file.")),
        ]

    @pytest.mark.anyio
    async def test_fn_returns_dict_with_resource(self):
        """Test returning a dict with resource content."""

        async def fn() -> dict:
            return {
                "role": "user",
                "content": {
                    "type": "resource",
                    "resource": {
                        "uri": FileUrl("file://file.txt"),
                        "text": "File contents",
                        "mimeType": "text/plain",
                    },
                },
            }

        prompt = Prompt.from_function(fn)
        assert await prompt.render() == [
            UserMessage(
                content=EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=FileUrl("file://file.txt"),
                        text="File contents",
                        mimeType="text/plain",
                    ),
                )
            )
        ]
