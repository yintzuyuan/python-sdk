import pytest
from pydantic import AnyUrl, BaseModel

from mcp.server.fastmcp.resources import FunctionResource


class TestFunctionResource:
    """Test FunctionResource functionality."""

    def test_function_resource_creation(self):
        """Test creating a FunctionResource."""

        def my_func() -> str:
            return "test content"

        resource = FunctionResource(
            uri=AnyUrl("fn://test"),
            name="test",
            description="test function",
            fn=my_func,
        )
        assert str(resource.uri) == "fn://test"
        assert resource.name == "test"
        assert resource.description == "test function"
        assert resource.mime_type == "text/plain"  # default
        assert resource.fn == my_func

    @pytest.mark.anyio
    async def test_read_text(self):
        """Test reading text from a FunctionResource."""

        def get_data() -> str:
            return "Hello, world!"

        resource = FunctionResource(
            uri=AnyUrl("function://test"),
            name="test",
            fn=get_data,
        )
        content = await resource.read()
        assert content == "Hello, world!"
        assert resource.mime_type == "text/plain"

    @pytest.mark.anyio
    async def test_read_binary(self):
        """Test reading binary data from a FunctionResource."""

        def get_data() -> bytes:
            return b"Hello, world!"

        resource = FunctionResource(
            uri=AnyUrl("function://test"),
            name="test",
            fn=get_data,
        )
        content = await resource.read()
        assert content == b"Hello, world!"

    @pytest.mark.anyio
    async def test_json_conversion(self):
        """Test automatic JSON conversion of non-string results."""

        def get_data() -> dict:
            return {"key": "value"}

        resource = FunctionResource(
            uri=AnyUrl("function://test"),
            name="test",
            fn=get_data,
        )
        content = await resource.read()
        assert isinstance(content, str)
        assert '"key": "value"' in content

    @pytest.mark.anyio
    async def test_error_handling(self):
        """Test error handling in FunctionResource."""

        def failing_func() -> str:
            raise ValueError("Test error")

        resource = FunctionResource(
            uri=AnyUrl("function://test"),
            name="test",
            fn=failing_func,
        )
        with pytest.raises(ValueError, match="Error reading resource function://test"):
            await resource.read()

    @pytest.mark.anyio
    async def test_basemodel_conversion(self):
        """Test handling of BaseModel types."""

        class MyModel(BaseModel):
            name: str

        resource = FunctionResource(
            uri=AnyUrl("function://test"),
            name="test",
            fn=lambda: MyModel(name="test"),
        )
        content = await resource.read()
        assert content == '{\n  "name": "test"\n}'

    @pytest.mark.anyio
    async def test_custom_type_conversion(self):
        """Test handling of custom types."""

        class CustomData:
            def __str__(self) -> str:
                return "custom data"

        def get_data() -> CustomData:
            return CustomData()

        resource = FunctionResource(
            uri=AnyUrl("function://test"),
            name="test",
            fn=get_data,
        )
        content = await resource.read()
        assert isinstance(content, str)

    @pytest.mark.anyio
    async def test_async_read_text(self):
        """Test reading text from async FunctionResource."""

        async def get_data() -> str:
            return "Hello, world!"

        resource = FunctionResource(
            uri=AnyUrl("function://test"),
            name="test",
            fn=get_data,
        )
        content = await resource.read()
        assert content == "Hello, world!"
        assert resource.mime_type == "text/plain"

    @pytest.mark.anyio
    async def test_from_function(self):
        """Test creating a FunctionResource from a function."""

        async def get_data() -> str:
            """get_data returns a string"""
            return "Hello, world!"

        resource = FunctionResource.from_function(
            fn=get_data,
            uri="function://test",
            name="test",
        )

        assert resource.description == "get_data returns a string"
        assert resource.mime_type == "text/plain"
        assert resource.name == "test"
        assert resource.uri == AnyUrl("function://test")
