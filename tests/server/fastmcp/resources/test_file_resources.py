import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from pydantic import FileUrl

from mcp.server.fastmcp.resources import FileResource


@pytest.fixture
def temp_file():
    """Create a temporary file for testing.

    File is automatically cleaned up after the test if it still exists.
    """
    content = "test content"
    with NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(content)
        path = Path(f.name).resolve()
    yield path
    try:
        path.unlink()
    except FileNotFoundError:
        pass  # File was already deleted by the test


class TestFileResource:
    """Test FileResource functionality."""

    def test_file_resource_creation(self, temp_file: Path):
        """Test creating a FileResource."""
        resource = FileResource(
            uri=FileUrl(temp_file.as_uri()),
            name="test",
            description="test file",
            path=temp_file,
        )
        assert str(resource.uri) == temp_file.as_uri()
        assert resource.name == "test"
        assert resource.description == "test file"
        assert resource.mime_type == "text/plain"  # default
        assert resource.path == temp_file
        assert resource.is_binary is False  # default

    def test_file_resource_str_path_conversion(self, temp_file: Path):
        """Test FileResource handles string paths."""
        resource = FileResource(
            uri=FileUrl(f"file://{temp_file}"),
            name="test",
            path=Path(str(temp_file)),
        )
        assert isinstance(resource.path, Path)
        assert resource.path.is_absolute()

    @pytest.mark.anyio
    async def test_read_text_file(self, temp_file: Path):
        """Test reading a text file."""
        resource = FileResource(
            uri=FileUrl(f"file://{temp_file}"),
            name="test",
            path=temp_file,
        )
        content = await resource.read()
        assert content == "test content"
        assert resource.mime_type == "text/plain"

    @pytest.mark.anyio
    async def test_read_binary_file(self, temp_file: Path):
        """Test reading a file as binary."""
        resource = FileResource(
            uri=FileUrl(f"file://{temp_file}"),
            name="test",
            path=temp_file,
            is_binary=True,
        )
        content = await resource.read()
        assert isinstance(content, bytes)
        assert content == b"test content"

    def test_relative_path_error(self):
        """Test error on relative path."""
        with pytest.raises(ValueError, match="Path must be absolute"):
            FileResource(
                uri=FileUrl("file:///test.txt"),
                name="test",
                path=Path("test.txt"),
            )

    @pytest.mark.anyio
    async def test_missing_file_error(self, temp_file: Path):
        """Test error when file doesn't exist."""
        # Create path to non-existent file
        missing = temp_file.parent / "missing.txt"
        resource = FileResource(
            uri=FileUrl("file:///missing.txt"),
            name="test",
            path=missing,
        )
        with pytest.raises(ValueError, match="Error reading file"):
            await resource.read()

    @pytest.mark.skipif(os.name == "nt", reason="File permissions behave differently on Windows")
    @pytest.mark.anyio
    async def test_permission_error(self, temp_file: Path):
        """Test reading a file without permissions."""
        temp_file.chmod(0o000)  # Remove all permissions
        try:
            resource = FileResource(
                uri=FileUrl(temp_file.as_uri()),
                name="test",
                path=temp_file,
            )
            with pytest.raises(ValueError, match="Error reading file"):
                await resource.read()
        finally:
            temp_file.chmod(0o644)  # Restore permissions
