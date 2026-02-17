"""Notion API client for fetching pages, databases, and block content."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from config import NotionConfig

logger = logging.getLogger(__name__)


@dataclass
class NotionPage:
    """Represents a Notion page with extracted content."""
    page_id: str
    title: str
    content: str
    last_edited_time: str
    url: str = ""


class NotionClient:
    """Async client for the Notion API."""

    def __init__(self, config: NotionConfig | None = None):
        self.config = config or NotionConfig()

        if not self.config.api_key:
            raise ValueError(
                "Notion API key not found. Set NOTION_API_KEY environment variable."
            )

        self._headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Notion-Version": self.config.api_version,
            "Content-Type": "application/json",
        }

    def _create_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=self._headers,
            timeout=self.config.request_timeout,
        )

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an API request with retry logic for transient errors."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = await client.request(method, url, **kwargs)

                if response.status_code == 200:
                    return response.json()

                if response.status_code == 401:
                    raise PermissionError(
                        "Notion API authentication failed. Check your NOTION_API_KEY."
                    )

                if response.status_code == 403:
                    raise PermissionError(
                        f"Access denied. Ensure the integration has access to the resource. "
                        f"URL: {url}"
                    )

                if response.status_code == 404:
                    raise FileNotFoundError(
                        f"Notion resource not found: {url}. "
                        f"Check the page/database ID and integration permissions."
                    )

                if response.status_code == 429:
                    retry_after = float(
                        response.headers.get("Retry-After", self.config.retry_delay)
                    )
                    logger.warning(
                        "Rate limited by Notion API, waiting %.1fs (attempt %d/%d)",
                        retry_after, attempt + 1, self.config.max_retries,
                    )
                    await asyncio.sleep(retry_after)
                    last_error = Exception(f"Rate limited: {response.status_code}")
                    continue

                if response.status_code >= 500:
                    last_error = Exception(
                        f"Notion server error: {response.status_code}"
                    )
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        "Notion server error %d, retrying in %.1fs (attempt %d/%d)",
                        response.status_code, wait_time,
                        attempt + 1, self.config.max_retries,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                raise Exception(
                    f"Notion API error {response.status_code}: {response.text}"
                )

            except httpx.ConnectError as e:
                last_error = e
                wait_time = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    "Connection error, retrying in %.1fs (attempt %d/%d): %s",
                    wait_time, attempt + 1, self.config.max_retries, e,
                )
                await asyncio.sleep(wait_time)

            except httpx.TimeoutException as e:
                last_error = e
                wait_time = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    "Timeout, retrying in %.1fs (attempt %d/%d): %s",
                    wait_time, attempt + 1, self.config.max_retries, e,
                )
                await asyncio.sleep(wait_time)

        raise RuntimeError(
            f"Notion API request failed after {self.config.max_retries} attempts: "
            f"{last_error}"
        )

    async def get_page(self, page_id: str) -> dict[str, Any]:
        """Fetch page metadata from Notion."""
        async with self._create_client() as client:
            return await self._request_with_retry(client, "GET", f"/pages/{page_id}")

    async def get_block_children(
        self, block_id: str
    ) -> list[dict[str, Any]]:
        """Fetch all child blocks of a block/page, handling pagination."""
        blocks: list[dict[str, Any]] = []
        start_cursor = None

        async with self._create_client() as client:
            while True:
                params: dict[str, Any] = {"page_size": 100}
                if start_cursor:
                    params["start_cursor"] = start_cursor

                data = await self._request_with_retry(
                    client, "GET", f"/blocks/{block_id}/children", params=params
                )

                blocks.extend(data.get("results", []))

                if not data.get("has_more", False):
                    break
                start_cursor = data.get("next_cursor")

        return blocks

    async def query_database(
        self, database_id: str, filter_obj: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Query a Notion database, handling pagination."""
        pages: list[dict[str, Any]] = []
        start_cursor = None

        async with self._create_client() as client:
            while True:
                body: dict[str, Any] = {"page_size": 100}
                if start_cursor:
                    body["start_cursor"] = start_cursor
                if filter_obj:
                    body["filter"] = filter_obj

                data = await self._request_with_retry(
                    client, "POST", f"/databases/{database_id}/query", json=body
                )

                pages.extend(data.get("results", []))

                if not data.get("has_more", False):
                    break
                start_cursor = data.get("next_cursor")

        return pages

    def extract_title(self, page_data: dict[str, Any]) -> str:
        """Extract the title from page metadata."""
        properties = page_data.get("properties", {})

        # Check common title property names
        for prop_name in ("title", "Title", "Name", "name"):
            prop = properties.get(prop_name, {})
            if prop.get("type") == "title":
                title_items = prop.get("title", [])
                if title_items:
                    return "".join(
                        item.get("plain_text", "") for item in title_items
                    )

        # Fallback: search all properties for a title type
        for prop in properties.values():
            if prop.get("type") == "title":
                title_items = prop.get("title", [])
                if title_items:
                    return "".join(
                        item.get("plain_text", "") for item in title_items
                    )

        return "Untitled"

    def extract_block_text(self, block: dict[str, Any]) -> str:
        """Extract plain text from a single Notion block."""
        block_type = block.get("type", "")
        block_data = block.get(block_type, {})

        # Blocks with rich_text arrays
        rich_text_types = {
            "paragraph", "heading_1", "heading_2", "heading_3",
            "bulleted_list_item", "numbered_list_item", "toggle",
            "quote", "callout",
        }

        if block_type in rich_text_types:
            rich_text = block_data.get("rich_text", [])
            text = "".join(item.get("plain_text", "") for item in rich_text)

            if block_type == "heading_1":
                return f"# {text}"
            elif block_type == "heading_2":
                return f"## {text}"
            elif block_type == "heading_3":
                return f"### {text}"
            elif block_type == "bulleted_list_item":
                return f"- {text}"
            elif block_type == "numbered_list_item":
                return f"* {text}"
            elif block_type == "quote":
                return f"> {text}"
            return text

        if block_type == "code":
            rich_text = block_data.get("rich_text", [])
            code = "".join(item.get("plain_text", "") for item in rich_text)
            language = block_data.get("language", "")
            return f"```{language}\n{code}\n```"

        if block_type == "to_do":
            rich_text = block_data.get("rich_text", [])
            text = "".join(item.get("plain_text", "") for item in rich_text)
            checked = block_data.get("checked", False)
            marker = "[x]" if checked else "[ ]"
            return f"- {marker} {text}"

        if block_type == "divider":
            return "---"

        if block_type == "table_row":
            cells = block_data.get("cells", [])
            row_texts = []
            for cell in cells:
                cell_text = "".join(
                    item.get("plain_text", "") for item in cell
                )
                row_texts.append(cell_text)
            return " | ".join(row_texts)

        if block_type == "equation":
            return block_data.get("expression", "")

        # Skip unsupported block types (image, file, embed, etc.)
        return ""

    async def get_page_content(self, page_id: str) -> NotionPage:
        """
        Fetch a complete Notion page: metadata + all block content as plain text.

        Args:
            page_id: The Notion page ID (with or without dashes)

        Returns:
            NotionPage with extracted title, content, and metadata
        """
        clean_id = page_id.replace("-", "")

        # Fetch page metadata and blocks concurrently
        page_data, blocks = await asyncio.gather(
            self.get_page(clean_id),
            self.get_block_children(clean_id),
        )

        title = self.extract_title(page_data)
        last_edited = page_data.get("last_edited_time", "")
        url = page_data.get("url", "")

        # Convert blocks to text
        text_parts: list[str] = []
        for block in blocks:
            text = self.extract_block_text(block)
            if text:
                text_parts.append(text)

            # Fetch nested children (one level deep)
            if block.get("has_children", False):
                children = await self.get_block_children(block["id"])
                for child in children:
                    child_text = self.extract_block_text(child)
                    if child_text:
                        text_parts.append(f"  {child_text}")

        content = "\n\n".join(text_parts)

        # Prepend title
        if title and title != "Untitled":
            content = f"# {title}\n\n{content}"

        return NotionPage(
            page_id=clean_id,
            title=title,
            content=content,
            last_edited_time=last_edited,
            url=url,
        )

    async def get_database_pages(
        self, database_id: str
    ) -> list[NotionPage]:
        """
        Fetch all pages from a Notion database.

        Returns list of NotionPage objects with content.
        """
        clean_id = database_id.replace("-", "")
        entries = await self.query_database(clean_id)

        pages: list[NotionPage] = []
        for entry in entries:
            entry_id = entry["id"].replace("-", "")
            try:
                page = await self.get_page_content(entry_id)
                pages.append(page)
            except Exception as e:
                logger.warning("Failed to fetch page %s: %s", entry_id, e)

        return pages
