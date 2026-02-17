"""Sync logic for pulling Notion content into the local RAG index."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import ServerConfig
from notion_client import NotionClient, NotionPage
from indexer.chunker import TextChunker
from indexer.embeddings import OpenAIEmbeddings, EmbeddingProvider
from indexer.index import DocumentIndex
from indexer.settings import ChunkingConfig, EmbeddingConfig, IndexConfig

logger = logging.getLogger(__name__)


@dataclass
class SyncState:
    """Tracks what has been synced and when."""
    pages: dict[str, str] = field(default_factory=dict)  # page_id -> last_edited_time

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"pages": self.pages}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SyncState":
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        state = cls()
        state.pages = data.get("pages", {})
        return state


@dataclass
class SyncResult:
    """Result of a sync operation."""
    pages_synced: int
    pages_skipped: int
    chunks_created: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages_synced": self.pages_synced,
            "pages_skipped": self.pages_skipped,
            "chunks_created": self.chunks_created,
            "errors": self.errors,
        }


class NotionSyncer:
    """Orchestrates syncing Notion content into the RAG index."""

    def __init__(
        self,
        config: ServerConfig,
        notion_client: NotionClient | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.config = config
        self.client = notion_client or NotionClient(config.notion)
        self.chunker = TextChunker(ChunkingConfig())
        self.embedder = embedding_provider or OpenAIEmbeddings(EmbeddingConfig())

    def _make_document_id(self, page_id: str) -> str:
        return f"notion_{page_id}"

    def _make_source(self, page_id: str) -> str:
        return f"notion://{page_id}"

    def _load_or_create_index(self) -> DocumentIndex:
        """Load existing index or create a new one."""
        path = Path(self.config.sync.index_path)
        if path.exists():
            return DocumentIndex.load(str(path))
        index = DocumentIndex(IndexConfig(output_path=str(path)))
        index.set_embedding_model(self.embedder.model_name)
        return index

    def _index_page(
        self,
        page: NotionPage,
        index: DocumentIndex,
    ) -> int:
        """Chunk, embed, and add a single page to the index."""
        doc_id = self._make_document_id(page.page_id)
        source = self._make_source(page.page_id)

        if not page.content.strip():
            logger.warning("Page %s (%s) has no content, skipping", page.page_id, page.title)
            return 0

        chunks = self.chunker.chunk_text(page.content, doc_id)
        if not chunks:
            return 0

        embeddings = self.embedder.embed_texts([c.text for c in chunks])
        index.add_document(
            document_id=doc_id,
            source=source,
            chunks=chunks,
            embeddings=embeddings,
        )

        return len(chunks)

    async def _fetch_all_pages(
        self,
        page_ids: list[str] | None = None,
        database_ids: list[str] | None = None,
    ) -> list[NotionPage]:
        """Fetch pages from specified IDs and databases."""
        pages: list[NotionPage] = []

        target_page_ids = page_ids or self.config.sync.page_ids
        target_db_ids = database_ids or self.config.sync.database_ids

        # Fetch individual pages
        for pid in target_page_ids:
            try:
                page = await self.client.get_page_content(pid)
                pages.append(page)
            except Exception as e:
                logger.error("Failed to fetch page %s: %s", pid, e)

        # Fetch pages from databases
        for db_id in target_db_ids:
            try:
                db_pages = await self.client.get_database_pages(db_id)
                pages.extend(db_pages)
            except Exception as e:
                logger.error("Failed to fetch database %s: %s", db_id, e)

        return pages

    async def full_sync(
        self,
        page_ids: list[str] | None = None,
        database_ids: list[str] | None = None,
    ) -> SyncResult:
        """
        Full sync: pull all pages and rebuild their index entries.

        Args:
            page_ids: Specific page IDs to sync (overrides config)
            database_ids: Specific database IDs to sync (overrides config)

        Returns:
            SyncResult with statistics
        """
        pages = await self._fetch_all_pages(page_ids, database_ids)

        if not pages:
            return SyncResult(
                pages_synced=0, pages_skipped=0, chunks_created=0,
                errors=["No pages found to sync"],
            )

        index = self._load_or_create_index()
        state = SyncState()
        result = SyncResult(pages_synced=0, pages_skipped=0, chunks_created=0)

        for page in pages:
            try:
                chunks_count = self._index_page(page, index)
                if chunks_count > 0:
                    result.pages_synced += 1
                    result.chunks_created += chunks_count
                    state.pages[page.page_id] = page.last_edited_time
                else:
                    result.pages_skipped += 1
            except Exception as e:
                error_msg = f"Error indexing page {page.page_id} ({page.title}): {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # Save index and sync state
        index.save(self.config.sync.index_path)
        state.save(self.config.sync.sync_state_path)

        logger.info(
            "Full sync complete: %d pages synced, %d chunks created",
            result.pages_synced, result.chunks_created,
        )

        return result

    async def incremental_sync(
        self,
        page_ids: list[str] | None = None,
        database_ids: list[str] | None = None,
    ) -> SyncResult:
        """
        Incremental sync: only re-index pages that changed since last sync.

        Compares last_edited_time from Notion with stored sync state.

        Args:
            page_ids: Specific page IDs to sync (overrides config)
            database_ids: Specific database IDs to sync (overrides config)

        Returns:
            SyncResult with statistics
        """
        state = SyncState.load(self.config.sync.sync_state_path)
        pages = await self._fetch_all_pages(page_ids, database_ids)

        if not pages:
            return SyncResult(
                pages_synced=0, pages_skipped=0, chunks_created=0,
                errors=["No pages found to sync"],
            )

        # Filter to only changed pages
        changed_pages: list[NotionPage] = []
        skipped = 0
        for page in pages:
            last_known = state.pages.get(page.page_id, "")
            if page.last_edited_time != last_known:
                changed_pages.append(page)
            else:
                skipped += 1

        if not changed_pages:
            logger.info("Incremental sync: no changes detected, %d pages up to date", skipped)
            return SyncResult(
                pages_synced=0, pages_skipped=skipped, chunks_created=0,
            )

        index = self._load_or_create_index()
        result = SyncResult(pages_synced=0, pages_skipped=skipped, chunks_created=0)

        for page in changed_pages:
            try:
                chunks_count = self._index_page(page, index)
                if chunks_count > 0:
                    result.pages_synced += 1
                    result.chunks_created += chunks_count
                    state.pages[page.page_id] = page.last_edited_time
                else:
                    result.pages_skipped += 1
            except Exception as e:
                error_msg = f"Error indexing page {page.page_id} ({page.title}): {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # Save updated index and state
        index.save(self.config.sync.index_path)
        state.save(self.config.sync.sync_state_path)

        logger.info(
            "Incremental sync complete: %d pages synced, %d skipped, %d chunks created",
            result.pages_synced, result.pages_skipped, result.chunks_created,
        )

        return result
