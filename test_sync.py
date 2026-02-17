"""Test script for Notion sync and RAG query pipeline."""

import asyncio
import json
import os
import sys


def check_env() -> bool:
    """Verify required API keys are available."""
    from config import NotionConfig
    from indexer.settings import EmbeddingConfig

    missing = []
    if not NotionConfig().api_key:
        missing.append("NOTION_API_KEY")
    if not EmbeddingConfig().api_key:
        missing.append("OPENAI_API_KEY")

    if missing:
        print(f"Missing API keys: {', '.join(missing)}")
        print("Set them as environment variables before running this test:")
        for var in missing:
            print(f"  export {var}=your-key-here")
        return False
    return True


async def test_notion_client(page_id: str) -> bool:
    """Test fetching a single Notion page."""
    from notion_client import NotionClient

    print("\n--- Test: Notion Client ---")
    try:
        client = NotionClient()
        page = await client.get_page_content(page_id)
        print(f"  Title: {page.title}")
        print(f"  Page ID: {page.page_id}")
        print(f"  Last edited: {page.last_edited_time}")
        print(f"  Content length: {len(page.content)} chars")
        print(f"  Content preview: {page.content[:200]}...")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_full_sync(page_id: str) -> bool:
    """Test full sync of a single page."""
    from config import ServerConfig, SyncConfig
    from notion_sync import NotionSyncer

    print("\n--- Test: Full Sync ---")
    try:
        cfg = ServerConfig(
            sync=SyncConfig(
                index_path="test_notion_index.json",
                sync_state_path="test_sync_state.json",
            ),
        )
        syncer = NotionSyncer(cfg)
        result = await syncer.full_sync(page_ids=[page_id])
        print(f"  Pages synced: {result.pages_synced}")
        print(f"  Pages skipped: {result.pages_skipped}")
        print(f"  Chunks created: {result.chunks_created}")
        if result.errors:
            print(f"  Errors: {result.errors}")
        return result.pages_synced > 0
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_incremental_sync(page_id: str) -> bool:
    """Test incremental sync (should skip unchanged pages)."""
    from config import ServerConfig, SyncConfig
    from notion_sync import NotionSyncer

    print("\n--- Test: Incremental Sync ---")
    try:
        cfg = ServerConfig(
            sync=SyncConfig(
                index_path="test_notion_index.json",
                sync_state_path="test_sync_state.json",
            ),
        )
        syncer = NotionSyncer(cfg)
        result = await syncer.incremental_sync(page_ids=[page_id])
        print(f"  Pages synced: {result.pages_synced}")
        print(f"  Pages skipped: {result.pages_skipped}")
        print(f"  Chunks created: {result.chunks_created}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_search() -> bool:
    """Test semantic search over synced content."""
    from rag_adapter import NotionRAGAdapter

    print("\n--- Test: Search ---")
    try:
        adapter = NotionRAGAdapter(index_path="test_notion_index.json")
        results = adapter.search("main topic of this page", top_k=3)
        print(f"  Results found: {len(results)}")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] score={r.score:.4f} page={r.notion_page_id}")
            print(f"      {r.text[:100]}...")
        return len(results) > 0
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_ask() -> bool:
    """Test RAG question answering."""
    from rag_adapter import NotionRAGAdapter

    print("\n--- Test: Ask ---")
    try:
        adapter = NotionRAGAdapter(index_path="test_notion_index.json")
        result = adapter.ask(
            "What is the main topic of this content?",
            top_k=3,
            threshold=0.2,
        )
        print(f"  Answer: {result['answer'][:300]}...")
        print(f"  Chunks retrieved: {result['chunks_retrieved']}")
        print(f"  Chunks after filter: {result['chunks_after_filter']}")
        print(f"  Avg similarity: {result['avg_similarity']}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_stats() -> bool:
    """Test index statistics."""
    from rag_adapter import NotionRAGAdapter

    print("\n--- Test: Index Stats ---")
    try:
        adapter = NotionRAGAdapter(index_path="test_notion_index.json")
        stats = adapter.get_index_stats()
        print(f"  Stats: {json.dumps(stats, indent=2)}")
        return stats.get("exists", False)
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def main() -> None:
    print("=" * 60)
    print("Notion MCP + RAG Memory — Test Suite")
    print("=" * 60)

    if not check_env():
        sys.exit(1)

    # Get test page ID from args or env
    page_id = ""
    if len(sys.argv) > 1:
        page_id = sys.argv[1]
    else:
        page_id = os.environ.get("NOTION_TEST_PAGE_ID", "")

    if not page_id:
        print("\nUsage: python test_sync.py <notion-page-id>")
        print("  or:  export NOTION_TEST_PAGE_ID=your-page-id")
        sys.exit(1)

    print(f"\nTest page ID: {page_id}")

    results: dict[str, bool] = {}

    # Test 1: Notion client
    results["notion_client"] = await test_notion_client(page_id)

    # Test 2: Full sync
    results["full_sync"] = await test_full_sync(page_id)

    # Test 3: Incremental sync (should detect no changes)
    if results["full_sync"]:
        results["incremental_sync"] = await test_incremental_sync(page_id)

    # Test 4: Search
    if results["full_sync"]:
        results["search"] = test_search()

    # Test 5: Ask
    if results["full_sync"]:
        results["ask"] = test_ask()

    # Test 6: Stats
    if results["full_sync"]:
        results["stats"] = test_stats()

    # Summary
    print("\n" + "=" * 60)
    print("Results:")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n{passed}/{total} tests passed")

    # Cleanup hint
    print("\nTo clean up test files:")
    print("  rm -f test_notion_index.json test_sync_state.json")


if __name__ == "__main__":
    asyncio.run(main())
