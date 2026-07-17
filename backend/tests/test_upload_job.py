import asyncio
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.upload_service import UploadJobStore


class UploadJobStoreTests(unittest.TestCase):
    def test_create_update_cancel_snapshot(self):
        store = UploadJobStore()
        job = store.create("file-1", filename="a.wav")
        self.assertEqual(job.status, "queued")

        store.update("file-1", status="recognizing", progress=0.5)
        snapshot = store.snapshot("file-1")
        self.assertEqual(snapshot["status"], "recognizing")
        self.assertEqual(snapshot["progress"], 0.5)

        cancelled = store.cancel("file-1")
        self.assertTrue(cancelled.cancel_event.is_set())

    def test_subscriber_receives_published_event(self):
        async def run():
            store = UploadJobStore()
            queue = store.subscribe("file-2")
            store.publish("file-2", {"type": "progress", "data": {"fraction": 0.2}})
            event = await asyncio.wait_for(queue.get(), timeout=1)
            store.unsubscribe("file-2", queue)
            return event

        event = asyncio.run(run())
        self.assertEqual(event["type"], "progress")

    def test_counts(self):
        store = UploadJobStore()
        store.create("queued")
        store.update("running", status="recognizing")
        store.update("done", status="complete")
        self.assertEqual(store.counts()["queued"], 1)
        self.assertEqual(store.counts()["running"], 1)
        self.assertEqual(store.counts()["total"], 3)


if __name__ == "__main__":
    unittest.main()
