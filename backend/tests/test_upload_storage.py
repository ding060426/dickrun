import os
import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from upload_storage import UploadTooLargeError, save_upload_to_temp


class _Upload:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0
        self.read_sizes = []

    async def read(self, size: int):
        self.read_sizes.append(size)
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


class UploadStorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_is_copied_in_bounded_chunks(self):
        upload = _Upload(b"a" * 25)
        stored = await save_upload_to_temp(
            upload,
            suffix=".wav",
            max_bytes=100,
            chunk_size=8,
        )
        try:
            self.assertEqual(Path(stored.path).read_bytes(), b"a" * 25)
            self.assertEqual(stored.size_bytes, 25)
            self.assertGreater(len(upload.read_sizes), 2)
        finally:
            os.unlink(stored.path)

    async def test_oversized_upload_is_rejected(self):
        upload = _Upload(b"a" * 17)
        with self.assertRaises(UploadTooLargeError):
            await save_upload_to_temp(
                upload,
                suffix=".wav",
                max_bytes=16,
                chunk_size=8,
            )


if __name__ == "__main__":
    unittest.main()
