"""UI 전용 PNG 상한과 손상 거부를 합성 바이트로 검사한다."""

import struct
import unittest
import zlib

from src.ui.image_validation import valid_report_png
from src.ui.report_view import referenced_png


def png(width, height):
    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload +
                struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(bytes(width + 1) * height)) + chunk(b"IEND", b""))


class ReportPngTest(unittest.TestCase):
    def test_tall_comparison_is_valid_and_available_inline(self):
        image = png(1080, 4320)
        self.assertTrue(valid_report_png(image))
        files = [{"name": "check/examples.png", "suffix": ".png", "data": image}]
        self.assertEqual(referenced_png("check/transform_check.md", "examples.png", files), image)

    def test_dimension_pixel_and_byte_limits(self):
        for width, height in ((0, 1), (1, 8193), (8193, 1), (4097, 4097)):
            with self.subTest(width=width, height=height):
                self.assertFalse(valid_report_png(png(width, height)))

        self.assertFalse(valid_report_png(bytes(8 * 1024 ** 2 + 1)))

    def test_corruption_truncation_and_trailing_bytes_are_rejected(self):
        image = png(1, 1)
        for damaged in (image[:-1], image + b"extra", image[:20] + b"X" + image[21:]):
            self.assertFalse(valid_report_png(damaged))
