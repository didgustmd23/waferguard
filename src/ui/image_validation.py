"""로컬 보고서 그림의 크기·청크·CRC를 제한한다. 이미지 안전성 인증은 아니다."""

import struct
import zlib


def valid_report_png(data):
    """세로로 긴 비교 그림을 허용하되 파일 크기와 총 픽셀 수를 제한한다."""
    if len(data) > 8 * 1024 ** 2 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False

    offset, first, image_data = 8, True, False
    while offset + 12 <= len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + size
        if end > len(data):
            return False

        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:end - 4]
        crc = struct.unpack_from(">I", data, end - 4)[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != crc:
            return False

        if first:
            if kind != b"IHDR" or size != 13:
                return False

            width, height = struct.unpack_from(">II", payload)
            if not (1 <= width <= 8192 and 1 <= height <= 8192 and width * height <= 16 * 1024 ** 2):
                return False

            first = False
        elif kind == b"IHDR":
            return False

        image_data = image_data or kind == b"IDAT"
        if kind == b"IEND":
            return size == 0 and end == len(data) and image_data

        offset = end

    return False
