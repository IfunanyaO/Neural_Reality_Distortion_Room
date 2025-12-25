# engine_integration/message_protocol.py

import struct
import json
from enum import IntEnum
from typing import Tuple

class MessageType(IntEnum):
    FRAME = 1
    CONTROL = 2

_HEADER_STRUCT = struct.Struct("!BI")  # 1 byte type, 4 bytes length

def pack_frame(jpeg_bytes: bytes) -> bytes:
    """
    Package a JPEG frame as a typed message.
    """
    header = _HEADER_STRUCT.pack(MessageType.FRAME, len(jpeg_bytes))
    return header + jpeg_bytes

def pack_control(payload: dict) -> bytes:
    """
    Package a JSON control message as a typed message.
    """
    data = json.dumps(payload).encode("utf-8")
    header = _HEADER_STRUCT.pack(MessageType.CONTROL, len(data))
    return header + data

def unpack_header(header_bytes: bytes) -> Tuple[MessageType, int]:
    """
    Parse the header to get (message_type, payload_length).
    """
    msg_type_val, length = _HEADER_STRUCT.unpack(header_bytes)
    return MessageType(msg_type_val), length

def decode_control(payload: bytes) -> dict:
    return json.loads(payload.decode("utf-8"))
