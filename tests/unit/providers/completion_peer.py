"""Owned native SDK protocol peer shared by provider tests."""

from contextlib import asynccontextmanager, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import asyncio
import json
import socket
import ssl
import struct
import threading
import zlib
from types import SimpleNamespace

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import pytest


def aws_event_frame(event):
    """Encode the AWS event-stream CRC framing consumed by botocore."""
    name, = event
    headers = b""
    for key, value in [(":message-type", "event"), (":event-type", name), (":content-type", "application/json")]:
        key, value = key.encode(), value.encode()
        headers += bytes([len(key)]) + key + b"\x07" + struct.pack(">H", len(value)) + value
    payload = json.dumps(event[name]).encode()
    preamble = struct.pack(">II", 16 + len(headers) + len(payload), len(headers))
    message = preamble + struct.pack(">I", zlib.crc32(preamble)) + headers + payload
    return message + struct.pack(">I", zlib.crc32(message))

def chunk(delta=None, *, reason=None, usage=None, index=0):
    return {
        "id": "owned-stream", "object": "chat.completion.chunk", "created": 0,
        "model": "owned-model",
        "choices": [] if delta is None else [{"index": index, "delta": delta, "finish_reason": reason}],
        "usage": usage,
    }


@pytest.fixture
def native_peer(monkeypatch, tmp_path):
    @contextmanager
    def serve(frames, *, pause_after_first=False, pause_after_second=False, wire="openai", json_response=False,
              pause_before_headers=False, tls=False):
        assert wire in {"openai", "vertex", "bedrock"}
        encoded = [] if json_response else [aws_event_frame(frame) if wire == "bedrock"
                   else ("data: " + json.dumps(frame) + "\n\n").encode() for frame in frames]
        calls, responses = [], []
        release = threading.Event()
        release_second = threading.Event()
        release_headers = threading.Event()
        headers_reached = asyncio.Event()
        loop = asyncio.get_running_loop()
        pause_timed_out = threading.Event()
        raw_responses = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                return None

            def do_GET(self):
                calls.append({"path": self.path, "body": None,
                              "aws_signed": self.headers.get("Authorization", "").startswith("AWS4-HMAC-SHA256 ")})
                valid = (wire == "vertex" and self.path.split("?", 1)[0] == "/v1beta/models") or (
                    wire == "bedrock" and self.path.split("?", 1)[0] == "/foundation-models")
                if not valid or not json_response or len(frames) != 1:
                    self.send_error(400)
                    return
                payload = json.dumps(frames[0]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append({"path": self.path, "body": body,
                              "aws_signed": self.headers.get("Authorization", "").startswith("AWS4-HMAC-SHA256 ")})
                loop.call_soon_threadsafe(headers_reached.set)
                if pause_before_headers and not release_headers.wait(timeout=10):
                    pause_timed_out.set()
                path = self.path.split("?", 1)[0]
                valid = (wire == "openai" and path == "/v1/chat/completions") or (
                    wire == "vertex" and path.startswith("/v1beta/models/")
                    and path.endswith((":generateContent", ":streamGenerateContent"))
                ) or (wire == "bedrock" and path.endswith(("/converse", "/converse-stream")))
                if not valid:
                    self.send_error(400)
                    return
                streaming = (wire == "openai" and body.get("stream") is True) or (
                    wire == "vertex" and ":streamGenerateContent" in path
                ) or (wire == "bedrock" and path.endswith("/converse-stream"))
                if not streaming:
                    if not frames:
                        self.send_error(400)
                        return
                    payload = json.dumps(frames[0]).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if json_response:
                    self.send_error(400)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.amazon.eventstream" if wire == "bedrock" else "text/event-stream")
                if wire == "bedrock":
                    self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    for position, payload in enumerate(encoded):
                        if position == 1 and pause_after_first:
                            if not release.wait(timeout=10):
                                pause_timed_out.set()
                        if position == 2 and pause_after_second:
                            if not release_second.wait(timeout=10):
                                pause_timed_out.set()
                        self.wfile.write((f"{len(payload):x}\r\n".encode() + payload + b"\r\n") if wire == "bedrock" else payload)
                        self.wfile.flush()
                    if wire == "openai":
                        self.wfile.write(b"data: [DONE]\n\n")
                    elif wire == "bedrock":
                        self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    # Closing a native SDK stream disconnects this owned peer.
                    return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        ca_bundle = None
        if tls:
            import datetime
            import ipaddress
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.x509.oid import NameOID

            key = ec.generate_private_key(ec.SECP256R1())
            name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "owned-provider-test")])
            now = datetime.datetime.now(datetime.timezone.utc)
            cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                    .serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=1))
                    .not_valid_after(now+datetime.timedelta(days=1))
                    .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
                    .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
                    .sign(key, hashes.SHA256()))
            cert_file, key_file = tmp_path / "owned-ca.pem", tmp_path / "owned-tls-key.pem"
            cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
            key_file.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
            key_file.chmod(0o600)
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert_file, key_file)
            server.socket = context.wrap_socket(server.socket, server_side=True)
            ca_bundle = str(cert_file)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

        def checked_connect(sock, address):
            assert address == ("127.0.0.1", port), "Test attempted a connection outside its owned peer"
            return connect(sock, address)

        def checked_connect_ex(sock, address):
            assert address == ("127.0.0.1", port), "Test attempted a connection outside its owned peer"
            return connect_ex(sock, address)

        with monkeypatch.context() as owned:
            owned.setenv("NO_PROXY", "127.0.0.1")
            owned.setattr(socket.socket, "connect", checked_connect)
            owned.setattr(socket.socket, "connect_ex", checked_connect_ex)
            try:
                scheme = "https" if tls else "http"
                yield SimpleNamespace(url=f"{scheme}://127.0.0.1:{port}/v1", origin=f"{scheme}://127.0.0.1:{port}", calls=calls,
                                      responses=responses, raw_responses=raw_responses,
                                      release=release, release_second=release_second, pause_timed_out=pause_timed_out,
                                      headers_reached=headers_reached, release_headers=release_headers,
                                      ca_bundle=ca_bundle)
            finally:
                release.set()
                release_second.set()
                release_headers.set()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                assert not thread.is_alive()

    return serve


@asynccontextmanager
async def owned_client(peer, *, after_response=None):
    async def remember_response(response):
        peer.responses.append(response)
        if after_response is not None:
            await after_response(response)

    async with AsyncOpenAI(
        api_key="owned-test-credential", base_url=peer.url, max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False, event_hooks={"response": [remember_response]}),
    ) as client:
        yield client


def openai_response(reason="stop", indices=(0,)):
    return ChatCompletion(
        id="owned-completion", model="owned-model", created=0, object="chat.completion",
        choices=[{"index": index, "finish_reason": reason,
                  "message": {"role": "assistant", "content": "owned text"}}
                 for index in indices],
        usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    )


@asynccontextmanager
async def owned_vertex_client(peer):
    from kestrel_llm_vertex import VertexAIAdapter

    async def remember_response(response):
        peer.responses.append(response)

    client = VertexAIAdapter.create_provider({"api_key": "owned-google-credential",
        "base_url": peer.origin, "api_version": "v1beta"}).client
    client.http.event_hooks["response"].append(remember_response)
    try:
        yield client
    finally:
        await client.close()


@asynccontextmanager
async def owned_bedrock_client(peer):
    from kestrel_llm_bedrock import BedrockAdapter

    config = {"endpoint_url": peer.origin, "region": "us-east-1",
              "aws_access_key_id": "owned-access", "aws_secret_access_key": "owned-secret"}
    if getattr(peer, "ca_bundle", None) is not None:
        config["ca_bundle"] = peer.ca_bundle
    client = BedrockAdapter.create_provider(config).client

    def remember_response(http_response, **kwargs):
        peer.raw_responses.append(http_response.raw)

    client.runtime.meta.events.register("after-call.bedrock-runtime.ConverseStream", remember_response)
    try:
        yield client
    finally:
        await client.close()
