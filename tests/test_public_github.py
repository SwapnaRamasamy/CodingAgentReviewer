import io
import json
import ssl
from urllib.error import URLError
from zipfile import ZipFile

import pytest

from reviewer import public_github as public


@pytest.mark.parametrize("url", [
    "https://github.com/example/repo",
    "[repository](https://github.com/example/repo)",
])
def test_public_snapshot_pins_default_branch_and_filters(monkeypatch, url):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for path, content in {
            "root/src/app.py": "print(1)", "root/.env": "SECRET=x",
            "root/node_modules/a.js": "ignored", "root/../escape.py": "bad",
            "root/.gitignore": "ignored.py", "root/ignored.py": "ignored",
            "root/large.py": "x" * 60001,
        }.items():
            archive.writestr(path, content)
    calls = []
    sha = "a" * 40

    def download(url, limit):
        calls.append(url)
        if "/commits/" in url:
            return json.dumps({"sha": sha}).encode()
        if "codeload" in url:
            return buffer.getvalue()
        return b'{"default_branch":"release/test"}'

    monkeypatch.setattr(public, "download", download)
    snapshot = public.public_snapshot(url)
    assert snapshot.files == {"src/app.py": "print(1)"}
    assert snapshot.revision == sha
    assert snapshot.branch == "release/test"
    assert calls[1].endswith("/commits/release%2Ftest")
    assert calls[2].endswith(sha)
    assert any("oversized" in item for item in snapshot.limitations)


@pytest.mark.parametrize("url", ["", "https://example.com/repo", "https://github.com/a/b/pull/1"])
def test_invalid_urls_do_not_download(monkeypatch, url):
    def unexpected(*args):
        pytest.fail("Invalid URLs must not access the network")
    monkeypatch.setattr(public, "download", unexpected)
    with pytest.raises(public.PublicRepositoryError):
        public.public_snapshot(url)


def test_download_size_limit(monkeypatch):
    monkeypatch.setattr(public, "urlopen", lambda *a, **kw: io.BytesIO(b"12345"))
    with pytest.raises(public.PublicRepositoryError, match="size limit"):
        public.download("https://api.github.com/repos/a/b", 4)


def test_temporary_connection_failure_is_retried(monkeypatch):
    from urllib.error import URLError
    calls = []

    def connect(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise URLError(TimeoutError())
        return io.BytesIO(b"ok")

    monkeypatch.setattr(public, "urlopen", connect)
    assert public.download("https://codeload.github.com/a/b/zip/sha", 10) == b"ok"
    assert len(calls) == 2


def test_timeout_names_failing_host(monkeypatch):
    from urllib.error import URLError

    def connect(*args, **kwargs):
        raise URLError(TimeoutError())

    monkeypatch.setattr(public, "urlopen", connect)
    with pytest.raises(public.PublicRepositoryError, match="codeload.github.com.*timed out"):
        public.download("https://codeload.github.com/a/b/zip/sha", 10)


@pytest.fixture(autouse=True)
def isolated_certificate_settings(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)


@pytest.mark.parametrize("host", ["api.github.com", "codeload.github.com"])
def test_download_requires_native_certificate_and_hostname_verification(monkeypatch, host):
    def connect(request, *, timeout, context):
        assert isinstance(context, public.truststore.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
        return io.BytesIO(b"ok")

    monkeypatch.setattr(public, "urlopen", connect)
    assert public.download(f"https://{host}/example", 10) == b"ok"


@pytest.mark.parametrize("wrapped", [True, False])
def test_certificate_failure_does_not_retry_or_disable_verification(monkeypatch, wrapped):
    calls = []

    def connect(*args, **kwargs):
        calls.append(kwargs["context"])
        error = ssl.SSLCertVerificationError(1, "certificate verify failed")
        raise URLError(error) if wrapped else error

    monkeypatch.setattr(public, "urlopen", connect)
    with pytest.raises(public.PublicRepositoryError, match="certificate for api.github.com.*SSL_CERT_FILE"):
        public.download("https://api.github.com/repos/a/b", 10)
    assert len(calls) == 1
    assert calls[0].verify_mode == ssl.CERT_REQUIRED
    assert calls[0].check_hostname is True


@pytest.mark.parametrize("invalid_pem", [False, True])
def test_invalid_ca_file_fails_before_network(monkeypatch, tmp_path, invalid_pem):
    ca_file = tmp_path / "company-ca.pem"
    if invalid_pem:
        ca_file.write_text("not a certificate", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_file))

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid CA configuration must not access the network")

    monkeypatch.setattr(public, "urlopen", unexpected)
    with pytest.raises(public.PublicRepositoryError, match="Could not load SSL_CERT_FILE"):
        public.download("https://api.github.com/repos/a/b", 10)


def test_custom_ca_settings_are_loaded_without_disabling_verification(monkeypatch):
    loaded = []
    monkeypatch.setenv("SSL_CERT_FILE", "company-ca.pem")
    monkeypatch.setenv("SSL_CERT_DIR", "company-certs")

    def load(context, *, cafile, capath):
        loaded.append((cafile, capath))

    monkeypatch.setattr(public.truststore.SSLContext, "load_verify_locations", load)
    context = public.github_ssl_context()
    assert loaded == [("company-ca.pem", "company-certs")]
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
