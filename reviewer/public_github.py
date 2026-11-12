from __future__ import annotations

import io
import json
import os
import re
import socket
import ssl
import stat
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pathspec
import truststore

from reviewer.models import Snapshot, safe_path
from reviewer.sources import MAX_FILE_BYTES, add_file, eligible, parse_github_url

MAX_ARCHIVE_BYTES = 25_000_000


class PublicRepositoryError(ValueError):
    pass


def github_ssl_context():
    # Use native certificate-chain validation, including Windows-managed CAs.
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    cafile = os.environ.get("SSL_CERT_FILE") or None
    capath = os.environ.get("SSL_CERT_DIR") or None
    if cafile or capath:
        try:
            context.load_verify_locations(cafile=cafile, capath=capath)
        except (OSError, ssl.SSLError):
            raise PublicRepositoryError(
                "Could not load SSL_CERT_FILE or SSL_CERT_DIR. "
                "Use an existing, valid CA certificate file/directory supplied by your IT team."
            ) from None
    context.set_alpn_protocols(["http/1.1"])
    return context


def download(url, limit):
    request = Request(url, headers={"User-Agent": "CodeReviewStudio", "Accept": "application/vnd.github+json"})
    host = urlsplit(url).hostname
    context = github_ssl_context()
    for attempt in range(2):
        try:
            with urlopen(request, timeout=90, context=context) as response:
                data = response.read(limit + 1)
            break
        except HTTPError as error:
            if error.code == 404:
                raise PublicRepositoryError("Repository or branch not found. Enter a public GitHub repository URL.") from None
            if error.code in (403, 429):
                raise PublicRepositoryError("GitHub denied this request or its anonymous request limit was reached. Try again later.") from None
            raise PublicRepositoryError(f"GitHub returned HTTP {error.code} from {host}. Please try again.") from None
        except (URLError, TimeoutError, ConnectionError, ssl.SSLCertVerificationError) as error:
            reason = error.reason if isinstance(error, URLError) else error
            if isinstance(reason, ssl.SSLCertVerificationError):
                raise PublicRepositoryError(
                    f"Could not verify the HTTPS certificate for {host}. "
                    "The app uses system-trusted certificates. If you use a company proxy or VPN, "
                    "ask IT to install its trusted CA or provide a PEM file for SSL_CERT_FILE, "
                    "then restart the app."
                ) from None
            if attempt == 0:
                continue
            if isinstance(reason, (TimeoutError, socket.timeout)):
                detail = "The request timed out; a large archive or slow connection may be responsible."
            elif isinstance(reason, socket.gaierror):
                detail = "DNS could not resolve this host. Check DNS or VPN settings."
            else:
                detail = "The connection failed. Check whether your proxy, VPN, or firewall allows this host."
            raise PublicRepositoryError(
                f"Could not reach {host} after two attempts. {detail} "
                "This is a connection failure, not a repository visibility check."
            ) from None
    if len(data) > limit:
        raise PublicRepositoryError("Repository download exceeds the size limit. Choose a smaller repository.")
    return data


def public_snapshot(url, ref=""):
    url = url.strip()
    formatted = re.fullmatch(r"\[[^\]\r\n]*\]\((https://github\.com/[^\s()]+)\)", url)
    if formatted:
        url = formatted.group(1)
    try:
        owner, repo, pr = parse_github_url(url)
    except ValueError:
        raise PublicRepositoryError("Enter a URL like https://github.com/owner/repository.") from None
    if pr or owner in (".", "..") or repo in ("", ".", ".."):
        raise PublicRepositoryError("Enter a repository URL; use the authenticated option for pull requests.")
    api = f"https://api.github.com/repos/{owner}/{repo}"
    metadata = json.loads(download(api, 1_000_000))
    branch = ref.strip() or metadata["default_branch"]
    commit = json.loads(download(f"{api}/commits/{quote(branch, safe='')}", 2_000_000))
    sha = commit.get("sha", "")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise PublicRepositoryError("GitHub did not return a valid commit revision.")
    archive = download(f"https://codeload.github.com/{owner}/{repo}/zip/{sha}", MAX_ARCHIVE_BYTES)
    files, limitations = {}, ["Public repository snapshot; nested .gitignore rules are not interpreted."]
    with ZipFile(io.BytesIO(archive)) as zipped:
        entries = zipped.infolist()
        if len(entries) > 20_000:
            raise PublicRepositoryError("Repository contains too many archive entries (limit 20,000).")
        candidates = []
        ignore_lines = []
        for entry in entries:
            _, separator, path = entry.filename.partition("/")
            if not separator or not safe_path(path) or entry.is_dir():
                continue
            if stat.S_ISLNK(entry.external_attr >> 16):
                continue
            if path == ".gitignore" and entry.file_size <= MAX_FILE_BYTES:
                ignore_lines = zipped.read(entry).decode("utf-8", errors="replace").splitlines()
            if eligible(path):
                candidates.append((path, entry))
        spec = pathspec.PathSpec.from_lines("gitwildmatch", ignore_lines)
        for path, entry in sorted(candidates, key=lambda pair: pair[0]):
            if spec.match_file(path):
                continue
            if entry.file_size > MAX_FILE_BYTES:
                limitations.append(f"Skipped oversized file: {path}")
                continue
            try:
                with zipped.open(entry) as source:
                    data = source.read(MAX_FILE_BYTES + 1)
                if len(data) > MAX_FILE_BYTES:
                    limitations.append(f"Skipped oversized file: {path}")
                    continue
                add_file(files, path, data.decode("utf-8"), limitations)
            except UnicodeError:
                limitations.append(f"Skipped non-UTF-8 file: {path}")
    if not files:
        raise PublicRepositoryError("No supported source files were found in this repository.")
    return Snapshot(source=url.strip(), revision=sha, owner=owner, repo=repo, branch=branch,
                    files=files, limitations=limitations)
