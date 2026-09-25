"""ChatClient's Chat v1 surface: the create body, the guarded get, chunking, and
the lazy-import seam.

Every test drives a ``MagicMock`` standing in for the ``googleapiclient``
discovery service — nothing here opens a socket, and **no Google library is
needed at collection or run time**, which is itself pinned below: one test loads
a fresh copy of the module with every ``google*`` import blocked and exercises
the whole client over it, and a second proves the block is real by watching
``build_chat_service`` — the one function that does import them — fail under it.

Four properties get more attention than the rest, because each breaks a live
bridge in a way a mock will not otherwise notice:

* a create **must not retry or swallow** — the engine's redelivery is built on
  ``post_answer`` raising, so a failure here makes exactly one call and
  propagates;
* ``messageReplyOption`` must be on every create, or an answer whose thread has
  since gone is an error rather than a new thread;
* the chunker's ceiling must stay Chat's — the splitting itself is shared and is
  tested in ``tests/bridges/test_text.py``, so what is left here is the default
  the ops layer relies on when it calls ``chunk_text`` bare;
* the HTTP leg must be serialized, because the engine calls the ops members from
  the drain thread and the ingest path at once over this one shared service.
"""

import importlib.util
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from osprey.bridges.core import text as core_text
from osprey.bridges.core.errors import UndeliverableError
from osprey.bridges.google_chat import GoogleChatBridgeConfig
from osprey.bridges.google_chat import client as client_module
from osprey.bridges.google_chat.client import (
    MAX_CHARS,
    REPLY_OPTION,
    SA_SCOPES,
    ChatClient,
    build_chat_service,
    chunk_text,
)

CFG = GoogleChatBridgeConfig(
    sa_key="/etc/osprey/sa-key.json",
    subscription="projects/p/subscriptions/s",
    app_id="users/1234567890",
)

SPACE = "spaces/AAAA"
THREAD = "spaces/AAAA/threads/TTTT"
MESSAGE = "spaces/AAAA/messages/MMMM"

CREATED = {"name": MESSAGE, "thread": {"name": THREAD}}


def make_service(*, created=None, fetched=None):
    """A discovery-service mock answering the two calls the client makes."""
    service = MagicMock()
    messages = service.spaces.return_value.messages.return_value
    messages.create.return_value.execute.return_value = CREATED if created is None else created
    messages.get.return_value.execute.return_value = CREATED if fetched is None else fetched
    return service


def messages_of(service):
    """The mock the client's ``spaces().messages()`` chain lands on."""
    return service.spaces.return_value.messages.return_value


def create_call(service):
    """Keyword arguments of the last ``messages.create`` call."""
    return messages_of(service).create.call_args.kwargs


# --- create_message --------------------------------------------------------


def test_create_posts_into_the_space_and_thread():
    service = make_service()
    ChatClient(CFG, service).create_message(SPACE, THREAD, "hello")

    call = create_call(service)
    assert call["parent"] == SPACE
    assert call["messageReplyOption"] == REPLY_OPTION
    assert call["body"] == {"text": "hello", "thread": {"name": THREAD}}


def test_create_reply_option_is_the_fallback_spelling():
    # Pinned as a literal, not just by name: this exact option is what makes a
    # reply into a vanished thread open a new one instead of failing.
    assert REPLY_OPTION == "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"


@pytest.mark.parametrize("thread", [None, ""])
def test_create_without_a_thread_omits_the_thread_body(thread):
    service = make_service()
    ChatClient(CFG, service).create_message(SPACE, thread, "hello")

    assert create_call(service)["body"] == {"text": "hello"}


def test_create_attaches_cards():
    cards = [{"cardId": "artifacts-run-1", "card": {"sections": []}}]
    service = make_service()
    ChatClient(CFG, service).create_message(SPACE, THREAD, "hello", cards=cards)

    assert create_call(service)["body"]["cardsV2"] == cards


@pytest.mark.parametrize("cards", [None, []])
def test_create_without_cards_sends_a_text_only_body(cards):
    # Not "an empty cardsV2": the key is absent, so a text-only post is identical
    # to what the bridge sent before card delivery existed.
    service = make_service()
    ChatClient(CFG, service).create_message(SPACE, THREAD, "hello", cards=cards)

    assert "cardsV2" not in create_call(service)["body"]


def test_create_returns_the_created_message():
    client = ChatClient(CFG, make_service())

    assert client.create_message(SPACE, THREAD, "hello") == CREATED


def test_create_returns_an_empty_dict_for_a_non_message_answer():
    # The message landed (the call returned); there is simply nothing to report.
    client = ChatClient(CFG, make_service(created="not-a-resource"))

    assert client.create_message(SPACE, THREAD, "hello") == {}


def test_create_raises_on_transport_failure_without_retrying():
    service = make_service()
    execute = messages_of(service).create.return_value.execute
    execute.side_effect = RuntimeError("chat is down")

    with pytest.raises(RuntimeError, match="chat is down"):
        ChatClient(CFG, service).create_message(SPACE, THREAD, "hello")

    # No retry and no swallow: the ops layer owns both, per outcome.
    assert execute.call_count == 1


class _FakeHttpError(Exception):
    """The shape of ``googleapiclient.errors.HttpError`` the mapping reads: a ``resp``
    with a ``status`` and the parsed ``reason``. Built by hand so this file keeps its
    no-Google-library property."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(f"<HttpError {status} returned {reason!r}>")
        self.resp = MagicMock(status=status)
        self.reason = reason


@pytest.mark.parametrize(
    "error",
    [
        _FakeHttpError(403, "This Chat app is not a member of this space."),
        _FakeHttpError(404, "Space not found."),
    ],
    ids=["not-a-member", "gone"],
)
def test_create_maps_a_permanent_refusal_of_the_destination_to_undeliverable(error):
    """Chat will refuse this destination on every later attempt too, so the engine gets
    the one signal that lets it stop retrying rather than the bare transport error."""
    service = make_service()
    messages_of(service).create.return_value.execute.side_effect = error

    with pytest.raises(UndeliverableError, match=error.reason) as caught:
        ChatClient(CFG, service).create_message(SPACE, THREAD, "hello")

    assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    "error",
    [
        _FakeHttpError(403, "The caller does not have permission"),
        _FakeHttpError(429, "Quota exceeded"),
        _FakeHttpError(500, "Internal error"),
        RuntimeError("connection reset"),
    ],
    ids=["other-403", "429", "500", "transport"],
)
def test_create_passes_every_other_failure_through_unchanged(error):
    # A missing scope is a 403 too, but it is a bridge misconfiguration that a fix will
    # clear for every destination at once — not this destination refusing forever.
    service = make_service()
    messages_of(service).create.return_value.execute.side_effect = error

    with pytest.raises(type(error)):
        ChatClient(CFG, service).create_message(SPACE, THREAD, "hello")


def test_create_does_not_chunk():
    # Chunking is the caller's call (which chunk carries the cards is an ops
    # decision), so an oversized body is sent as one message, exactly as given.
    text = "x" * (MAX_CHARS + 500)
    service = make_service()
    ChatClient(CFG, service).create_message(SPACE, THREAD, text)

    assert create_call(service)["body"]["text"] == text


# --- get_message -----------------------------------------------------------


def test_get_fetches_by_resource_name():
    service = make_service()

    assert ChatClient(CFG, service).get_message(MESSAGE) == CREATED
    assert messages_of(service).get.call_args.kwargs == {"name": MESSAGE}


def test_get_returns_none_on_failure_without_retrying():
    service = make_service()
    execute = messages_of(service).get.return_value.execute
    execute.side_effect = RuntimeError("chat is down")

    # Guarded, unlike create: reply context is enrichment, so a hiccup degrades
    # to answering without the quoted text rather than failing the answer.
    assert ChatClient(CFG, service).get_message(MESSAGE) is None
    assert execute.call_count == 1


def test_get_returns_none_for_a_non_message_answer():
    client = ChatClient(CFG, make_service(fetched=["not", "a", "message"]))

    assert client.get_message(MESSAGE) is None


# --- chunk_text: the Chat ceiling ------------------------------------------
#
# The splitting rules live in osprey.bridges.core.text and are tested, at both
# shipped ceilings, in tests/bridges/test_text.py. What this module owns is the
# one Chat-specific thing left: the default limit a bare call gets.


def test_the_wrapper_defaults_to_the_chat_limit():
    # ops.py calls chunk_text bare, so this default is what keeps a long answer
    # under the limit Chat actually enforces.
    text = "x" * (MAX_CHARS + 500)

    assert chunk_text(text) == core_text.chunk_text(text, MAX_CHARS)
    assert all(len(chunk) <= MAX_CHARS for chunk in chunk_text(text))


def test_the_fence_helper_is_still_importable_from_here():
    # Re-exported rather than reimplemented: the name moved to the core with the
    # chunker, and importers of this module should not have had to notice.
    assert client_module._fence_spans is core_text._fence_spans


# --- the service seam ------------------------------------------------------


def test_an_injected_service_is_used_as_given(monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr(client_module, "build_chat_service", factory)
    service = make_service()

    ChatClient(CFG, service).create_message(SPACE, THREAD, "hello")

    factory.assert_not_called()
    assert messages_of(service).create.call_count == 1


def test_no_service_builds_the_default_one(monkeypatch):
    service = make_service()
    factory = MagicMock(return_value=service)
    monkeypatch.setattr(client_module, "build_chat_service", factory)

    client_module.ChatClient(CFG).create_message(SPACE, THREAD, "hello")

    factory.assert_called_once_with(CFG)
    assert messages_of(service).create.call_count == 1


@pytest.fixture
def fake_google(monkeypatch):
    """Stand-ins for the two modules :func:`build_chat_service` imports.

    Installed in ``sys.modules`` so the function-local imports resolve to these
    without ``google-auth``/``google-api-python-client`` being installed.
    """
    credentials = object()
    service_account = SimpleNamespace(
        Credentials=SimpleNamespace(from_service_account_file=MagicMock(return_value=credentials))
    )
    oauth2 = ModuleType("google.oauth2")
    oauth2.service_account = service_account
    google = ModuleType("google")
    google.oauth2 = oauth2
    discovery = ModuleType("googleapiclient.discovery")
    discovery.build = MagicMock(return_value="the-chat-service")
    googleapiclient = ModuleType("googleapiclient")
    googleapiclient.discovery = discovery
    for name, module in (
        ("google", google),
        ("google.oauth2", oauth2),
        ("googleapiclient", googleapiclient),
        ("googleapiclient.discovery", discovery),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        credentials=credentials,
        from_key_file=service_account.Credentials.from_service_account_file,
        build=discovery.build,
        service="the-chat-service",
    )


def test_default_factory_authenticates_as_the_chat_bot(fake_google):
    assert build_chat_service(CFG) is fake_google.service

    fake_google.from_key_file.assert_called_once_with(
        CFG.sa_key, scopes=["https://www.googleapis.com/auth/chat.bot"]
    )
    fake_google.build.assert_called_once_with(
        "chat", "v1", credentials=fake_google.credentials, cache_discovery=False
    )


def test_the_scope_is_chat_bot():
    # Posting "as the app" and reading a message snapshot both hang off this one
    # scope; the bridge holds no user OAuth token.
    assert SA_SCOPES == ("https://www.googleapis.com/auth/chat.bot",)


# --- no Google library required --------------------------------------------


class BlockGoogleImports:
    """A meta-path finder that refuses every ``google*`` import."""

    def find_spec(self, fullname, _path=None, _target=None):
        if fullname == "google" or fullname.startswith(("google.", "googleapiclient")):
            raise ImportError(f"blocked in this test: {fullname}")
        return None


@pytest.fixture
def no_google(monkeypatch):
    """Make every ``google*`` import fail, cached ones included."""
    for name in list(sys.modules):
        if name == "google" or name.startswith(("google.", "googleapiclient")):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [BlockGoogleImports(), *sys.meta_path])


def load_client_copy():
    """Load a fresh, private copy of the client module.

    A copy rather than a reload, so blocking imports for one test cannot leave the
    canonical module rebound for the rest of the suite. The alias keeps the
    package prefix, which is what makes its ``from .config import ...`` resolve.
    """
    alias = "osprey.bridges.google_chat._client_import_probe"
    spec = importlib.util.spec_from_file_location(alias, client_module.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.usefixtures("no_google")
def test_the_client_works_with_no_google_library_installed():
    fresh = load_client_copy()
    service = make_service()
    client = fresh.ChatClient(CFG, service)

    assert client.create_message(SPACE, THREAD, "hello") == CREATED
    assert client.get_message(MESSAGE) == CREATED
    assert fresh.chunk_text("abcdefgh", limit=3) == ["abc", "def", "gh"]


@pytest.mark.usefixtures("no_google")
def test_only_the_default_factory_needs_the_google_libraries():
    # The other half of the test above: proof the block is real rather than
    # vacuous, and that the imports are function-local to exactly one place.
    fresh = load_client_copy()

    with pytest.raises(ImportError, match="blocked in this test"):
        fresh.build_chat_service(CFG)


# --- thread safety ---------------------------------------------------------


def test_calls_do_not_interleave_across_threads():
    # The engine drives the ops members from the drain thread and the ingest path
    # at once, over this one shared service.
    timeline = []
    guard = threading.Lock()

    def execute(*_args, **_kwargs):
        with guard:
            timeline.append("enter")
        time.sleep(0.005)
        with guard:
            timeline.append("exit")
        return CREATED

    service = make_service()
    messages_of(service).create.return_value.execute.side_effect = execute
    client = ChatClient(CFG, service)
    threads = [
        threading.Thread(target=client.create_message, args=(SPACE, THREAD, "hello"))
        for _ in range(6)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert timeline == ["enter", "exit"] * 6


# --- list_members ----------------------------------------------------------


def members_service(*pages):
    """A service whose ``spaces().members().list(...).execute()`` answers ``pages`` in
    order (an exception in the list is raised)."""
    service = MagicMock()
    execute = service.spaces.return_value.members.return_value.list.return_value.execute
    execute.side_effect = list(pages)
    return service


def list_calls(service):
    return [c.kwargs for c in service.spaces.return_value.members.return_value.list.call_args_list]


def membership(n):
    return {"member": {"name": f"users/{n}", "type": "HUMAN"}, "state": "JOINED"}


def test_list_members_follows_the_page_token():
    service = members_service(
        {"memberships": [membership(1)], "nextPageToken": "t2"},
        {"memberships": [membership(2)]},
    )
    members, more = ChatClient(CFG, service=service).list_members(SPACE, limit=50)
    assert [m["member"]["name"] for m in members] == ["users/1", "users/2"]
    assert more is False
    assert list_calls(service) == [
        {"parent": SPACE, "pageSize": 50},
        {"parent": SPACE, "pageSize": 50, "pageToken": "t2"},
    ]


def test_list_members_stops_at_the_limit_and_reports_more():
    service = members_service(
        {"memberships": [membership(1), membership(2)], "nextPageToken": "t2"},
        {"memberships": [membership(3)]},
    )
    members, more = ChatClient(CFG, service=service).list_members(SPACE, limit=2)
    assert len(members) == 2
    assert more is True
    assert len(list_calls(service)) == 1
    assert list_calls(service)[0]["pageSize"] == 2


def test_list_members_stops_on_a_repeated_token():
    service = members_service(
        {"memberships": [membership(1)], "nextPageToken": "loop"},
        {"memberships": [membership(2)], "nextPageToken": "loop"},
        {"memberships": [membership(3)], "nextPageToken": "loop"},
    )
    members, more = ChatClient(CFG, service=service).list_members(SPACE, limit=50)
    assert [m["member"]["name"] for m in members] == ["users/1", "users/2"]
    assert more is False
    assert len(list_calls(service)) == 2


def test_list_members_raises_on_transport_failure():
    service = members_service(RuntimeError("chat is down"))
    with pytest.raises(RuntimeError, match="chat is down"):
        ChatClient(CFG, service=service).list_members(SPACE, limit=50)


def test_list_members_treats_a_non_dict_page_as_empty():
    service = members_service(["not", "a", "page"])
    assert ChatClient(CFG, service=service).list_members(SPACE, limit=50) == ([], False)
