"""Tests for the feedback-POST body-size contract in the per-user nginx proxy
location (osprey.deployment.web_terminals.render / nginx.conf.j2).

Nginx's own default (1m) is smaller than the feedback form's upload cap, so
without an explicit `client_max_body_size` a screenshot attached to a bug
report would 413 before the request ever reaches the user's app. These tests
pin the directive as an explicit contract of the per-user proxy location —
and, crucially, confirm it lands INSIDE that location block rather than
merely somewhere in the rendered file (a bare substring check would pass
even if the directive landed in the wrong `location {}`, or at server scope
where it would do nothing for the location that actually needs it).
"""

from __future__ import annotations

import re

from osprey.deployment.web_terminals.render import render_web_terminals

_CONFIG: dict = {
    "facility": {
        "name": "Demo Light Source",
        "prefix": "dls",
        "timezone": "America/Los_Angeles",
    },
    "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
    "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
    "modules": {
        "web_terminals": {
            "enabled": True,
            "users": ["alice", "bob"],
        }
    },
}


def _user_location_block(nginx_conf: str, user: str) -> str:
    """The body of `location /u/<user>/ { ... }` — everything between that
    directive's opening brace and its matching close. Isolating the block
    this way means an assertion against it can only be satisfied by a
    directive that actually landed inside THIS location, never by one that
    landed in a sibling location (the internal auth target, `/auth/`, the
    landing page, or another user's block)."""
    match = re.search(
        rf"location /u/{re.escape(user)}/ \{{(.*?)\n    \}}",
        nginx_conf,
        re.DOTALL,
    )
    assert match is not None, f"no location /u/{user}/ block found in rendered nginx.conf"
    return match.group(1)


def test_client_max_body_size_set_inside_each_per_user_location() -> None:
    """Every per-user proxy location states an explicit 4m body-size limit,
    beside the other request-shaping directives (`proxy_buffering off;` etc.)
    already scoped to that same location."""
    artifacts = render_web_terminals(_CONFIG)
    nginx_conf = artifacts["nginx/nginx.conf"]

    for user in ("alice", "bob"):
        block = _user_location_block(nginx_conf, user)
        assert "client_max_body_size 4m;" in block
        assert "proxy_buffering off;" in block


def test_client_max_body_size_scoped_to_only_the_per_user_locations() -> None:
    """The directive appears exactly once per user — never at server scope,
    never on the landing page's `location = /`, and never on a stable-bookmark
    redirect location, where it would either do nothing or apply too broadly."""
    artifacts = render_web_terminals(_CONFIG)
    nginx_conf = artifacts["nginx/nginx.conf"]

    assert nginx_conf.count("client_max_body_size 4m;") == 2
