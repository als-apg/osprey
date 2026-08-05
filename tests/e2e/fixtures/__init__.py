"""Fixture material for the e2e suite.

Historically this directory held only non-Python fixture assets (Dockerfiles, a
provisioning script, a facility config). The Google Chat bridge lane adds
importable fakes here, so the directory is a package: ``tests`` and ``tests.e2e``
are packages already, and the fakes are imported as
``tests.e2e.fixtures.gchat_chat_fake`` and friends.
"""
