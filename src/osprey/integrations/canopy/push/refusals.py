"""The exit codes of the push review and the refusal that carries one."""

from __future__ import annotations

# --- exit codes ---------------------------------------------------------------------------
# One per refusal. They are a contract: ``accept`` and ``reject`` reuse them, and the
# held-stack push negatives assert them one by one.

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ROOT_UID = 10
EXIT_NOT_OWNER = 11
EXIT_NO_DEPLOYMENT = 12
EXIT_NO_LIBRARY = 13
EXIT_RUNTIME_PODMAN = 14
EXIT_RUNTIME_MISSING = 15
EXIT_BAD_USER = 16
EXIT_NOT_ON_ROSTER = 17
EXIT_NO_VOLUME = 18
EXIT_BAD_RID = 19
EXIT_BUSY = 20
EXIT_STALE = 21
EXIT_CONTAINER_START = 22
EXIT_CONTAINER_TIMEOUT = 23
EXIT_CONTAINER_FAILED = 24
EXIT_BUNDLE_MISSING = 25
EXIT_STREAM_LINK = 26
EXIT_STREAM_CAP = 27
EXIT_BAD_LAYOUT = 28
EXIT_BAD_REQUEST = 29
EXIT_BAD_SEGMENT = 30
EXIT_RESERVED_SEGMENT = 31
EXIT_SUFFIX = 32
EXIT_FILE_TOO_LARGE = 33
EXIT_TOO_MANY_FILES = 34
EXIT_BUNDLE_TOO_LARGE = 35
EXIT_INVALID_TEXT = 36
EXIT_CASE_COLLISION = 37
EXIT_FILES_MISMATCH = 38
EXIT_BAD_SOURCE_DIR = 39
EXIT_BAD_TARGET_DIR = 40
EXIT_CASE_FILE = 41
EXIT_CASE_DIR = 42
EXIT_NOT_IMAGE = 43
EXIT_CHECK_ERRORS = 44
EXIT_CLOSURE_MISMATCH = 45
EXIT_MACRO_REF = 46
EXIT_CROSS_ROOT_REF = 47
EXIT_ROOT_RELATIVE_REF = 48
EXIT_REF_LEAVES_SOURCE = 49
EXIT_UNRESOLVED_REF = 50
EXIT_RULE_RETARGET = 51
EXIT_CLOSURE_REFUSED = 52
EXIT_REPORT = 53
EXIT_LIBRARY_COPY = 54
EXIT_BAD_VERDICT = 55
EXIT_REVIEWER = 56
EXIT_NO_STAGE = 57
EXIT_DIGEST_MISMATCH = 58
EXIT_BAD_STAGE = 59
EXIT_COLLISION = 60
EXIT_EMIT_DIGEST = 61
EXIT_REPLACED_EXISTS = 62
EXIT_BAD_REASON = 63
EXIT_WRITE_FAILED = 64

#: The exit code of each grammar code (the names ``request_push`` uses).
GRAMMAR_EXIT = {
    "bad_segment": EXIT_BAD_SEGMENT,
    "reserved_segment": EXIT_RESERVED_SEGMENT,
    "suffix": EXIT_SUFFIX,
    "file_too_large": EXIT_FILE_TOO_LARGE,
    "too_many_files": EXIT_TOO_MANY_FILES,
    "bundle_too_large": EXIT_BUNDLE_TOO_LARGE,
    "invalid_text": EXIT_INVALID_TEXT,
    "case_collision": EXIT_CASE_COLLISION,
}
#: The exit code of each refusal the in-container check reports.
CHECK_EXIT = {
    "not_image": EXIT_NOT_IMAGE,
    "check_errors": EXIT_CHECK_ERRORS,
    "closure_mismatch": EXIT_CLOSURE_MISMATCH,
    "macro_ref": EXIT_MACRO_REF,
    "cross_root_ref": EXIT_CROSS_ROOT_REF,
    "root_relative_ref": EXIT_ROOT_RELATIVE_REF,
    "ref_leaves_source": EXIT_REF_LEAVES_SOURCE,
    "unresolved_ref": EXIT_UNRESOLVED_REF,
    "rule_retargets_ref": EXIT_RULE_RETARGET,
    "rule_retargets_channel": EXIT_RULE_RETARGET,
    "report": EXIT_REPORT,
}


class Refusal(Exception):
    """A refused run: ``code`` is the exit status, ``lines`` what goes to stderr."""

    def __init__(self, code: int, *lines: str):
        super().__init__(lines[0] if lines else "")
        self.code = code
        self.lines = lines
