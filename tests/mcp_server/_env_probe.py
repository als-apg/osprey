"""A sandbox probe that prints the environment names its child process can see."""

import json

#: Sandboxed code that prints its sorted ``os.environ`` keys between two markers.
DUMP_ENV_CODE = (
    "import json, os\n"
    "print('ENV_KEYS_JSON_START')\n"
    "print(json.dumps(sorted(os.environ.keys())))\n"
    "print('ENV_KEYS_JSON_END')\n"
)


def extract_env_keys(stdout: str) -> list[str]:
    """Pull the sorted env key list :data:`DUMP_ENV_CODE` printed out of a sandbox's stdout."""
    start = stdout.index("ENV_KEYS_JSON_START") + len("ENV_KEYS_JSON_START")
    end = stdout.index("ENV_KEYS_JSON_END")
    return json.loads(stdout[start:end])
