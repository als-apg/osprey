"""Auth sidecar: a FastAPI service that authenticates multi-user web-terminal access.

The sidecar answers nginx ``auth_request`` subrequests for each ``/u/<user>/``
location and serves the login flow. Enforcement lives entirely at this
perimeter; the per-user OSPREY containers hold no auth logic.
"""
