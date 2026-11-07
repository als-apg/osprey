A deployment's CANOPY screen library lives at `data/canopy/library`: screens
under `screens/` and, beside them, CANOPY's write-binding report
(`LIBRARY-REPORT.json`, schema `canopy-library-report/1`, and `LIBRARY-REPORT.md`),
written by `python -m canopy_mcp.report <library> --served-at library
--write`. `osprey build` refuses a library its report no longer covers, even
with `OSPREY_CANOPY_PAYLOAD_CHECK=0`: files changed since the report was
written (the paths are printed), a file that does not belong in it, any
`.svg`, or a file the container's runtime user cannot read or could write; the
refusal prints the command that repairs it. A user's agent files a screen for
the library with `request_push`, and the host reviews it with
`python -m osprey.integrations.canopy.push list|show|accept|reject`: `show`
re-checks the bundle in a hardened throwaway container with the image's own
CANOPY and prints every pushed file, report row and string with a digest that
`accept` must repeat; `accept` takes a reviewer who may edit the deployment,
writes the library in an order that an interruption leaves refused rather than
half-written, logs it, keeps replaced files for 14 days and prints the git
commands for the admin to run. The build turns the library into the
`canopy-screens` skill, which tells the agent which screen shows which device.
The how-to "Author and Review Screens in the CANOPY Library" covers both
avenues.
