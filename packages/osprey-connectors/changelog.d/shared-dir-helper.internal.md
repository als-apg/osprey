`safety_file.ensure_shared_dir` creates a group-shared directory (every level it
creates at exactly `2770`, existing levels left as provisioned), and
`SHARED_DIR_MODE`, `SHARED_FILE_MODE` and `SHARED_LOCK_MODE` name the location's
modes once; the control-context record, write authorizations and the hooks' spool
use them in place of their own copies.
It makes each missing level itself, from the top, and sets the mode only on the
levels it made, on a descriptor that refuses a link. Two shifts for the hooks'
spool: a spool directory that already exists is no longer chmodded, and the
parent levels the spool creates on the way to it now get `2770` as well.
