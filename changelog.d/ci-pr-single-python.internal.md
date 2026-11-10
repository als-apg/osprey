Pull requests run the unit and serial-suite lanes on Python 3.12 only; 3.11 and 3.14 run on main
pushes and the nightly revalidation, and on pull requests labelled `full-ci` or opened by
Dependabot.
ci.yml writes the setup steps it repeats across lanes once, as YAML anchors, and drops the comments
that repeated alongside them.
