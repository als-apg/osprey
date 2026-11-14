The JS toolchain moves to happy-dom 21, vitest 5.1, eslint 10.14, globals
17.14, jsdom 30.1.3 and prettier 3.9.10. happy-dom 21 stops loading iframe
pages by default, so the suite turns that back on: the panel, theme and SSE
suites build their own iframes and read their windows. Dependabot no longer
bumps dockview-core; like marked and dompurify, the vendor manifest pins it
and it moves by hand.
