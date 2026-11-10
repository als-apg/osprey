Generated project images build again. Once Playwright 1.65 was released the
image build failed at the Chromium step, because Playwright's bundled Node
needs `libatomic1`, which the image no longer carried; the image now installs it.
