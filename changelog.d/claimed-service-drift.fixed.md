A claimed service template (`services/<name>`) now reports framework drift when a
newer OSPREY changes the packaged template, as a claimed Claude Code artifact
already did. The drift check used to look for every template under the Claude
Code tree and skipped services without a word.
