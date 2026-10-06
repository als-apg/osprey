An `OSPREY_<SERVICE>_IMAGE` override that names another image now keeps that
service out of every build a deploy makes. The deploy reports
`<service> runs <image> (<variable>); not built` and starts the image you named,
instead of rebuilding OSPREY's recipe and tagging it with that name.
