**Breaking change:** connector writes now follow one verdict and start off on
every machine until that machine's switch is turned on. An unknown owner,
target or record refuses the write. A write tier now arms every machine; the
shipped `readwrite` and `admin` tiers armed only the virtual accelerator
before. Each machine stays off until its own switch is on.
