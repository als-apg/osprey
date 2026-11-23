Every framework reader of `facility.json` checks the document's version header
and uses none of a document whose version it does not declare: a reader that
refuses an unreadable file refuses it, a soft reader treats it as absent, and
both tell you to rebuild with `osprey build`. The write guard fails closed: it
refuses a write when the facility document is missing, unreadable, of an
undeclared version, not an object, or without a channel list. The facility
description no longer imports the framework, click or pyaml-cs-osprey.
