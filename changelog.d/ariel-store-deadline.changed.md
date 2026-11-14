An ARIEL database that takes the connection and never answers no longer holds
`osprey ariel` commands, the ARIEL panel's start-up, the ARIEL tools or a deploy
for ever: each gives it a stated time, then ends with `the ARIEL store at
<host>:<port>/<database> did not answer within <N> s`. A refused connection
reports the store's own first error. An HTML export of a page that never
settles now ends at the export deadline with "did not load".
