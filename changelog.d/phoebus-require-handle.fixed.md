Multi-user web terminals whose project runs a Phoebus server now refuse the
implicit `"active"` display and address a handle or a named display instead,
so two users of one Phoebus product no longer act on each other's focused
display. Set `phoebus.require_handle: false` to keep `"active"`.
