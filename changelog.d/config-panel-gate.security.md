The ARIEL panel's settings editor now honours `web.config_panel.enabled`.
A deployment that has taken the Config panel away gets `403` from both
`GET` and `PUT /api/config` there, as it already did on the web terminal, and
the Settings entry disappears from ARIEL's display menu. Previously the key
hid the terminal's tab while ARIEL still served — and accepted — the whole
document, provider base URLs included.
