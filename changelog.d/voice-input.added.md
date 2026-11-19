The Simple-view chat can take voice input. A mic icon beside Send records the
operator, a locally run Whisper model turns the speech into text, and the text
lands in the message box for the operator to review and send; nothing is sent
on its own and no audio is kept. `web.voice.inputs` lists the ways to talk:
`button` (hold the icon, the default), `key` (hold `web.voice.key`, default
Ctrl+Space) and `mic` (click the icon to listen for a push-to-talk microphone);
`[]` turns voice off. Whisper and ffmpeg are optional: without them the icon
stays hidden.
