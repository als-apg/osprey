Each machine caption now records the prompt it was made with. After a change to
`ariel.enhancement_modules.image_caption.prompt_template`, `osprey ariel status`
reports how many captions were made with an older prompt, and `osprey ariel
enhance --module image_caption --refresh-stale` captions those pictures again;
nothing is re-captioned without that command. Captions stored before this
release record no prompt and are left as they are.
