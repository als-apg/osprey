Raising `ariel.enhancement_modules.image_caption.max_images_per_entry` now
reaches pictures already recorded as over the cap: `osprey ariel enhance
--module image_caption --retry-failed` captions the ones now within it, and
lowering the cap keeps existing captions. `osprey ariel status` reports how many
pictures are past the cap and names the command.
