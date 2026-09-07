The writes-disabled refusal now sends you to the build profile: "Arm them under
`config:` in profile.yml ... then run `osprey build`". It used to say
`config.yml`, which every `osprey build` regenerates — so the change an operator
made there was gone by the time it would have taken effect.
