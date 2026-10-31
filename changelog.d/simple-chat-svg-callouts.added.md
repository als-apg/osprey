In the Simple view, the agent can draw a small picture in its reply: a fenced
`svg` block is shown as the drawing, in a box with its source one click away.
The agent uses it for a quick look at a result, such as the top three values
or a few key numbers, and links the artifact that holds the full result. It
draws bars only for values in the same unit, with each bar's length set by its
value, and shows values in different units as labelled numbers. The drawing
uses only the theme's colours, so it is easy to read in light and dark themes:
any other colour is removed, and a part with no colour uses the text colour.
It cannot run code, embed a web page, animate, or load anything from outside
the page. A block that is not one complete drawing stays code, so a drawing
that is still streaming shows as code until it is complete.
