Bokeh dashboards built with inline resources render in the artifact gallery
again instead of showing their JavaScript as page text. The gallery's theme
snippet now goes before the page's real `</head>`, not before a `</head>` that
appears in a string inside an inlined library such as `bokeh.min.js`.
