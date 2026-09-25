A structured reply that carries a raw control character, such as a newline,
inside a JSON string now parses on every provider instead of failing the call.
A reply that still does not parse is asked for once more, with the same
request, before the error is raised.
