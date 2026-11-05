A dock layout report from the web terminal that does not reach the server is
now sent again. A report with no answer, a network error or a server error
goes out again after a wait that doubles up to 30 s. A report that the server
refuses goes to the browser console with the reason, and goes out again only
after the layout changes. Before, a lost report went out again only at the
next layout change, and until then the agent read an old layout.
