The web terminal's session picker now escapes each session's id where it
writes it into the dropdown, as it already did for the preview text, so an id
containing markup can no longer inject HTML into the page.
