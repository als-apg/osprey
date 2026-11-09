ARIEL: the web interface composes in a sheet over the list (``c``; **Expand**
spans the panel), autosaved to the browser only, with a line saying where
Submit goes. **Follow up** (``r``) links a new entry to the selected one via
``metadata.follows_up``, and the reader shows the thread from the new
``/api/entries/{entry_id}/thread``. The **Drafts** mailbox lists agent-staged
drafts (``GET /api/drafts``) and **Discard** deletes one
(``DELETE /api/drafts/{draft_id}``). Entry creation returns
``attachments_published``; ``entry_publish`` carries ``follows_up``. Run
``osprey ariel migrate`` to index follow-ups.
