Theming
========

Every OSPREY browser interface — the terminal and all of its panels — draws its
colors and fonts from one shared design system. Pick a theme once and everything
matches automatically, in light or dark. You can also design your own.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item::

      .. image:: /_static/screenshots/web_terminal_hero_light.png
         :alt: The Web Terminal in the light Osprey theme
         :width: 100%

   .. grid-item::

      .. image:: /_static/screenshots/web_terminal_hero_dark.png
         :alt: The Web Terminal in the dark Osprey theme
         :width: 100%

Choosing a theme
----------------

Themes come in **families**. OSPREY ships four:

- **main** — the default look, in light and dark.
- **desy** — the DESY corporate palette, in light and dark.
- **high-contrast** — a stronger-contrast family for accessibility, also in
  light and dark.
- **retro** — a navy-and-teal family, in light and dark, for teams who
  prefer that look.

In the terminal, click the sliders button at the top right to open the
display menu — it holds the light/dark switch, the Expert/Simple view toggle,
and the theme family picker, as pictured below. Every panel opened as its own
page (ARIEL, the workspace, Channel Finder, the lattice dashboard, the
knowledge panel) has the same button and the same display controls in its
own header.

A pick made on any of those pages is remembered by your browser and becomes
your preference for every OSPREY page served from the same address — set a
theme in ARIEL and the terminal comes up in it the next time you load it.
Until you pick Light or Dark, pages follow your operating system's light/dark
preference.

Panels shown inside the terminal are a separate case. The terminal hands each
one its theme and view in the page address, and an address outranks the
remembered preference, so an embedded panel always matches the terminal
around it.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item::

      .. image:: /_static/screenshots/display_menu_light.png
         :alt: The open display menu in the light Osprey theme
         :width: 100%

   .. grid-item::

      .. image:: /_static/screenshots/display_menu_dark.png
         :alt: The open display menu in the dark Osprey theme
         :width: 100%

The display menu a standalone panel opens from its header, light and dark.
Captured with OSPREY |captured_display_menu|.

To set the theme a deployment *starts* in, use ``web.theme`` in ``config.yml``:

.. code-block:: yaml

   web:
     theme: main        # a family, or a specific theme like high-contrast-light

Name a family to start visitors in that family (light or dark then follows
their operating-system preference), or name a specific theme to pin an exact
look — including whether the deployment starts light or dark. Whatever you pick
in the browser always wins over this and sticks across reloads.

.. note::

   ``web.theme`` (the browser interfaces) is separate from ``cli.theme`` (the
   colors of OSPREY's plain terminal output). They never affect each other.

Giving each user their own default
----------------------------------

In a multi-user deployment every user gets their own terminal, but they all run
the same image — so they all read the same ``web.theme``. To start a particular
user somewhere else, add ``theme`` to their entry in the user list:

.. code-block:: yaml

   modules:
     web_terminals:
       users:
         - alice
         - name: bob
           index: 1
           theme: desy-light

The value takes the same two forms as ``web.theme``: a family, or a specific
theme to also pin light or dark. It applies to that user only, and their own
pick in the display menu still wins over it.

The landing page that lists everyone's terminals uses the deployment-wide
``web.theme``. It is shown before anyone has said who they are, so there is no
personal setting to apply yet.

A navy-and-teal terminal
------------------------

Two settings in ``config.yml`` give the terminal a navy-and-teal look with its
panel buttons along the top:

.. code-block:: yaml

   web:
     theme: retro            # the navy/teal palette
     rail_position: top      # panel buttons along the top

Each user can still override both from the interface: the theme from the
display menu in the header, and the rail position from the panel "+" menu.

Propose a theme
---------------

Have a color of your own in mind? Run:

.. code-block:: bash

   osprey theme-lab

That opens the Theme Lab in your browser. Pick an accent color and the dark and
light mock-ups of the terminal re-skin as you go, while the contrast badges tell
you at a glance whether text stays readable against it — the same check every
shipped theme has to pass. When you are happy, give the theme a name, copy the
export block, and paste it into a GitHub issue. The lab doesn't create the theme
itself; the export spells out exactly what needs to change to make it real.

A theme has two accents: the main one, and a second used for highlights and
warnings. Switch between them with the **Accent** / **Second accent** buttons
above the color wheel — one set of controls edits whichever is selected, and
both previews update either way. The second accent gets a contrast badge of its
own, because the build holds it to the stricter body-text standard: it is used
for readable text, where the main accent mostly is not. If that badge goes red,
drag the lightness slider for the mode it failed in until it clears.

.. dropdown:: Going deeper — the design system
   :icon: paintbrush

   The colors above aren't hand-written CSS scattered across each interface —
   they come from one small, machine-checked token system. You only need this if
   you are adding a theme or a new interface; the steps below and the design
   system's source are the real reference. The tabs are the rough idea.

   .. tab-set::

      .. tab-item:: What a theme is

         A theme is a single JSON document listing named colors — backgrounds,
         text, accents, status colors, the terminal palette. Every theme defines
         the *same* set of names, so they are interchangeable, and a build step
         **checks each one's contrast automatically** — more strictly for the
         high-contrast family. A theme that reads fine but fails the check does
         not ship until its colors are nudged; the gate is never loosened.

      .. tab-item:: Author a theme

         Copy an existing theme's JSON as a starting point, adjust its color
         values (and its name/family label), then run the generator:

         .. code-block:: bash

            python -m osprey.interfaces.design_system.generator.build

         It validates the whole set and rewrites the compiled CSS/JS. If a color
         is missing, a name doesn't match its siblings, or a contrast gate fails,
         the build stops and tells you exactly where. Fix and re-run until clean,
         then commit the regenerated files alongside your theme.

      .. tab-item:: How it fits together

         Themes are authored as JSON and compiled into the CSS and JavaScript
         every interface loads. The compiled files are checked in, not built on
         deploy, so you regenerate and commit them whenever you change a color. A
         new interface opts in by loading those files and following the same
         theme boot every OSPREY page uses.

.. seealso::

   :doc:`panels`
      Panels use these same tokens, so they theme themselves for free.
