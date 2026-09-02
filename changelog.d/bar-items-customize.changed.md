Customize bars now shows each item as it will look in the bar, filed under
headings, dims the items that are already placed instead of offering a second,
empty copy, and labels the two bars while you edit. The right-click menu is
**Customize bars**, plus **Hide header** or **Hide status bar** on that bar
itself (`web.bar_items.header_visible` is the deployment's side); the one
preset is **Default**, which returns to the arrangement `web.bar_items`
configures. A hidden bar comes back from the right-click menu of any panel's
title bar, which offers **Show header** or **Show status bar** while it is
hidden. The fixed gap is gone: drag either end of a space to give it a
width, or leave it at 0 to fill the bar. The clock is plain by default, with
no zone name beside it, and can switch between 24-hour and 12-hour time. The
Feedback item now opens the feedback dialog.
Every item can be removed, the wordmark, identity, control-target chip and
display menu included; the `locked` key under `web.bar_items` is gone. The
command palette's **Log out** and **Switch to … mode** no longer need the
header items that also offer them.
