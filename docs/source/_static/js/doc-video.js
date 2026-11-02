// The player for every docs video recorded by docs/screenshots (see videos.py).
//
// Each ``.osprey-demo`` block holds a <video> whose <source> names the light
// take, ``<prefix>-light.mp4``, and whose poster is ``<prefix>-light-poster.jpg``;
// that markup alone plays without JavaScript. This script swaps both files to
// the dark take with the docs theme, falls back to the poster alone when the
// video cannot play, and wires the speed buttons. The video's stem is read
// from the <source> path, so a page embeds a new video with markup only.
(function () {
  function wire(box) {
    var video = box.querySelector("video");
    var source = video.querySelector("source");
    var note = box.querySelector(".osprey-demo-note");
    var speed = box.querySelector(".osprey-demo-speed");
    var buttons = speed.querySelectorAll("button");
    var prefix = source.getAttribute("src").replace(/-light\.mp4$/, "");
    var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    var still = null;
    function file(theme, poster) {
      return prefix + "-" + theme + (poster ? "-poster.jpg" : ".mp4");
    }
    // The video cannot play (missing, blocked or unsupported): the poster alone.
    function posterOnly() {
      if (still) return;
      still = document.createElement("img");
      still.className = "osprey-demo-poster";
      still.alt = video.querySelector("img").alt;
      still.src = video.getAttribute("poster");
      video.pause();
      video.controls = false;
      video.removeAttribute("controls");
      video.hidden = true;
      note.hidden = true;
      speed.hidden = true;
      video.parentNode.insertBefore(still, video);
    }
    source.addEventListener("error", posterOnly);
    video.addEventListener("error", posterOnly);
    function markRate(rate) {
      buttons.forEach(function (b) {
        b.setAttribute("aria-pressed", String(Number(b.dataset.rate) === rate));
      });
    }
    function setRate(rate) {
      // load() on a theme change resets playbackRate to defaultPlaybackRate,
      // so the chosen speed is kept in both.
      video.defaultPlaybackRate = rate;
      video.playbackRate = rate;
      markRate(rate);
    }
    buttons.forEach(function (b) {
      b.addEventListener("click", function () { setRate(Number(b.dataset.rate)); });
    });
    video.addEventListener("ratechange", function () { markRate(video.playbackRate); });
    setRate(1);
    speed.hidden = false;
    var root = document.documentElement;
    var dark = window.matchMedia("(prefers-color-scheme: dark)");
    var current = null;
    function update() {
      var t = root.dataset.theme;
      var theme = t === "dark" || t === "light" ? t : (dark.matches ? "dark" : "light");
      if (theme === current) return;
      current = theme;
      video.setAttribute("poster", file(theme, true));
      if (still) {
        still.src = file(theme, true);
        return;
      }
      source.src = file(theme, false);
      video.load();
      if (!reduceMotion.matches) {
        var playing = video.play();
        if (playing) playing.catch(function () {});
      }
    }
    update();
    new MutationObserver(update).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
    dark.addEventListener("change", update);
  }
  // A block is wired once, by the script tag that follows it.
  document.querySelectorAll(".osprey-demo:not([data-wired])").forEach(function (box) {
    box.setAttribute("data-wired", "");
    wire(box);
  });
})();
