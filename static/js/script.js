// script.js
// Small progressive-enhancement helpers for the AP Welfare Fraud Audit dashboard.
// Charts themselves are initialized inline on each page (they need server-rendered
// data); this file handles the splash screen and other generic UI behaviour
// shared across pages.

document.addEventListener("DOMContentLoaded", () => {
  // ------------------------------------------------------------------
  // Splash screen: show once per browser session (sessionStorage),
  // then fade it out. Reopening the app in a new tab/session shows it
  // again; navigating between pages in the same tab does not repeat it.
  // ------------------------------------------------------------------
  const splash = document.getElementById("splashScreen");
  if (splash) {
    const alreadySeen = window.sessionStorage.getItem("ap_splash_seen");
    if (alreadySeen) {
      splash.classList.add("is-hidden");
    } else {
      const MIN_DISPLAY_MS = 1600;
      window.setTimeout(() => {
        splash.classList.add("is-hidden");
        window.sessionStorage.setItem("ap_splash_seen", "1");
      }, MIN_DISPLAY_MS);
    }

    // Allow clicking the splash screen to dismiss it immediately.
    splash.addEventListener("click", () => {
      splash.classList.add("is-hidden");
      window.sessionStorage.setItem("ap_splash_seen", "1");
    });
  }

  // Highlight the active nav link based on current path as a fallback
  // in case the Jinja "active_page" flag was not set on a given page.
  const path = window.location.pathname;
  document.querySelectorAll(".mainnav a").forEach((link) => {
    if (link.getAttribute("href") === path) {
      link.classList.add("is-active");
    }
  });

  // Basic client-side sanity check on the prediction form before submit,
  // so users get instant feedback instead of waiting for a server round-trip.
  const form = document.querySelector(".ledger-form");
  if (form) {
    form.addEventListener("submit", (event) => {
      const age = form.querySelector('input[name="Age"]');
      if (age && (Number(age.value) < 0 || Number(age.value) > 110)) {
        event.preventDefault();
        alert("Please enter a valid age between 0 and 110.");
      }
    });
  }
});
