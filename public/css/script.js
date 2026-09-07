// Minimal progressive-enhancement JS: shows a scanning state on submit
// so the user gets feedback while the server extracts features and runs
// the model. No frameworks, no build step.

document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector(".check-form");
  if (!form) return;

  form.addEventListener("submit", () => {
    const button = form.querySelector("button[type='submit']");
    if (!button || button.disabled) return;
    button.dataset.originalText = button.textContent;
    button.textContent = "Scanning…";
    button.disabled = true;
  });
});
