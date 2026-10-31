const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const newChatButton = document.querySelector("#new-chat");
const uploadButton = document.querySelector("#upload-document");
const documentInput = document.querySelector("#document-input");
const themeButton = document.querySelector("#theme-toggle");
const toolsButton = document.querySelector("#tools-toggle");
const toolsPanel = document.querySelector("#tools-panel");
const toolsDoneButton = document.querySelector("#tools-done");
const settingsButton = document.querySelector("#settings-toggle");
const settingsPanel = document.querySelector("#settings-panel");
const settingsCloseButton = document.querySelector("#settings-close");
const footerCopyright = document.querySelector("#app-copyright");
const footerVersion = document.querySelector("#app-version");

// Editable application identity: change these two values to customize the footer.
const APP_METADATA = {
  copyright: "© 2026 Omar Abu Alnasr. All rights reserved.",
  version: "0.1"
};

const toolInputs = [...document.querySelectorAll("[data-tool]")];
// Keep the popup in the top-level viewport so composer transforms and the welcome
// layout cannot clip it. It remains controlled by the same button and checkboxes.
document.body.append(toolsPanel, settingsPanel);
const messages = document.querySelector("#messages");
const chatLayout = document.querySelector("#chat-layout");
const composerArea = document.querySelector("#composer-area");
const suggestions = document.querySelector("#suggestions");
const emptyState = document.querySelector("#empty-state");
const activity = document.querySelector("#assistant-activity");
const activityLabel = document.querySelector("#activity-label");
const themeMeta = document.querySelector('meta[name="theme-color"]');

let isSending = false;
let documentUploadEnabled = true;
let memoryEnabled = false;
let toolAvailability = {
  calculator: true,
  unit_converter: true,
  date_time: true,
  date_calculator: true,
  scientific_math: true,
  text_analyzer: true,
  json_tools: true,
  web_search: true,
  document_reader: true,
  memory_context: true
};
// Every tool starts disabled. The versioned key intentionally prevents older
// default-on preferences from silently re-enabling tools after this update.
const TOOL_SETTINGS_STORAGE_KEY = "tiny-ai-tool-settings-v2";
const defaultToolSettings = {
  calculator: false,
  unit_converter: false,
  date_time: false,
  date_calculator: false,
  scientific_math: false,
  text_analyzer: false,
  json_tools: false,
  web_search: false,
  document_reader: false,
  memory_context: false
};

function loadToolSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(TOOL_SETTINGS_STORAGE_KEY) || "null");
    if (saved && typeof saved === "object" && !Array.isArray(saved)) {
      return Object.fromEntries(Object.keys(defaultToolSettings).map(key => [key, typeof saved[key] === "boolean" ? saved[key] : defaultToolSettings[key]]));
    }
  } catch { /* Use default tool preferences when browser storage is unavailable. */ }
  return { ...defaultToolSettings };
}

let enabledTools = loadToolSettings();

function createIdentifier() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") return window.crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, character => {
    const random = Math.random() * 16 | 0;
    return (character === "x" ? random : (random & 0x3 | 0x8)).toString(16);
  });
}

function renderAppMetadata() {
  if (footerCopyright) footerCopyright.textContent = APP_METADATA.copyright;
  if (footerVersion) footerVersion.textContent = `Version ${APP_METADATA.version}`;
}

function persistentIdentifier(key) {
  try {
    const existing = localStorage.getItem(key);
    if (existing && /^[0-9a-f-]{36}$/i.test(existing)) return existing;
    const created = createIdentifier();
    localStorage.setItem(key, created);
    return created;
  } catch {
    return createIdentifier();
  }
}

const userId = persistentIdentifier("tiny-ai-user-id");
let conversationId = persistentIdentifier("tiny-ai-conversation-id");

const icons = {
  sun: `<svg class="theme-icon theme-icon-sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.7"></circle><path d="M12 2.5v2M12 19.5v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2.5 12h2m15 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path></svg>`,
  moon: `<svg class="theme-icon theme-icon-moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20.3 15.2A8.4 8.4 0 0 1 8.8 3.7 8.5 8.5 0 1 0 20.3 15.2Z"></path></svg>`
};

function currentTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

async function syncUploadCapability() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) return;
    const health = await response.json();
    documentUploadEnabled = health.document_upload_enabled !== false;
    memoryEnabled = health.memory_enabled === true;
    const available = health.available_tools && typeof health.available_tools === "object" ? health.available_tools : {};
    toolAvailability = {
      calculator: available.calculator !== false,
      unit_converter: available.unit_converter !== false,
      date_time: available.date_time !== false,
      date_calculator: available.date_calculator !== false,
      scientific_math: available.scientific_math !== false,
      text_analyzer: available.text_analyzer !== false,
      json_tools: available.json_tools !== false,
      web_search: available.web_search !== false,
      document_reader: available.document_reader !== false,
      memory_context: memoryEnabled
    };
    updateToolsControl();
    if (!documentUploadEnabled) {
      uploadButton.title = "Document uploads are disabled on this hosted instance";
      uploadButton.setAttribute("aria-label", "Document uploads are disabled");
    }
    if (memoryEnabled) await restoreConversation();
    updateDisclaimer();
  } catch {
    // The chat remains usable when the health check is temporarily unavailable.
  }
}

async function restoreConversation() {
  try {
    const params = new URLSearchParams({ user_id: userId, conversation_id: conversationId });
    const response = await fetch(`/api/conversation?${params.toString()}`, { cache: "no-store" });
    if (!response.ok) return;
    const result = await response.json();
    const history = Array.isArray(result.messages) ? result.messages : [];
    if (!history.length) return;
    setChatMode(true, false);
    history.forEach(turn => {
      if ((turn.role === "user" || turn.role === "assistant") && typeof turn.content === "string") {
        addMessage(turn.role, turn.content);
      }
    });
  } catch {
    // Conversation restore is best-effort; a failed restore never blocks chat.
  }
}

function updateThemeControl() {
  const dark = currentTheme() === "dark";
  const label = dark ? "Switch to light mode" : "Switch to dark mode";
  themeButton.setAttribute("aria-label", label);
  themeButton.title = label;
  themeButton.innerHTML = dark ? icons.sun : icons.moon;
  if (themeMeta) themeMeta.content = dark ? "#050505" : "#ffffff";
}

function saveToolSettings() {
  try { localStorage.setItem(TOOL_SETTINGS_STORAGE_KEY, JSON.stringify(enabledTools)); } catch { /* Preferences still apply for this session. */ }
}

function updateToolsControl() {
  toolsButton.classList.toggle("has-active-tools", Object.keys(enabledTools).some(name => enabledTools[name] && toolAvailability[name] !== false));
  toolsButton.setAttribute("aria-label", "Tools settings");
  toolsButton.title = "Choose which tools are enabled";
  toolInputs.forEach(control => {
    const name = control.dataset.tool;
    control.checked = Boolean(enabledTools[name] && toolAvailability[name] !== false);
    control.disabled = isSending || toolAvailability[name] === false;
    const option = control.closest(".tool-option");
    if (option) option.classList.toggle("is-unavailable", toolAvailability[name] === false);
  });
  uploadButton.disabled = isSending || !documentUploadEnabled || !enabledTools.document_reader;
  uploadButton.title = !documentUploadEnabled
    ? "Document uploads are disabled on this hosted instance"
    : !enabledTools.document_reader ? "Enable Document Reader in the Tools menu to upload files" : "Upload documents";
  uploadButton.setAttribute("aria-label", uploadButton.disabled ? "Document upload unavailable" : "Upload documents");
  updateDisclaimer();
}

function updateDisclaimer() {
  const disclaimer = document.querySelector(".disclaimer");
  if (!disclaimer) return;
  if (!memoryEnabled) {
    disclaimer.textContent = "Persistent memory is disabled on this hosted instance.";
  } else if (enabledTools.memory_context) {
    disclaimer.textContent = "Memory & Context is on. New Chat clears the current conversation but keeps saved memories.";
  } else {
    disclaimer.textContent = "Memory & Context is off for new messages. Enable it in Tools to use saved memories and earlier context.";
  }
}

function positionToolsPanel() {
  if (toolsPanel.hidden) return;

  const viewportWidth = document.documentElement.clientWidth || window.innerWidth;
  const viewportHeight = window.innerHeight;
  const trigger = toolsButton.getBoundingClientRect();
  const maxHeight = Math.max(120, Math.min(620, viewportHeight - 24));
  const panelWidth = Math.min(360, Math.max(280, viewportWidth - 20));

  toolsPanel.style.width = `${panelWidth}px`;
  toolsPanel.style.maxHeight = `${maxHeight}px`;
  toolsPanel.style.right = "auto";
  const left = Math.max(10, Math.min(trigger.right - panelWidth, viewportWidth - panelWidth - 10));
  toolsPanel.style.left = `${left}px`;

  const panelHeight = Math.min(toolsPanel.scrollHeight, maxHeight);
  let top = trigger.top - panelHeight - 10;
  if (top < 10) {
    const below = trigger.bottom + 10;
    top = below + panelHeight <= viewportHeight - 10
      ? below
      : viewportHeight - panelHeight - 10;
  }
  top = Math.max(10, Math.min(top, viewportHeight - panelHeight - 10));
  toolsPanel.style.top = `${top}px`;
}

function setToolsPanel(open) {
  if (open) setSettingsPanel(false);
  toolsPanel.hidden = !open;
  toolsButton.setAttribute("aria-expanded", String(open));
  if (open) {
    positionToolsPanel();
    toolsDoneButton.focus();
  }
}

function positionSettingsPanel() {
  if (settingsPanel.hidden) return;

  const viewportWidth = document.documentElement.clientWidth || window.innerWidth;
  const viewportHeight = window.innerHeight;
  const trigger = settingsButton.getBoundingClientRect();
  const panelWidth = Math.min(286, Math.max(250, viewportWidth - 20));
  const maxHeight = Math.max(120, viewportHeight - 20);

  settingsPanel.style.width = `${panelWidth}px`;
  settingsPanel.style.maxHeight = `${maxHeight}px`;
  settingsPanel.style.right = "auto";
  const left = Math.max(10, Math.min(trigger.right - panelWidth, viewportWidth - panelWidth - 10));
  settingsPanel.style.left = `${left}px`;

  const panelHeight = Math.min(settingsPanel.scrollHeight, maxHeight);
  let top = trigger.top - panelHeight - 9;
  if (top < 10) {
    const below = trigger.bottom + 9;
    top = below + panelHeight <= viewportHeight - 10
      ? below
      : viewportHeight - panelHeight - 10;
  }
  settingsPanel.style.top = `${Math.max(10, Math.min(top, viewportHeight - panelHeight - 10))}px`;
}

function setSettingsPanel(open) {
  if (open) setToolsPanel(false);
  settingsPanel.hidden = !open;
  settingsButton.setAttribute("aria-expanded", String(open));
  if (open) {
    positionSettingsPanel();
    settingsCloseButton.focus();
  }
}

function setTheme(theme, persist = true) {
  document.documentElement.dataset.theme = theme;
  if (persist) {
    try { localStorage.setItem("chat-theme", theme); } catch { /* Theme still changes for this session. */ }
  }
  updateThemeControl();
}

function setChatMode(isChatting, animate = true) {
  const alreadyChatting = chatLayout.classList.contains("is-chatting");
  if (alreadyChatting === isChatting) return;

  const before = composerArea.getBoundingClientRect();
  chatLayout.classList.toggle("is-chatting", isChatting);
  chatLayout.classList.toggle("is-empty", !isChatting);
  emptyState.hidden = isChatting;
  suggestions.hidden = isChatting;

  if (!animate || !before.width || typeof composerArea.animate !== "function") return;
  const after = composerArea.getBoundingClientRect();
  const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
  if (reducedMotion || !after.width) return;

  const deltaX = before.left - after.left;
  const deltaY = before.top - after.top;
  const scaleX = before.width / after.width;
  const scaleY = before.height / after.height;
  composerArea.animate([
    { transformOrigin: "top left", transform: `translate(${deltaX}px, ${deltaY}px) scale(${scaleX}, ${scaleY})`, opacity: .94 },
    { transformOrigin: "top left", transform: "translate(0, 0) scale(1, 1)", opacity: 1 }
  ], { duration: 540, easing: "cubic-bezier(.2,.78,.2,1)" });
}

function scrollToLatest() {
  // Scroll only the transcript. The viewport and composer remain fixed.
  window.requestAnimationFrame(() => {
    messages.scrollTo({ top: messages.scrollHeight, behavior: "smooth" });
  });
}

function setActivity(visible, label = "Thinking") {
  activity.hidden = !visible;
  activityLabel.textContent = label;
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      // Fall back to the selection-based copy path when clipboard permission is denied.
    }
  }

  const temporary = document.createElement("textarea");
  temporary.value = text;
  temporary.setAttribute("readonly", "");
  temporary.style.position = "fixed";
  temporary.style.opacity = "0";
  temporary.style.pointerEvents = "none";
  document.body.append(temporary);
  temporary.select();
  temporary.setSelectionRange(0, temporary.value.length);
  const copied = document.execCommand("copy");
  temporary.remove();
  if (!copied) throw new Error("Clipboard access is unavailable.");
}

function createMessageActions(text) {
  const actions = document.createElement("div");
  actions.className = "message-actions";
  actions.setAttribute("aria-label", "Assistant message actions");

  const copyButton = document.createElement("button");
  copyButton.className = "message-action-button";
  copyButton.type = "button";
  copyButton.setAttribute("aria-label", "Copy response");
  copyButton.title = "Copy response";
  copyButton.innerHTML = '<i data-lucide="copy" aria-hidden="true"></i><span>Copy</span>';
  copyButton.addEventListener("click", async () => {
    if (copyButton.disabled) return;
    const originalMarkup = '<i data-lucide="copy" aria-hidden="true"></i><span>Copy</span>';
    try {
      await copyTextToClipboard(text);
      copyButton.innerHTML = '<i data-lucide="check" aria-hidden="true"></i><span>Copied</span>';
      copyButton.setAttribute("aria-label", "Response copied");
      copyButton.title = "Copied";
      copyButton.classList.add("is-copied");
      if (window.lucide && typeof window.lucide.createIcons === "function") {
        window.lucide.createIcons({ root: copyButton, attrs: { "stroke-width": 1.7 } });
      }
      window.setTimeout(() => {
        if (!copyButton.isConnected) return;
        copyButton.innerHTML = originalMarkup;
        copyButton.setAttribute("aria-label", "Copy response");
        copyButton.title = "Copy response";
        copyButton.classList.remove("is-copied");
        if (window.lucide && typeof window.lucide.createIcons === "function") {
          window.lucide.createIcons({ root: copyButton, attrs: { "stroke-width": 1.7 } });
        }
      }, 1500);
    } catch {
      copyButton.innerHTML = '<i data-lucide="copy" aria-hidden="true"></i><span>Copy unavailable</span>';
      copyButton.setAttribute("aria-label", "Copy unavailable");
      copyButton.title = "Clipboard access is unavailable";
      if (window.lucide && typeof window.lucide.createIcons === "function") {
        window.lucide.createIcons({ root: copyButton, attrs: { "stroke-width": 1.7 } });
      }
    }
  });

  // Feedback is intentionally quiet and local to this response: selecting one
  // option clears the other, with no toast, modal, or confirmation message.
  let selectedFeedback = null;
  const makeFeedbackButton = (kind, label, iconName) => {
    const button = document.createElement("button");
    button.className = "message-action-button message-feedback-button";
    button.type = "button";
    button.dataset.feedback = kind;
    button.setAttribute("aria-label", `${label} response`);
    button.setAttribute("aria-pressed", "false");
    button.title = `${label} response`;
    button.innerHTML = `<i data-lucide="${iconName}" aria-hidden="true"></i><span>${label}</span>`;
    button.addEventListener("click", () => {
      // Clicking the active choice clears it; choosing the other switches the
      // selection, so Like and Dislike can never be active simultaneously.
      selectedFeedback = selectedFeedback === kind ? null : kind;
      [likeButton, dislikeButton].forEach(candidate => {
        const active = candidate.dataset.feedback === selectedFeedback;
        candidate.classList.toggle("is-selected", active);
        candidate.setAttribute("aria-pressed", String(active));
      });
    });
    return button;
  };
  const likeButton = makeFeedbackButton("like", "Like", "thumbs-up");
  const dislikeButton = makeFeedbackButton("dislike", "Dislike", "thumbs-down");

  const readButton = document.createElement("button");
  readButton.className = "message-action-button is-future-action";
  readButton.type = "button";
  readButton.disabled = true;
  readButton.setAttribute("aria-label", "Read aloud — coming later");
  readButton.title = "Read aloud — coming later";
  readButton.innerHTML = '<i data-lucide="volume-2" aria-hidden="true"></i><span>Read aloud</span>';

  const retryButton = document.createElement("button");
  retryButton.className = "message-action-button is-future-action";
  retryButton.type = "button";
  retryButton.disabled = true;
  retryButton.setAttribute("aria-label", "Try again — coming later");
  retryButton.title = "Try again — coming later";
  retryButton.innerHTML = '<i data-lucide="refresh-cw" aria-hidden="true"></i><span>Try again</span>';

  actions.append(copyButton, likeButton, dislikeButton, readButton, retryButton);
  if (window.lucide && typeof window.lucide.createIcons === "function") {
    window.lucide.createIcons({ root: actions, attrs: { "stroke-width": 1.7 } });
  }
  return actions;
}

function addMessage(role, text, meta = "", sources = []) {
  if (role === "user" && chatLayout.classList.contains("is-empty")) setChatMode(true, true);
  const row = document.createElement("article");
  row.className = `message-row ${role}`;
  row.setAttribute("aria-label", role === "user" ? "Your message" : "Reply");

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;
  row.append(bubble);
  if (role === "assistant") row.append(createMessageActions(text));

  if (meta) {
    const details = document.createElement("div");
    details.className = "message-meta";
    details.textContent = meta;
    row.append(details);
  }

  if (role === "assistant" && Array.isArray(sources) && sources.length) {
    const disclosure = document.createElement("details");
    disclosure.className = "sources-disclosure";
    const summary = document.createElement("summary");
    summary.className = "sources-summary";
    const sourceGroup = document.createElement("section");
    sourceGroup.className = "message-sources";
    sourceGroup.setAttribute("aria-label", "Answer sources");
    let sourceCount = 0;

    sources.slice(0, 8).forEach(source => {
      if (!source || typeof source !== "object") return;
      const isDocument = source.source_type === "document";
      let card;
      let hostname = "";
      if (isDocument) {
        card = document.createElement("div");
        card.className = "source-card document-source-card";
        const page = source.page ? `Page ${source.page}` : (source.section || "Uploaded document");
        hostname = page;
      } else {
        if (typeof source.url !== "string") return;
        let parsed;
        try { parsed = new URL(source.url); } catch { return; }
        if (parsed.protocol !== "https:" && parsed.protocol !== "http:") return;
        card = document.createElement("a");
        card.className = "source-card";
        card.href = parsed.href;
        card.target = "_blank";
        card.rel = "noopener noreferrer";
        hostname = source.publisher || source.domain || parsed.hostname;
      }
      const title = document.createElement("span");
      title.className = "source-title";
      title.textContent = source.title || hostname;
      const domain = document.createElement("span");
      domain.className = "source-domain";
      domain.textContent = hostname;
      card.append(title, domain);
      if (isDocument && typeof source.excerpt === "string" && source.excerpt.trim()) {
        const excerpt = document.createElement("span");
        excerpt.className = "source-meta document-source-excerpt";
        excerpt.textContent = source.excerpt;
        card.append(excerpt);
      } else if (source.published_at) {
        const metadata = document.createElement("span");
        metadata.className = "source-meta";
        const published = new Date(source.published_at);
        metadata.textContent = Number.isNaN(published.getTime())
          ? source.published_at
          : published.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
        card.append(metadata);
      }
      sourceGroup.append(card);
      sourceCount += 1;
    });

    if (sourceCount) {
      summary.textContent = `Sources · ${sourceCount}`;
      disclosure.append(summary, sourceGroup);
      row.append(disclosure);
    }
  }

  messages.append(row);
  scrollToLatest();
  return row;
}

function setSending(value) {
  isSending = value;
  sendButton.disabled = value;
  newChatButton.disabled = value;
  uploadButton.disabled = value || !documentUploadEnabled;
  toolsButton.disabled = value;
  toolInputs.forEach(control => { control.disabled = value || toolAvailability[control.dataset.tool] === false; });
  input.disabled = value;
  updateToolsControl();
  sendButton.setAttribute("aria-busy", String(value));
  form.setAttribute("aria-busy", String(value));
}

async function sendMessage(rawText) {
  const text = rawText.trim();
  if (!text || isSending) return;

  emptyState.hidden = true;
  addMessage("user", text);
  input.value = "";
  input.style.height = "auto";
  setSending(true);
  setActivity(true, enabledTools.web_search ? "Searching the web" : "Thinking");

  const requestTools = Object.fromEntries(Object.keys(enabledTools).map(name => [name, Boolean(enabledTools[name] && toolAvailability[name] !== false)]));
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        web_search: requestTools.web_search,
        enabled_tools: requestTools,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || null,
        user_id: userId,
        conversation_id: conversationId
      })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The request could not be completed.");

    const meta = result.source === "calculator" ? "Calculator"
      : result.source === "tool" && result.tool === "date_time" ? "Date & time"
      : result.source === "tool" && result.tool === "unit_converter" ? "Unit converter"
      : result.source === "tool" && result.tool === "date_calculator" ? "Date calculator"
      : result.source === "tool" && result.tool === "scientific_math" ? "Scientific math"
      : result.source === "tool" && result.tool === "text_analyzer" ? "Text analyzer"
      : result.source === "tool" && result.tool === "json_tools" ? "JSON tools"
      : result.source === "learning" ? "Saved to knowledge"
      : result.source === "document" ? "Document reader"
      : result.source === "web" ? ""
      : result.source === "search_setup" ? "Web search setup required"
      : result.source === "web_search_disabled" ? "Enable Web Search in Tools to search online"
      : result.source === "search_rate_limited" ? "Search limit reached"
      : result.source === "web_error" ? "Search unavailable"
      : result.source === "memory" ? "Memory updated"
      : result.source === "conversation" ? "Conversation context"
      : result.context_used ? `Using context · ${result.context_topic || "earlier messages"}`
      : result.memory_saved ? "Saved to memory"
      : "";
    addMessage("assistant", result.answer, meta, result.sources || []);
  } catch (error) {
    addMessage("assistant", error.message || "I couldn't connect. Check that the server is running.", "Connection error");
  } finally {
    setSending(false);
    setActivity(false);
    input.focus();
  }
}

async function uploadDocuments(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length || isSending || !enabledTools.document_reader || !documentUploadEnabled) return;
  if (files.length > 5) {
    addMessage("assistant", "Please upload no more than five documents at a time.", "Upload limit");
    documentInput.value = "";
    return;
  }

  const names = files.map(file => file.name);
  addMessage("user", `Upload ${files.length === 1 ? "document" : "documents"}: ${names.join(", ")}`);
  setSending(true);
  setActivity(true, "Reading document" + (files.length > 1 ? "s" : ""));
  const body = new FormData();
  body.append("user_id", userId);
  body.append("conversation_id", conversationId);
  const requestTools = Object.fromEntries(Object.keys(enabledTools).map(name => [name, Boolean(enabledTools[name] && toolAvailability[name] !== false)]));
  body.append("enabled_tools", JSON.stringify(requestTools));
  files.forEach(file => body.append("files", file, file.name));

  try {
    const response = await fetch("/api/upload", { method: "POST", body });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The document could not be processed.");
    const uploaded = Array.isArray(result.documents) ? result.documents : [];
    const sourceCards = uploaded.map(document => ({
      source_type: "document",
      title: document.filename,
      publisher: `${document.chunk_count} indexed sections · ${document.word_count} words`,
      excerpt: document.topics?.length ? `Topics: ${document.topics.join(", ")}` : "Text extracted and indexed for question answering."
    }));
    addMessage("assistant", result.answer, `${uploaded.length} document${uploaded.length === 1 ? "" : "s"} added to knowledge`, sourceCards);
  } catch (error) {
    addMessage("assistant", error.message || "I couldn't read that document. Please try another file.", "Document upload failed");
  } finally {
    documentInput.value = "";
    setSending(false);
    setActivity(false);
    input.focus();
  }
}

function resetChat() {
  if (isSending) return;
  conversationId = createIdentifier();
  try { localStorage.setItem("tiny-ai-conversation-id", conversationId); } catch { /* New chat still starts for this page. */ }
  messages.replaceChildren();
  setChatMode(false, true);
  emptyState.classList.remove("welcome-reenter");
  // Restart the welcome animation when returning to a fresh conversation.
  void emptyState.offsetWidth;
  emptyState.classList.add("welcome-reenter");
  input.value = "";
  input.style.height = "auto";
  setActivity(false);
  input.focus();
}

form.addEventListener("submit", event => {
  event.preventDefault();
  sendMessage(input.value);
});

input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
  if (!isSending) setActivity(Boolean(input.value.trim()), "You're typing");
});

document.querySelectorAll("[data-prompt]").forEach(button => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

newChatButton.addEventListener("click", resetChat);
uploadButton.addEventListener("click", () => { if (!isSending && documentUploadEnabled && enabledTools.document_reader) documentInput.click(); });
documentInput.addEventListener("change", () => uploadDocuments(documentInput.files));
toolsButton.addEventListener("click", () => {
  if (isSending) return;
  setToolsPanel(toolsPanel.hidden);
});

settingsButton.addEventListener("click", () => {
  setSettingsPanel(settingsPanel.hidden);
});

settingsCloseButton.addEventListener("click", () => {
  setSettingsPanel(false);
  settingsButton.focus();
});

toolsDoneButton.addEventListener("click", () => {
  setToolsPanel(false);
  input.focus();
});

toolInputs.forEach(control => {
  control.addEventListener("change", () => {
    if (isSending) return;
    enabledTools[control.dataset.tool] = control.checked;
    saveToolSettings();
    updateToolsControl();
  });
});

document.addEventListener("pointerdown", event => {
  if (!toolsPanel.hidden && !toolsPanel.contains(event.target) && !toolsButton.contains(event.target)) {
    setToolsPanel(false);
  }
  if (!settingsPanel.hidden && !settingsPanel.contains(event.target) && !settingsButton.contains(event.target)) {
    setSettingsPanel(false);
  }
});

window.addEventListener("resize", positionToolsPanel);
window.addEventListener("scroll", positionToolsPanel, true);
window.visualViewport?.addEventListener("resize", positionToolsPanel);
window.visualViewport?.addEventListener("scroll", positionToolsPanel);
window.addEventListener("resize", positionSettingsPanel);
window.addEventListener("scroll", positionSettingsPanel, true);
window.visualViewport?.addEventListener("resize", positionSettingsPanel);
window.visualViewport?.addEventListener("scroll", positionSettingsPanel);

document.addEventListener("keydown", event => {
  if (event.key !== "Escape") return;
  if (!settingsPanel.hidden) {
    setSettingsPanel(false);
    settingsButton.focus();
  } else if (!toolsPanel.hidden) {
    setToolsPanel(false);
    toolsButton.focus();
  }
});
themeButton.addEventListener("click", () => setTheme(currentTheme() === "dark" ? "light" : "dark"));

document.addEventListener("keydown", event => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    resetChat();
  }
});

renderAppMetadata();
updateThemeControl();
if (window.lucide && typeof window.lucide.createIcons === "function") {
  window.lucide.createIcons({ attrs: { "stroke-width": 1.7 } });
}
updateToolsControl();
syncUploadCapability();
