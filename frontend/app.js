const composer = document.querySelector("[data-composer]");
const messageInput = document.querySelector("[data-message-input]");
const sendButton = document.querySelector("[data-send]");
const characterCount = document.querySelector("[data-character-count]");
const conversationLog = document.querySelector("#conversation-log");
const conversationEnd = document.querySelector("[data-conversation-end]");
const sessionStatus = document.querySelector("[data-session-status]");
const sessionChannel = document.querySelector("[data-session-channel]");
const sessionBusiness = document.querySelector("[data-session-business]");
const sessionIntent = document.querySelector("[data-session-intent]");
const announcement = document.querySelector("[data-announcement]");
const resetButton = document.querySelector("[data-reset]");
const profileForm = document.querySelector("[data-profile-form]");
const profileSubmitButton = profileForm.querySelector(".profile-form__submit");
const businessTypeSelect = document.querySelector("[data-business-type]");
const businessNameInput = document.querySelector("[data-business-name]");
const quickActionButtons = document.querySelectorAll("[data-prompt]");
const voiceInputButton = document.querySelector("[data-voice-input]");
const speakLatestButton = document.querySelector("[data-speak-latest]");
const voiceStatus = document.querySelector("[data-voice-status]");
const contextBusiness = document.querySelector("[data-context-business]");
const contextService = document.querySelector("[data-context-service]");
const contextAppointment = document.querySelector("[data-context-appointment]");
const initialAssistantText = document.querySelector("[data-initial-assistant-text]");
const railStatus = document.querySelector("[data-rail-status]");
const railBusiness = document.querySelector("[data-rail-business]");
const railProfileType = document.querySelector("[data-rail-profile-type]");
const railProfileState = document.querySelector("[data-rail-profile-state]");
const railIntent = document.querySelector("[data-rail-intent]");
const railChannel = document.querySelector("[data-rail-channel]");
const railValidation = document.querySelector("[data-rail-validation]");
const railService = document.querySelector("[data-rail-service]");
const railAppointment = document.querySelector("[data-rail-appointment]");
const railServices = document.querySelector("[data-rail-services]");

const MAX_MESSAGE_LENGTH = Number(messageInput.maxLength);
const RESPONSE_PROGRESS_STEPS = [
  {
    title: "Reading conversation context",
    detail: "Checking recent messages and remembered appointment details.",
  },
  {
    title: "Checking business profile",
    detail: "Using the selected demo business and supported services.",
  },
  {
    title: "Reviewing appointment rules",
    detail: "Looking at services, dates, times, and handoff context.",
  },
  {
    title: "Preparing Aster's reply",
    detail: "Turning the result into a concise reception response.",
  },
];
const BUSINESS_PROFILES = {
  dental: {
    label: "Dental Clinic",
    exampleName: "BrightSmile Dental",
    services: [
      "dental cleaning",
      "dental exam",
      "teeth whitening",
      "filling",
      "emergency dental visit",
    ],
  },
  salon: {
    label: "Salon",
    exampleName: "Luxe Hair Studio",
    services: ["haircut", "blowout", "hair color", "manicure", "facial"],
  },
  auto_repair: {
    label: "Auto Repair Shop",
    exampleName: "TurboFix Garage",
    services: [
      "oil change",
      "brake inspection",
      "tire rotation",
      "battery diagnostic",
      "engine diagnostic",
    ],
  },
};
const SpeechRecognitionConstructor =
  window.SpeechRecognition || window.webkitSpeechRecognition;

let recognition = null;
let isListening = false;
let isRecognitionStarting = false;
let isRecognitionStopping = false;
let suppressRecognitionFeedback = false;
let recognitionBaseText = "";
let currentTranscript = "";
let recognitionErrorMessage = "";
let activeUtterance = null;
let isSubmitting = false;
let isProfileReady = false;
let requestGeneration = 0;
let sessionId = createSessionId();
let activeBusinessType = "";
let activeBusinessName = "";
let responseProgressMessage = null;
let responseProgressTimer = null;
let responseProgressStepIndex = 0;

const supportsSpeechSynthesis =
  typeof window.speechSynthesis !== "undefined" &&
  typeof window.SpeechSynthesisUtterance === "function";

function resizeMessageInput() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 144)}px`;
}

function appendConversationMessage(messageElement) {
  if (conversationEnd) {
    conversationLog.insertBefore(messageElement, conversationEnd);
    return;
  }

  conversationLog.append(messageElement);
}

function scrollConversationToEnd(behavior = "smooth") {
  const scrollToBottom = (scrollBehavior = behavior) => {
    conversationLog.scrollTo({
      top: conversationLog.scrollHeight,
      behavior: scrollBehavior,
    });
  };

  requestAnimationFrame(() => {
    scrollToBottom();
    requestAnimationFrame(() => scrollToBottom("auto"));
  });
}

function updateComposerState() {
  const messageLength = messageInput.value.length;
  const hasMessage = messageInput.value.trim().length > 0;

  messageInput.disabled = !isProfileReady;
  messageInput.placeholder = isProfileReady
    ? "Message the reception desk"
    : "Select business type and enter business name first";
  sendButton.disabled =
    !isProfileReady ||
    !hasMessage ||
    isListening ||
    isRecognitionStarting ||
    isSubmitting;
  characterCount.textContent = `${messageLength} / ${MAX_MESSAGE_LENGTH}`;
  characterCount.classList.toggle("is-near-limit", messageLength >= 450);
  resizeMessageInput();
}

function defaultVoiceStatus() {
  if (recognition && supportsSpeechSynthesis) {
    return "Voice ready";
  }

  if (recognition) {
    return "Voice input ready";
  }

  if (supportsSpeechSynthesis) {
    return "Read-aloud ready";
  }

  return "Voice unavailable";
}

function setVoiceStatus(message, isActive = false) {
  voiceStatus.textContent = message;
  voiceStatus.classList.toggle("is-active", isActive);
}

function updateVoiceControls() {
  voiceInputButton.disabled =
    !isProfileReady ||
    !recognition ||
    isRecognitionStarting ||
    isRecognitionStopping ||
    isSubmitting;
  speakLatestButton.disabled =
    !isProfileReady ||
    !supportsSpeechSynthesis ||
    isListening ||
    isRecognitionStarting ||
    isSubmitting;
}

function setListeningState(listening) {
  isListening = listening;
  voiceInputButton.classList.toggle("is-listening", listening);
  voiceInputButton.setAttribute("aria-pressed", String(listening));
  voiceInputButton.setAttribute(
    "aria-label",
    listening ? "Stop voice input" : "Start voice input",
  );
  voiceInputButton.title = listening ? "Stop voice input" : "Start voice input";
  updateComposerState();
  updateVoiceControls();
}

function formatCurrentTime() {
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date());
}

function selectedProfile() {
  return BUSINESS_PROFILES[businessTypeSelect.value] || null;
}

function cleanBusinessName(value) {
  const cleaned = value.replace(/[\u0000-\u001F\u007F]/g, " ").replace(/\s+/g, " ").trim();
  return cleaned.slice(0, 80);
}

function setText(element, value) {
  if (element) {
    element.textContent = value;
  }
}

function renderServiceMenu() {
  if (!railServices) {
    return;
  }

  const services = BUSINESS_PROFILES[activeBusinessType]?.services || [
    "Select a business type",
  ];

  railServices.replaceChildren(
    ...services.map((service) => {
      const item = document.createElement("li");
      item.textContent = service;
      return item;
    }),
  );
}

function profileSummary() {
  const profile = BUSINESS_PROFILES[activeBusinessType];
  if (!profile || !activeBusinessName) {
    return "Not selected";
  }

  return `${activeBusinessName} (${profile.label})`;
}

function appointmentSummary(appointment) {
  return `${appointment.date} at ${appointment.time} (${formatIntent(appointment.status)})`;
}

function slotAppointmentSummary(slots = {}) {
  const date = slots.date || "";
  const time = slots.time || "";

  if (date && time) {
    return `${date} at ${time}`;
  }

  return date || time || "No details";
}

function missingFieldsSummary(missingFields) {
  if (!Array.isArray(missingFields) || missingFields.length === 0) {
    return "";
  }

  return `Missing ${missingFields.map(formatIntent).join(", ")}.`;
}

function railNextStep(response = null) {
  if (!isProfileReady) {
    return "Select a profile to begin.";
  }

  if (!response) {
    return "Ready for customer messages.";
  }

  if (response.active_appointment) {
    return `${formatIntent(response.active_appointment.status)} appointment captured.`;
  }

  const missingFields = missingFieldsSummary(response.missing_fields);
  if (missingFields) {
    return missingFields;
  }

  return response.workflow_stage
    ? formatIntent(response.workflow_stage)
    : "Response ready.";
}

function greetingForActiveProfile() {
  return `Hi! I'm Aster, the AI receptionist for ${activeBusinessName}. How can I help today?`;
}

function updateQuickActionState() {
  quickActionButtons.forEach((button) => {
    button.disabled = !isProfileReady || isSubmitting;
  });
}

function updateProfileDisplay() {
  if (!isProfileReady) {
    sessionBusiness.textContent = "Not selected";
    contextBusiness.textContent = "Not selected";
    initialAssistantText.textContent = "Choose a demo business profile to begin.";
    setText(railBusiness, "Not selected");
    setText(railProfileType, "Choose a demo profile");
    setText(railProfileState, "Setup");
    setText(railValidation, railNextStep());
    renderServiceMenu();
    return;
  }

  const profile = BUSINESS_PROFILES[activeBusinessType];
  const summary = profileSummary();
  sessionBusiness.textContent = activeBusinessName;
  contextBusiness.textContent = summary;
  initialAssistantText.textContent = greetingForActiveProfile();
  setText(railBusiness, activeBusinessName);
  setText(railProfileType, profile?.label || "Business profile");
  setText(railProfileState, "Active");
  setText(railValidation, railNextStep());
  renderServiceMenu();
}

function createUserMessage(message) {
  const article = document.createElement("article");
  const content = document.createElement("div");
  const metadata = document.createElement("div");
  const author = document.createElement("strong");
  const time = document.createElement("time");
  const bubble = document.createElement("div");
  const text = document.createElement("p");

  article.className = "message message--user";
  article.dataset.userMessage = "true";
  article.dataset.dynamicMessage = "true";
  content.className = "message__content";
  metadata.className = "message__meta";
  bubble.className = "message__bubble";
  author.textContent = "You";
  time.textContent = formatCurrentTime();
  text.textContent = message;

  metadata.append(author, time);
  bubble.append(text);
  content.append(metadata, bubble);
  article.append(content);

  return article;
}

function createAssistantMessage(message, isError = false) {
  const article = document.createElement("article");
  const avatar = document.createElement("span");
  const content = document.createElement("div");
  const metadata = document.createElement("div");
  const author = document.createElement("strong");
  const time = document.createElement("time");
  const bubble = document.createElement("div");
  const text = document.createElement("p");

  article.className = "message message--assistant";
  article.dataset.assistantMessage = "true";
  article.dataset.dynamicMessage = "true";
  article.classList.toggle("message--error", isError);
  avatar.className = "agent-avatar message__avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "AI";
  content.className = "message__content";
  metadata.className = "message__meta";
  bubble.className = "message__bubble";
  author.textContent = "Aster";
  time.textContent = formatCurrentTime();
  text.textContent = message;

  metadata.append(author, time);
  bubble.append(text);
  content.append(metadata, bubble);
  article.append(avatar, content);

  return article;
}

function createResponseProgressMessage() {
  const article = document.createElement("article");
  const avatar = document.createElement("span");
  const content = document.createElement("div");
  const metadata = document.createElement("div");
  const author = document.createElement("strong");
  const time = document.createElement("time");
  const bubble = document.createElement("div");
  const progress = document.createElement("div");
  const statusRow = document.createElement("div");
  const dots = document.createElement("span");
  const copy = document.createElement("div");
  const title = document.createElement("strong");
  const detail = document.createElement("p");
  const steps = document.createElement("ol");

  article.className = "message message--assistant message--progress";
  article.dataset.dynamicMessage = "true";
  article.dataset.progressMessage = "true";
  avatar.className = "agent-avatar message__avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "AI";
  content.className = "message__content";
  metadata.className = "message__meta";
  bubble.className = "message__bubble";
  progress.className = "response-progress";
  progress.setAttribute("role", "status");
  progress.setAttribute("aria-live", "polite");
  statusRow.className = "response-progress__status";
  dots.className = "response-progress__dots";
  dots.setAttribute("aria-hidden", "true");
  copy.className = "response-progress__copy";
  title.className = "response-progress__title";
  title.dataset.progressTitle = "true";
  detail.className = "response-progress__detail";
  detail.dataset.progressDetail = "true";
  steps.className = "response-progress__steps";
  steps.setAttribute("aria-label", "Response progress");
  author.textContent = "Aster";
  time.textContent = formatCurrentTime();

  for (let index = 0; index < 3; index += 1) {
    dots.append(document.createElement("span"));
  }

  RESPONSE_PROGRESS_STEPS.forEach((step, index) => {
    const item = document.createElement("li");
    item.textContent = step.title;
    item.dataset.progressStep = String(index);
    steps.append(item);
  });

  copy.append(title, detail);
  statusRow.append(dots, copy);
  progress.append(statusRow, steps);
  metadata.append(author, time);
  bubble.append(progress);
  content.append(metadata, bubble);
  article.append(avatar, content);

  return article;
}

function updateResponseProgressStep(index) {
  if (!responseProgressMessage) {
    return;
  }

  const safeIndex = Math.min(index, RESPONSE_PROGRESS_STEPS.length - 1);
  const step = RESPONSE_PROGRESS_STEPS[safeIndex];
  const title = responseProgressMessage.querySelector("[data-progress-title]");
  const detail = responseProgressMessage.querySelector("[data-progress-detail]");

  setText(title, step.title);
  setText(detail, step.detail);
  responseProgressMessage
    .querySelectorAll("[data-progress-step]")
    .forEach((item, itemIndex) => {
      item.classList.toggle("is-active", itemIndex === safeIndex);
      item.classList.toggle("is-complete", itemIndex < safeIndex);
    });
}

function showResponseProgress() {
  clearResponseProgress();

  responseProgressStepIndex = 0;
  responseProgressMessage = createResponseProgressMessage();
  appendConversationMessage(responseProgressMessage);
  updateResponseProgressStep(responseProgressStepIndex);
  responseProgressTimer = window.setInterval(() => {
    responseProgressStepIndex = Math.min(
      responseProgressStepIndex + 1,
      RESPONSE_PROGRESS_STEPS.length - 1,
    );
    updateResponseProgressStep(responseProgressStepIndex);
  }, 850);
  scrollConversationToEnd();
}

function clearResponseProgress() {
  if (responseProgressTimer !== null) {
    window.clearInterval(responseProgressTimer);
    responseProgressTimer = null;
  }

  responseProgressMessage?.remove();
  responseProgressMessage = null;
  responseProgressStepIndex = 0;
}

function createSessionId() {
  if (typeof window.crypto?.randomUUID === "function") {
    return window.crypto.randomUUID();
  }

  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function formatIntent(intent) {
  if (!intent) {
    return "Not identified";
  }

  return intent
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function updateCollectedContext(response) {
  if (response.business_type && response.business_name) {
    activeBusinessType = response.business_type;
    activeBusinessName = response.business_name;
    updateProfileDisplay();
  }

  if (response.slots?.service) {
    contextService.textContent = response.slots.service;
    setText(railService, response.slots.service);
  }

  const slotAppointment = slotAppointmentSummary(response.slots);
  if (slotAppointment !== "No details") {
    setText(railAppointment, slotAppointment);
  }

  if (response.active_appointment) {
    const appointment = response.active_appointment;
    contextService.textContent = appointment.service;
    contextAppointment.textContent = appointmentSummary(appointment);
    setText(railService, appointment.service);
    setText(railAppointment, appointmentSummary(appointment));
  }

  setText(railValidation, railNextStep(response));
}

async function submitMessage() {
  const message = messageInput.value.trim();

  if (
    !isProfileReady ||
    !message ||
    isListening ||
    isRecognitionStarting ||
    isSubmitting
  ) {
    return;
  }

  const activeGeneration = ++requestGeneration;
  const activeSessionId = sessionId;
  appendConversationMessage(createUserMessage(message));
  messageInput.value = "";
  isSubmitting = true;
  sessionStatus.textContent = "Processing";
  sessionIntent.textContent = "Awaiting classification";
  setText(railStatus, "Processing");
  setText(railIntent, "Awaiting classification");
  setText(railValidation, "Waiting for assistant response.");
  announcement.textContent = "Request sent to the assistant.";
  updateComposerState();
  updateQuickActionState();
  showResponseProgress();
  scrollConversationToEnd();

  try {
    const response = await fetch("/api/v1/conversation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: activeSessionId,
        business_type: activeBusinessType,
        business_name: activeBusinessName,
      }),
    });

    if (!response.ok) {
      throw new Error(`Conversation request failed with status ${response.status}`);
    }

    const payload = await response.json();
    if (activeGeneration !== requestGeneration || activeSessionId !== sessionId) {
      return;
    }

    clearResponseProgress();
    appendConversationMessage(createAssistantMessage(payload.response));
    sessionStatus.textContent = "Response ready";
    sessionIntent.textContent = formatIntent(payload.intent);
    setText(railStatus, "Response ready");
    setText(railIntent, formatIntent(payload.intent));
    updateCollectedContext(payload);
    announcement.textContent = "Assistant response received.";
  } catch {
    if (activeGeneration !== requestGeneration) {
      return;
    }

    clearResponseProgress();
    appendConversationMessage(
      createAssistantMessage(
        "I'm sorry, the assistant is temporarily unavailable. Please try again.",
        true,
      ),
    );
    sessionStatus.textContent = "Service unavailable";
    sessionIntent.textContent = "Not identified";
    setText(railStatus, "Service unavailable");
    setText(railIntent, "Not identified");
    setText(railValidation, "Try again when the assistant is available.");
    announcement.textContent = "The assistant could not complete the request.";
  } finally {
    if (activeGeneration === requestGeneration) {
      isSubmitting = false;
      updateComposerState();
      updateVoiceControls();
      updateQuickActionState();
      scrollConversationToEnd();
      messageInput.focus();
    }
  }
}

function cancelSpeech() {
  if (!supportsSpeechSynthesis) {
    return;
  }

  activeUtterance = null;
  window.speechSynthesis.cancel();
}

function cancelVoiceActivity() {
  if (recognition && (isListening || isRecognitionStarting)) {
    suppressRecognitionFeedback = true;

    try {
      recognition.abort();
    } catch {
      suppressRecognitionFeedback = false;
    }
  }

  isRecognitionStarting = false;
  isRecognitionStopping = false;
  cancelSpeech();
  setListeningState(false);
  setVoiceStatus(defaultVoiceStatus());
}

function resetConversation({ announce = true } = {}) {
  cancelVoiceActivity();
  clearResponseProgress();
  requestGeneration += 1;
  isSubmitting = false;
  sessionId = createSessionId();
  document
    .querySelectorAll("[data-dynamic-message]")
    .forEach((message) => message.remove());
  messageInput.value = "";
  sessionStatus.textContent = isProfileReady ? "Ready" : "Setup required";
  sessionChannel.textContent = "Text";
  sessionIntent.textContent = "Not identified";
  contextService.textContent = "Not selected";
  contextAppointment.textContent = "No details";
  setText(railStatus, isProfileReady ? "Ready" : "Setup required");
  setText(railChannel, "Text");
  setText(railIntent, "Not identified");
  setText(railService, "Not selected");
  setText(railAppointment, "No details");
  setText(railValidation, railNextStep());
  updateProfileDisplay();
  announcement.textContent = announce
    ? "Conversation reset."
    : `Demo profile set to ${profileSummary()}.`;
  updateComposerState();
  updateVoiceControls();
  updateQuickActionState();
  conversationLog.scrollTo({ top: 0, behavior: "smooth" });
  if (isProfileReady) {
    messageInput.focus();
  } else {
    businessNameInput.focus();
  }
}

function updateProfileFormState() {
  const profile = selectedProfile();
  const businessName = cleanBusinessName(businessNameInput.value);

  businessNameInput.placeholder = profile
    ? `Example: ${profile.exampleName}`
    : "Enter business name";
  businessTypeSelect.setCustomValidity(profile ? "" : "Select a business type.");
  businessNameInput.setCustomValidity(businessName ? "" : "Enter a business name.");
  profileSubmitButton.disabled = !profile || !businessName;
}

function applyBusinessProfile(event) {
  event.preventDefault();

  const profile = selectedProfile();
  const businessName = cleanBusinessName(businessNameInput.value);

  businessTypeSelect.setCustomValidity(profile ? "" : "Select a business type.");
  businessNameInput.setCustomValidity(businessName ? "" : "Enter a business name.");
  if (!profile || !businessName) {
    updateProfileFormState();
    profileForm.reportValidity();
    return;
  }

  activeBusinessType = businessTypeSelect.value;
  activeBusinessName = businessName;
  businessNameInput.value = activeBusinessName;
  businessTypeSelect.setCustomValidity("");
  businessNameInput.setCustomValidity("");
  isProfileReady = true;
  resetConversation({ announce: false });
}

function combineTranscript(baseText, transcript) {
  return [baseText, transcript]
    .filter(Boolean)
    .join(" ")
    .slice(0, MAX_MESSAGE_LENGTH);
}

function recognitionErrorText(error) {
  const messages = {
    "audio-capture": "Microphone unavailable",
    network: "Voice service unavailable",
    "no-speech": "No speech detected",
    "not-allowed": "Microphone access denied",
    "service-not-allowed": "Voice service blocked",
  };

  return messages[error] || "Voice input failed";
}

function configureSpeechRecognition() {
  if (!SpeechRecognitionConstructor) {
    voiceInputButton.title = "Voice input is not supported in this browser";
    voiceInputButton.setAttribute(
      "aria-label",
      "Voice input is not supported in this browser",
    );
    return;
  }

  try {
    recognition = new SpeechRecognitionConstructor();
  } catch {
    recognition = null;
    voiceInputButton.title = "Voice input could not be initialized";
    voiceInputButton.setAttribute(
      "aria-label",
      "Voice input could not be initialized",
    );
    return;
  }

  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = "en-US";
  recognition.maxAlternatives = 1;

  recognition.addEventListener("start", () => {
    isRecognitionStarting = false;
    isRecognitionStopping = false;
    recognitionErrorMessage = "";
    setListeningState(true);
    sessionChannel.textContent = "Voice";
    setText(railChannel, "Voice");
    setVoiceStatus("Listening", true);
    announcement.textContent = "Voice input started.";
  });

  recognition.addEventListener("result", (event) => {
    const finalParts = [];
    const interimParts = [];

    for (let index = 0; index < event.results.length; index += 1) {
      const result = event.results[index];
      const transcript = result[0]?.transcript.trim();

      if (!transcript) {
        continue;
      }

      if (result.isFinal) {
        finalParts.push(transcript);
      } else {
        interimParts.push(transcript);
      }
    }

    currentTranscript = [...finalParts, ...interimParts].join(" ");
    messageInput.value = combineTranscript(recognitionBaseText, currentTranscript);
    updateComposerState();
    setVoiceStatus(finalParts.length > 0 ? "Transcript ready" : "Listening", true);
  });

  recognition.addEventListener("error", (event) => {
    isRecognitionStarting = false;
    recognitionErrorMessage = recognitionErrorText(event.error);

    if (suppressRecognitionFeedback && event.error === "aborted") {
      return;
    }

    setVoiceStatus(recognitionErrorMessage);
    announcement.textContent = `${recognitionErrorMessage}.`;
  });

  recognition.addEventListener("end", () => {
    const feedbackWasSuppressed = suppressRecognitionFeedback;

    suppressRecognitionFeedback = false;
    isRecognitionStarting = false;
    isRecognitionStopping = false;
    setListeningState(false);

    if (feedbackWasSuppressed) {
      setVoiceStatus(defaultVoiceStatus());
      return;
    }

    if (recognitionErrorMessage) {
      setVoiceStatus(recognitionErrorMessage);
      return;
    }

    if (currentTranscript) {
      sessionChannel.textContent = "Voice + text";
      setText(railChannel, "Voice + text");
      setVoiceStatus("Transcript ready");
      announcement.textContent = "Voice transcript ready for review.";
      messageInput.focus();
      return;
    }

    setVoiceStatus("No speech detected");
  });
}

function toggleVoiceInput() {
  if (!recognition) {
    return;
  }

  if (isListening) {
    isRecognitionStopping = true;
    setVoiceStatus("Finishing transcript", true);
    updateVoiceControls();
    recognition.stop();
    return;
  }

  cancelSpeech();
  recognitionBaseText = messageInput.value.trim();
  currentTranscript = "";
  recognitionErrorMessage = "";
  suppressRecognitionFeedback = false;
  isRecognitionStarting = true;
  setVoiceStatus("Starting microphone", true);
  updateComposerState();
  updateVoiceControls();

  try {
    recognition.start();
  } catch {
    isRecognitionStarting = false;
    setVoiceStatus("Voice input is already active");
    updateComposerState();
    updateVoiceControls();
  }
}

function preferredVoice() {
  const voices = window.speechSynthesis.getVoices();

  return (
    voices.find((voice) => voice.lang.toLowerCase() === "en-us") ||
    voices.find((voice) => voice.lang.toLowerCase().startsWith("en")) ||
    null
  );
}

function speakLatestAssistantMessage() {
  if (!supportsSpeechSynthesis) {
    return;
  }

  const assistantMessages = document.querySelectorAll("[data-assistant-message]");
  const latestMessage = assistantMessages.item(assistantMessages.length - 1);
  const messageText = latestMessage
    ?.querySelector(".message__bubble")
    ?.textContent.trim();

  if (!messageText) {
    setVoiceStatus("No assistant response to read");
    return;
  }

  cancelSpeech();

  const utterance = new SpeechSynthesisUtterance(messageText);
  const voice = preferredVoice();

  utterance.lang = "en-US";
  utterance.rate = 1;
  utterance.pitch = 1;

  if (voice) {
    utterance.voice = voice;
  }

  activeUtterance = utterance;

  utterance.addEventListener("start", () => {
    if (activeUtterance === utterance) {
      setVoiceStatus("Speaking", true);
      announcement.textContent = "Reading the latest assistant response.";
    }
  });

  utterance.addEventListener("end", () => {
    if (activeUtterance === utterance) {
      activeUtterance = null;
      setVoiceStatus(defaultVoiceStatus());
    }
  });

  utterance.addEventListener("error", () => {
    if (activeUtterance === utterance) {
      activeUtterance = null;
      setVoiceStatus("Read-aloud failed");
    }
  });

  window.speechSynthesis.speak(utterance);
}

function initializeVoiceSupport() {
  configureSpeechRecognition();

  if (!supportsSpeechSynthesis) {
    speakLatestButton.title = "Read-aloud is not supported in this browser";
    speakLatestButton.setAttribute(
      "aria-label",
      "Read-aloud is not supported in this browser",
    );
  }

  setVoiceStatus(defaultVoiceStatus());
  updateVoiceControls();
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  submitMessage();
});

messageInput.addEventListener("input", updateComposerState);

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

quickActionButtons.forEach((button) => {
  button.addEventListener("click", () => {
    if (!isProfileReady) {
      return;
    }

    cancelVoiceActivity();
    messageInput.value = button.dataset.prompt;
    sessionChannel.textContent = "Text";
    setText(railChannel, "Text");
    updateComposerState();
    messageInput.focus();
  });
});

profileForm.addEventListener("submit", applyBusinessProfile);
businessTypeSelect.addEventListener("change", updateProfileFormState);
businessNameInput.addEventListener("input", updateProfileFormState);
voiceInputButton.addEventListener("click", toggleVoiceInput);
speakLatestButton.addEventListener("click", speakLatestAssistantMessage);
resetButton.addEventListener("click", () => resetConversation());

updateProfileFormState();
updateProfileDisplay();
initializeVoiceSupport();
updateQuickActionState();
updateComposerState();
