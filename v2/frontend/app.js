// Smart Parking KSU — all frontend logic in one vanilla JS file.
// No framework, no build step. Served by FastAPI at the same origin as the API.

// ============================================================
// Config
// ============================================================
// Same origin as the page, so no CORS concerns and no hardcoded host.
const API = "";                    // empty string = same origin
const POLL_INTERVAL_MS = 1000;     // detection screen refresh rate

// ============================================================
// DOM references (grabbed once at load)
// ============================================================
const screens = {
  menu:      document.getElementById("screen-menu"),
  preview:   document.getElementById("screen-preview"),
  detection: document.getElementById("screen-detection"),
};

const menuStatus       = document.getElementById("menu-status");
const btnSelectVideo   = document.getElementById("btn-select-video");
const btnDetectSpots   = document.getElementById("btn-detect-spots");
const btnEndProgram    = document.getElementById("btn-end-program");

const videoSelect      = document.getElementById("video-select");
const previewVideoName = document.getElementById("preview-video-name");
const previewImage     = document.getElementById("preview-image");
const btnPreviewBack   = document.getElementById("btn-preview-back");

const detectionVideoName = document.getElementById("detection-video-name");
const liveImage          = document.getElementById("live-image");
const spotsSelect        = document.getElementById("spots-select");
const countTotal         = document.getElementById("count-total");
const countAvailable     = document.getElementById("count-available");
const countOccupied      = document.getElementById("count-occupied");
const detectionStatus    = document.getElementById("detection-status");
const btnDetectionBack   = document.getElementById("btn-detection-back");

// ============================================================
// App-level state (just enough to coordinate screens)
// ============================================================
let selectedVideo = null;      // filename string, e.g. "demo_angled.mp4"
let pollTimer     = null;      // setInterval handle for the detection loop

// Connection health tracking. We don't crash on poll failures — we count
// consecutive failures and only show "connection lost" if it's been >5s.
let consecutivePollFailures = 0;
const FAILURES_BEFORE_WARNING = 5;  // 5 polls × 1s = 5 seconds

// ============================================================
// Screen management
// ============================================================
function showScreen(name) {
  for (const key of Object.keys(screens)) {
    if (key === name) {
      screens[key].classList.remove("hidden");
    } else {
      screens[key].classList.add("hidden");
    }
  }
}

// ============================================================
// API helpers
// ============================================================
async function apiGet(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
  return res.json();
}

async function apiPost(path, body = null) {
  const opts = { method: "POST" };
  if (body !== null) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(`${API}${path}`, opts);
  if (!res.ok) {
    let detail = `POST ${path} -> ${res.status}`;
    try {
      const j = await res.json();
      if (j.detail) detail += ` (${j.detail})`;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

// ============================================================
// Main menu — Select Video flow
// ============================================================
async function enterPreviewScreen() {
  showScreen("preview");
  previewVideoName.textContent = "Loading videos…";
  videoSelect.innerHTML = `<option value="">Loading…</option>`;

  try {
    const data = await apiGet("/videos");
    const videos = data.videos || [];

    if (videos.length === 0) {
      previewVideoName.textContent = "No videos available";
      videoSelect.innerHTML = `<option value="">(none found)</option>`;
      return;
    }

    // Populate dropdown
    videoSelect.innerHTML =
      `<option value="">Choose a video…</option>` +
      videos.map(v => `<option value="${v.name}">${v.name}</option>`).join("");

    // Auto-select the previously-selected video if still present, else the first one.
    const preferred = selectedVideo && videos.find(v => v.name === selectedVideo)
      ? selectedVideo
      : videos[0].name;
    videoSelect.value = preferred;
    await loadPreviewFor(preferred);

  } catch (err) {
    previewVideoName.textContent = "Failed to load videos";
    console.error(err);
  }
}

async function loadPreviewFor(videoName) {
  if (!videoName) return;
  previewVideoName.textContent = videoName;
  try {
    await apiPost("/select-video", { video_name: videoName });
    // Cache-bust so switching videos actually reloads the image
    previewImage.src = `${API}/preview-frame?t=${Date.now()}`;
    selectedVideo = videoName;
    menuStatus.textContent = `Selected: ${videoName}`;
    btnDetectSpots.disabled = false;
  } catch (err) {
    previewVideoName.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

videoSelect.addEventListener("change", (e) => {
  loadPreviewFor(e.target.value);
});

btnSelectVideo.addEventListener("click", enterPreviewScreen);

btnPreviewBack.addEventListener("click", () => {
  showScreen("menu");
});

// ============================================================
// Detection flow
// ============================================================
async function enterDetectionScreen() {
  if (!selectedVideo) {
    menuStatus.textContent = "Select a video first";
    return;
  }

  showScreen("detection");
  detectionVideoName.textContent = selectedVideo;
  detectionStatus.textContent = "Starting detection…";
  countTotal.textContent = "—";
  countAvailable.textContent = "—";
  countOccupied.textContent = "—";

  // Populate the spots dropdown for this video before starting detection.
  await loadSpotsList();

  try {
    await apiPost("/start-detection");
    detectionStatus.textContent = "Detection running";
    startPolling();
  } catch (err) {
    detectionStatus.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

async function loadSpotsList() {
  spotsSelect.innerHTML = `<option value="">Loading…</option>`;
  try {
    const data = await apiGet("/spots-files");
    const files = data.files || [];
    if (files.length === 0) {
      spotsSelect.innerHTML = `<option value="">(no spots files)</option>`;
      return;
    }
    spotsSelect.innerHTML = files
      .map(f => `<option value="${f}">${f}</option>`)
      .join("");
    if (data.active) {
      spotsSelect.value = data.active;
    }
  } catch (err) {
    spotsSelect.innerHTML = `<option value="">(load failed)</option>`;
    console.error("loadSpotsList:", err);
  }
}

spotsSelect.addEventListener("change", async (e) => {
  const filename = e.target.value;
  if (!filename) return;
  try {
    await apiPost("/switch-spots", { video_name: filename });
    detectionStatus.textContent = `Switched to ${filename} — resmoothing…`;
    consecutivePollFailures = 0;
  } catch (err) {
    detectionStatus.textContent = `Switch failed: ${err.message}`;
    console.error(err);
  }
});

function startPolling() {
  stopPolling();  // belt and suspenders
  // Immediate first tick so the UI doesn't sit blank for a whole second.
  pollTick();
  pollTimer = setInterval(pollTick, POLL_INTERVAL_MS);
}

function stopPolling() {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function pollTick() {
  // Update the live frame by cache-busting the img src.
  // The browser handles failed loads silently — no try/catch needed.
  liveImage.src = `${API}/live-frame?t=${Date.now()}`;

  // Update the counts from /parking-status. This is the call we care about
  // for connection health — if it fails, the backend is unreachable or busy.
  try {
    const data = await apiGet("/parking-status");
    const status = data.status || {};
    if (status.total_spots !== undefined) {
      countTotal.textContent     = status.total_spots;
      countAvailable.textContent = status.available_spots;
      countOccupied.textContent  = status.occupied_spots;
      detectionStatus.textContent =
        `Frame ${status.frame_index} · ${status.timestamp_sec?.toFixed?.(1) ?? "?"}s`;
    }
    // Successful poll — clear any failure state.
    consecutivePollFailures = 0;
  } catch (err) {
    // Don't break the UI on transient failures. Count, retry next tick.
    consecutivePollFailures++;
    if (consecutivePollFailures >= FAILURES_BEFORE_WARNING) {
      detectionStatus.textContent =
        `⚠ Connection lost — retrying… (${consecutivePollFailures} failed polls)`;
    }
    // Don't log every failure to console — that's noise. Only log first one.
    if (consecutivePollFailures === 1) {
      console.warn("poll failed:", err.message);
    }
  }
}

async function exitDetectionScreen() {
  stopPolling();
  consecutivePollFailures = 0;  // reset for next session
  detectionStatus.textContent = "Stopping…";
  try {
    await apiPost("/stop-detection");
  } catch (err) {
    console.error(err);  // swallow — we're leaving the screen anyway
  }
  showScreen("menu");
  menuStatus.textContent = selectedVideo
    ? `Selected: ${selectedVideo}`
    : "No video selected";
}

btnDetectSpots.addEventListener("click", enterDetectionScreen);
btnDetectionBack.addEventListener("click", exitDetectionScreen);

// ============================================================
// End Program (soft shutdown)
// ============================================================
btnEndProgram.addEventListener("click", async () => {
  stopPolling();
  try {
    await apiPost("/stop-detection");
  } catch (err) {
    // Not running is fine, swallow.
  }
  selectedVideo = null;
  btnDetectSpots.disabled = true;
  menuStatus.textContent = "Program ended — select a video to start again";
  showScreen("menu");
});

// ============================================================
// Boot
// ============================================================
showScreen("menu");