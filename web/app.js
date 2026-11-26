/* PanoramaStudio Frontend Logic */

const state = {
  sourceType: null, // "upload" | "dataset"
  datasetId: null,
  items: [], // [{ name, url, file? }]
  datasets: [],
  currentJob: null,
  pollTimer: null,
  viewMode: "final", // "final" | "uncropped" | "compare"
  benchmarksLoaded: false,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* ---------- Navigation ---------- */
$$(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".nav-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const target = btn.dataset.view;
    $$(".view").forEach((v) => v.classList.remove("active"));
    $(`#view-${target}`).classList.add("active");
    if (target === "benchmarks" && !state.benchmarksLoaded) {
      loadBenchmarks();
    }
  });
});

/* ---------- Sliders ---------- */
const bindSlider = (inputId, labelId) => {
  const inp = $(inputId);
  const lbl = $(labelId);
  inp.addEventListener("input", () => {
    lbl.textContent = inp.value;
  });
};
bindSlider("#cfg-ratio", "#val-ratio");
bindSlider("#cfg-ransac", "#val-ransac");
bindSlider("#cfg-levels", "#val-levels");
bindSlider("#cfg-fov", "#val-fov");

/* ---------- Reset Config ---------- */
$("#btn-reset-cfg").addEventListener("click", () => {
  $("#cfg-detector").value = "sift";
  $("#cfg-matcher").value = "custom";
  $("#cfg-homography").value = "custom";
  $("#cfg-projection").value = "planar";
  $("#cfg-blending").value = "seam_multiband";
  $("#cfg-auto-order").checked = true;
  $("#cfg-exposure").checked = true;
  $("#cfg-auto-crop").checked = true;
  $("#cfg-visuals").checked = true;
  $("#cfg-ratio").value = "0.75";
  $("#val-ratio").textContent = "0.75";
  $("#cfg-ransac").value = "4.0";
  $("#val-ransac").textContent = "4.0";
  $("#cfg-levels").value = "5";
  $("#val-levels").textContent = "5";
  $("#cfg-fov").value = "55";
  $("#val-fov").textContent = "55";
});

/* ---------- Load Sample Datasets ---------- */
async function loadDatasets() {
  const res = await fetch("/api/datasets");
  const data = await res.json();
  state.datasets = data.datasets || [];
  const box = $("#dataset-list");
  box.innerHTML = "";
  state.datasets.forEach((ds) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ds-btn" + (ds.expected_error ? " ds-error" : "");
    btn.dataset.id = ds.id;
    btn.innerHTML = `
      <span>${ds.name}</span>
      <span class="ds-tag">${ds.expected_error ? "Test lỗi" : ds.num_images + " ảnh"}</span>
    `;
    btn.title = ds.description || ds.name;
    btn.addEventListener("click", () => selectDataset(ds));
    box.appendChild(btn);
  });
}

function selectDataset(ds) {
  state.sourceType = "dataset";
  state.datasetId = ds.id;
  state.items = ds.files.map((fn, idx) => ({
    name: fn,
    url: ds.thumbnails[idx],
  }));
  $$(".ds-btn").forEach((b) => b.classList.toggle("active", b.dataset.id === ds.id));
  renderFilmstrip();
}

/* ---------- File Upload & Drag-Drop ---------- */
const dropzone = $("#dropzone");
const fileInput = $("#file-input");

const addUploadedFiles = (fileList) => {
  const arr = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
  if (!arr.length) return;
  if (state.sourceType !== "upload") {
    state.items = [];
    state.datasetId = null;
    $$(".ds-btn").forEach((b) => b.classList.remove("active"));
  }
  state.sourceType = "upload";
  arr.slice(0, 10 - state.items.length).forEach((f) => {
    state.items.push({
      name: f.name,
      url: URL.createObjectURL(f),
      file: f,
    });
  });
  renderFilmstrip();
};

fileInput.addEventListener("change", (e) => {
  addUploadedFiles(e.target.files);
  fileInput.value = "";
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  addUploadedFiles(e.dataTransfer.files);
});

$("#btn-clear-images").addEventListener("click", () => {
  state.items = [];
  state.sourceType = null;
  state.datasetId = null;
  $$(".ds-btn").forEach((b) => b.classList.remove("active"));
  renderFilmstrip();
});

function renderFilmstrip() {
  const wrap = $("#filmstrip-wrap");
  const strip = $("#filmstrip");
  const countLbl = $("#selected-count");
  const stitchBtn = $("#btn-stitch");

  if (!state.items.length) {
    wrap.classList.add("hidden");
    stitchBtn.disabled = true;
    return;
  }

  wrap.classList.remove("hidden");
  countLbl.textContent = `Đã chọn ${state.items.length} ảnh`;
  stitchBtn.disabled = state.items.length < 2;

  strip.innerHTML = "";
  state.items.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "thumb-card";
    card.innerHTML = `
      <span class="thumb-order">#${idx + 1}</span>
      <img src="${item.url}" alt="${item.name}" />
      <div class="thumb-bar">
        <span class="thumb-name" title="${item.name}">${item.name}</span>
        <div class="thumb-acts">
          <button type="button" data-act="left" title="Chuyển trái">◀</button>
          <button type="button" data-act="right" title="Chuyển phải">▶</button>
          <button type="button" data-act="del" title="Xóa">✕</button>
        </div>
      </div>
    `;
    card.querySelector('[data-act="left"]').addEventListener("click", () => {
      if (idx > 0) {
        [state.items[idx - 1], state.items[idx]] = [state.items[idx], state.items[idx - 1]];
        renderFilmstrip();
      }
    });
    card.querySelector('[data-act="right"]').addEventListener("click", () => {
      if (idx < state.items.length - 1) {
        [state.items[idx + 1], state.items[idx]] = [state.items[idx], state.items[idx + 1]];
        renderFilmstrip();
      }
    });
    card.querySelector('[data-act="del"]').addEventListener("click", () => {
      state.items.splice(idx, 1);
      renderFilmstrip();
    });
    strip.appendChild(card);
  });
}

/* ---------- Collect Config & Trigger Stitching ---------- */
function getConfigPayload() {
  return {
    detector: $("#cfg-detector").value,
    matcher_mode: $("#cfg-matcher").value,
    homography_mode: $("#cfg-homography").value,
    projection: $("#cfg-projection").value,
    blending_mode: $("#cfg-blending").value,
    auto_order: $("#cfg-auto-order").checked,
    exposure_compensation: $("#cfg-exposure").checked,
    auto_crop: $("#cfg-auto-crop").checked,
    generate_visuals: $("#cfg-visuals").checked,
    ratio_threshold: parseFloat($("#cfg-ratio").value),
    ransac_threshold: parseFloat($("#cfg-ransac").value),
    multiband_levels: parseInt($("#cfg-levels").value, 10),
    cylindrical_fov_deg: parseFloat($("#cfg-fov").value),
  };
}

const STAGE_ORDER = ["features", "matching", "warping", "blending", "postprocess", "done"];

function updateStageUI(stage, pct, msg) {
  $("#progress-title").textContent =
    stage === "done"
      ? "Hoàn tất ghép ảnh toàn cảnh!"
      : "Đang xử lý ghép ảnh toàn cảnh...";
  $("#progress-pct").textContent = `${pct}%`;
  $("#progress-fill").style.width = `${pct}%`;
  $("#progress-msg").textContent = msg || "";
  const curIdx = STAGE_ORDER.indexOf(stage);
  $$("#stage-steps span").forEach((el) => {
    const idx = STAGE_ORDER.indexOf(el.dataset.stage);
    el.classList.remove("active", "done");
    if (curIdx >= 0) {
      if (idx < curIdx || stage === "done") el.classList.add("done");
      else if (idx === curIdx) el.classList.add("active");
    }
  });
}

$("#btn-stitch").addEventListener("click", async () => {
  if (state.items.length < 2) return;
  if (state.pollTimer) clearInterval(state.pollTimer);

  $("#empty-state").classList.add("hidden");
  $("#error-card").classList.add("hidden");
  $("#result-area").classList.add("hidden");
  $("#progress-card").classList.remove("hidden");
  $("#btn-stitch").disabled = true;
  updateStageUI("features", 4, "Đang gửi ảnh và khởi chạy pipeline...");
  window.scrollTo({ top: 0, behavior: "smooth" });

  const form = new FormData();
  form.append("config_json", JSON.stringify(getConfigPayload()));

  if (state.sourceType === "dataset" && state.datasetId) {
    form.append("dataset_id", state.datasetId);
    form.append(
      "dataset_files_json",
      JSON.stringify(state.items.map((it) => it.name))
    );
  } else {
    state.items.forEach((it) => {
      if (it.file) form.append("files", it.file, it.name);
    });
  }

  try {
    const resp = await fetch("/api/stitch", { method: "POST", body: form });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      showError(data.error || { code: "REQUEST_ERROR", message: data.detail || "Lỗi gửi yêu cầu" });
      return;
    }
    pollJob(data.job_id);
  } catch (err) {
    showError({
      code: "NETWORK_ERROR",
      message: "Không kết nối được tới máy chủ xử lý.",
      suggestion: "Kiểm tra lại xem tiến trình `python app.py` đang chạy hay chưa.",
    });
  }
});

function pollJob(jobId) {
  state.pollTimer = setInterval(async () => {
    try {
      const r = await fetch(`/api/jobs/${jobId}`);
      const job = await r.json();
      if (job.status === "running") {
        updateStageUI(job.stage, job.progress || 10, job.message);
      } else if (job.status === "done") {
        clearInterval(state.pollTimer);
        updateStageUI("done", 100, job.message);
        state.currentJob = job;
        renderResult(job);
        $("#btn-stitch").disabled = false;
      } else if (job.status === "error") {
        clearInterval(state.pollTimer);
        showError(job.error);
      }
    } catch (e) {
      clearInterval(state.pollTimer);
      showError({ code: "POLL_ERROR", message: String(e) });
    }
  }, 350);
}

function showError(err) {
  $("#progress-card").classList.add("hidden");
  $("#result-area").classList.add("hidden");
  $("#error-card").classList.remove("hidden");
  $("#error-code").textContent = err.code || "ERROR";
  $("#error-message").textContent = err.message || "Xảy ra lỗi khi ghép ảnh.";
  $("#error-suggestion").textContent = err.suggestion ? `Gợi ý khắc phục: ${err.suggestion}` : "";
  $("#btn-stitch").disabled = state.items.length < 2;
}

/* ---------- Render Result ---------- */
function renderResult(job) {
  const s = job.summary;
  const m = s.metrics;
  const out = job.outputs;

  $("#error-card").classList.add("hidden");
  $("#result-area").classList.remove("hidden");

  $("#dl-png").href = out.download_png;
  $("#dl-jpg").href = out.download_jpg;
  $("#dl-zip").href = out.download_zip;

  // KPI strip
  const orderText = s.visual_order.map((i) => `#${i + 1}`).join(" → ");
  $("#kpi-strip").innerHTML = `
    <div class="kpi">
      <div class="kpi-label">Thời gian lõi</div>
      <div class="kpi-val">${m.core_pipeline_time_sec.toFixed(2)}s</div>
      <div class="kpi-sub">Mục tiêu &lt; 30s</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Kích thước đầu ra</div>
      <div class="kpi-val">${s.output_size.width}×${s.output_size.height}</div>
      <div class="kpi-sub">Giữ ${m.crop_retained_pct}% diện tích</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Trung bình Inliers</div>
      <div class="kpi-val">${m.mean_inliers}</div>
      <div class="kpi-sub">Tỷ lệ ${m.mean_inlier_ratio_pct}%</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Sai số chiếu lại</div>
      <div class="kpi-val">${m.mean_reprojection_rmse_px} px</div>
      <div class="kpi-sub">Chuẩn subpixel</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Độ mượt Seam</div>
      <div class="kpi-val">${m.seam_gradient_jump}</div>
      <div class="kpi-sub">Chế độ: ${s.config.blending_mode}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Thứ tự Trái → Phải</div>
      <div class="kpi-val" style="font-size:0.86rem">${orderText}</div>
      <div class="kpi-sub">Gốc: ${s.anchor_name}</div>
    </div>
  `;

  // Luôn đưa về tab Ảnh Toàn cảnh khi có kết quả mới
  $$(".subtab").forEach((b) => b.classList.toggle("active", b.dataset.subtab === "panorama"));
  $$(".subtab-pane").forEach((p) => p.classList.toggle("active", p.id === "subtab-panorama"));

  setViewerMode("final");
  renderIntermediates(job.visuals);
  renderProfiling(s);
}

/* ---------- Result Subtabs ---------- */
$$(".subtab").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".subtab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    $$(".subtab-pane").forEach((p) => p.classList.remove("active"));
    $(`#subtab-${btn.dataset.subtab}`).classList.add("active");
  });
});

/* ---------- Panorama Viewer & Compare Slider ---------- */
function setViewerMode(mode) {
  state.viewMode = mode;
  $("#mode-final").classList.toggle("active", mode === "final");
  $("#mode-uncropped").classList.toggle("active", mode === "uncropped");
  $("#mode-compare").classList.toggle("active", mode === "compare");

  if (!state.currentJob) return;
  const out = state.currentJob.outputs;
  const panoImg = $("#pano-img");
  const cmpWrap = $("#compare-wrap");

  if (mode === "compare") {
    panoImg.classList.add("hidden");
    cmpWrap.classList.remove("hidden");
    $("#cmp-base").src = out.panorama_uncropped;
    $("#cmp-top").src = out.panorama_jpg;
    setComparePos(50);
  } else {
    cmpWrap.classList.add("hidden");
    panoImg.classList.remove("hidden");
    panoImg.src = mode === "final" ? out.panorama_jpg : out.panorama_uncropped;
    applyZoom(parseInt($("#zoom-slider").value, 10));
  }
}

$("#mode-final").addEventListener("click", () => setViewerMode("final"));
$("#mode-uncropped").addEventListener("click", () => setViewerMode("uncropped"));
$("#mode-compare").addEventListener("click", () => setViewerMode("compare"));

function applyZoom(pct) {
  $("#zoom-label").textContent = `${pct}%`;
  const img = $("#pano-img");
  img.style.maxWidth = pct > 100 ? "none" : "100%";
  img.style.width = `${pct}%`;
}

$("#zoom-slider").addEventListener("input", (e) => applyZoom(parseInt(e.target.value, 10)));
$("#pano-img").addEventListener("click", () => {
  if ($("#pano-img").src) openLightbox($("#pano-img").src, "Ảnh Toàn cảnh kết quả");
});

// Compare Slider Drag
let draggingCmp = false;
function setComparePos(percent) {
  const clamped = Math.max(2, Math.min(98, percent));
  $("#cmp-overlay").style.width = `${clamped}%`;
  $("#cmp-handle").style.left = `${clamped}%`;
}
$("#compare-wrap").addEventListener("pointerdown", (e) => {
  draggingCmp = true;
  const rect = $("#compare-wrap").getBoundingClientRect();
  setComparePos(((e.clientX - rect.left) / rect.width) * 100);
});
window.addEventListener("pointermove", (e) => {
  if (!draggingCmp) return;
  const rect = $("#compare-wrap").getBoundingClientRect();
  setComparePos(((e.clientX - rect.left) / rect.width) * 100);
});
window.addEventListener("pointerup", () => {
  draggingCmp = false;
});

/* ---------- Intermediates Gallery ---------- */
function renderIntermediates(visuals) {
  const box = $("#intermediates-content");
  if (!visuals || (!visuals.keypoints?.length && !visuals.warped_layout)) {
    box.innerHTML = `<div class="muted">Bạn đã tắt tùy chọn "Xuất ảnh trung gian". Hãy bật lại rồi ghép để xem chi tiết từng bước.</div>`;
    return;
  }
  let html = "";

  if (visuals.match_graph || visuals.warped_layout) {
    html += `<div class="inter-group"><h3>Đồ thị Tự sắp xếp (Auto-Order MST) &amp; Bố cục Warping / Đường nối DP</h3><div class="inter-grid" style="grid-template-columns: repeat(auto-fit, minmax(420px, 1fr))">`;
    if (visuals.match_graph) {
      html += `<div class="inter-card" data-lb="${visuals.match_graph}" data-cap="Đồ thị liên kết Cây khung lớn nhất (Auto-Order MST)">
        <img src="${visuals.match_graph}" alt="Match graph" />
        <div class="inter-cap">Đồ thị liên kết Cây khung lớn nhất (Auto-Order MST)</div>
      </div>`;
    }
    if (visuals.warped_layout) {
      html += `<div class="inter-card" data-lb="${visuals.warped_layout}" data-cap="Khung ảnh sau Warping &amp; đường nối tối ưu DP (màu đỏ cam)">
        <img src="${visuals.warped_layout}" alt="Warped layout" />
        <div class="inter-cap">Khung ảnh sau Warping &amp; đường nối tối ưu DP (màu đỏ cam)</div>
      </div>`;
    }
    html += `</div></div>`;
  }

  if (visuals.matches?.length) {
    html += `<div class="inter-group"><h3>Khớp đặc trưng &amp; Phân loại RANSAC (Xanh lá = Inliers, Đỏ = Outliers)</h3><div class="inter-grid" style="grid-template-columns: repeat(auto-fit, minmax(420px, 1fr))">`;
    visuals.matches.forEach((m) => {
      html += `<div class="inter-card" data-lb="${m.url}" data-cap="${m.label}">
        <img src="${m.url}" alt="${m.label}" />
        <div class="inter-cap">${m.label}</div>
      </div>`;
    });
    html += `</div></div>`;
  }

  if (visuals.keypoints?.length) {
    html += `<div class="inter-group"><h3>Điểm đặc trưng phát hiện trên từng ảnh đầu vào (Keypoints)</h3><div class="inter-grid">`;
    visuals.keypoints.forEach((k) => {
      html += `<div class="inter-card" data-lb="${k.url}" data-cap="${k.label}">
        <img src="${k.url}" alt="${k.label}" />
        <div class="inter-cap">${k.label}</div>
      </div>`;
    });
    html += `</div></div>`;
  }

  box.innerHTML = html;
  box.querySelectorAll(".inter-card").forEach((el) => {
    el.addEventListener("click", () => openLightbox(el.dataset.lb, el.dataset.cap));
  });
}

/* ---------- Profiling & Tables ---------- */
function renderProfiling(s) {
  const stageLabels = {
    preprocess: "Tiền xử lý / Chiếu trụ",
    feature_detection: "1. Phát hiện đặc trưng",
    matching_and_homography: "2–3. Khớp đặc trưng & Homography (RANSAC)",
    warping: "4. Biến đổi phối cảnh (Warping)",
    blending: "5. Cân bằng sáng & Trộn ảnh",
    postprocess: "6. Hậu xử lý & Cắt biên tự động",
    visualizations: "Kết xuất ảnh minh họa trung gian",
  };
  const total = Math.max(1, s.timings_ms.total || 1);
  let tHtml = "";
  Object.entries(stageLabels).forEach(([k, label]) => {
    const val = s.timings_ms[k] || 0;
    const pct = (val / total) * 100;
    tHtml += `
      <div class="timing-row">
        <div class="timing-head"><span>${label}</span><strong>${val.toFixed(1)} ms (${pct.toFixed(1)}%)</strong></div>
        <div class="timing-bar-bg"><div class="timing-bar-fg" style="width:${Math.min(100, pct)}%"></div></div>
      </div>
    `;
  });
  $("#timing-bars").innerHTML = tHtml;

  // Exposure gains
  let gHtml = `<table class="data-table"><thead><tr><th>Ảnh</th><th>Gain B</th><th>Gain G</th><th>Gain R</th></tr></thead><tbody>`;
  Object.entries(s.exposure_gains || {}).forEach(([name, g]) => {
    gHtml += `<tr><td>${name}</td><td>${g[0].toFixed(3)}</td><td>${g[1].toFixed(3)}</td><td>${g[2].toFixed(3)}</td></tr>`;
  });
  gHtml += `</tbody></table>`;
  $("#gains-table").innerHTML = gHtml;

  // MST Pairs
  let pHtml = `<table class="data-table"><thead><tr><th>Ảnh nguồn</th><th>Ảnh đích</th><th>Matches (Ratio Test)</th><th>Inliers (RANSAC)</th><th>Tỷ lệ Inlier</th><th>Sai số RMSE</th></tr></thead><tbody>`;
  (s.metrics.mst_pairs || []).forEach((p) => {
    pHtml += `<tr>
      <td>${p.src}</td>
      <td>${p.dst}</td>
      <td>${p.matches}</td>
      <td><strong>${p.inliers}</strong></td>
      <td>${p.inlier_ratio_pct}%</td>
      <td>${p.reprojection_rmse_px} px</td>
    </tr>`;
  });
  pHtml += `</tbody></table>`;
  $("#pairs-table").innerHTML = pHtml;
}

/* ---------- Benchmarks View ---------- */
async function loadBenchmarks() {
  const box = $("#benchmark-body");
  try {
    const res = await fetch("/api/benchmarks");
    const payload = await res.json();
    if (!payload.available) {
      box.innerHTML = `<div class="card">Chưa có dữ liệu benchmark. Hãy chạy <code>python scripts/run_benchmarks.py</code>.</div>`;
      return;
    }
    state.benchmarksLoaded = true;
    const d = payload.data;
    const c = payload.charts;

    let dsRows = "";
    d.datasets_summary.datasets.forEach((item) => {
      dsRows += `<tr>
        <td><strong>${item.label}</strong></td>
        <td>${item.output_size.width}×${item.output_size.height}</td>
        <td>${item.metrics.core_pipeline_time_sec}s</td>
        <td>${item.metrics.mean_inliers} (${item.metrics.mean_inlier_ratio_pct}%)</td>
        <td>${item.metrics.mean_reprojection_rmse_px} px</td>
        <td>${item.auto_order_correct ? "Đúng 100%" : "Chưa khớp"}</td>
      </tr>`;
    });

    let cvRows = "";
    d.custom_vs_opencv.forEach((r) => {
      cvRows += `<tr>
        <td><strong>${r.config_name}</strong></td>
        <td>${r.match_homog_ms} ms</td>
        <td>${r.mean_inliers} (${r.mean_inlier_ratio_pct}%)</td>
        <td><strong>${r.mean_rmse_px} px</strong></td>
      </tr>`;
    });

    const neg = d.datasets_summary.negative_test;
    box.innerHTML = `
      <div class="kpi-strip" style="grid-template-columns: repeat(4, 1fr);">
        <div class="kpi"><div class="kpi-label">Tỷ lệ ghép thành công</div><div class="kpi-val">${d.success_rate_pct}%</div><div class="kpi-sub">Yêu cầu đề tài ≥ 80%</div></div>
        <div class="kpi"><div class="kpi-label">5 ảnh Full HD (1920×1080)</div><div class="kpi-val">${d.datasets_summary.datasets[0].metrics.core_pipeline_time_sec}s</div><div class="kpi-sub">Yêu cầu &lt; 30 giây</div></div>
        <div class="kpi"><div class="kpi-label">Custom vs OpenCV (SIFT RMSE)</div><div class="kpi-val">${d.custom_vs_opencv[0].mean_rmse_px} px</div><div class="kpi-sub">Tương đương tuyệt đối OpenCV</div></div>
        <div class="kpi"><div class="kpi-label">Kiểm thử ảnh không chồng lấn</div><div class="kpi-val">${neg.handled_gracefully ? "An toàn" : "Lỗi"}</div><div class="kpi-sub">Bắt mã: ${neg.code || "-"}</div></div>
      </div>

      <div class="card" style="margin-bottom:16px">
        <h3>Bảng 1: Kết quả ghép trên 4 bộ dữ liệu thực nghiệm (Cấu hình Tự cài đặt)</h3>
        <table class="data-table">
          <thead><tr><th>Bộ dữ liệu</th><th>Kích thước đầu ra</th><th>Thời gian lõi</th><th>Inliers (Tỷ lệ)</th><th>RMSE</th><th>Auto-Order</th></tr></thead>
          <tbody>${dsRows}</tbody>
        </table>
      </div>

      <div class="card" style="margin-bottom:16px">
        <h3>Bảng 2: Đối chiếu Thuật toán Tự cài đặt (Custom NumPy) và OpenCV chuẩn</h3>
        <table class="data-table">
          <thead><tr><th>Cấu hình</th><th>Thời gian Match + RANSAC</th><th>Trung bình Inliers</th><th>Sai số RMSE</th></tr></thead>
          <tbody>${cvRows}</tbody>
        </table>
      </div>

      <div class="bench-grid">
        <div class="chart-card">
          <h3>Phân rã thời gian từng giai đoạn (so với ngưỡng 30s)</h3>
          <img src="${c.stage_timings}" alt="Stage timings" />
        </div>
        <div class="chart-card">
          <h3>Đối chiếu Custom NumPy và OpenCV</h3>
          <img src="${c.custom_vs_opencv}" alt="Custom vs OpenCV" />
        </div>
        <div class="chart-card full-span">
          <h3>So sánh trực quan 4 kỹ thuật Trộn ảnh (Chênh sáng ±30% &amp; Vật chuyển động)</h3>
          <img src="${c.blending_grid}" alt="Blending grid" />
        </div>
        <div class="chart-card full-span">
          <h3>Phóng to vùng chồng lấn — Khả năng né vật chuyển động (chống Ghosting) của DP Seam</h3>
          <img src="${c.blending_zoom}" alt="Blending zoom" />
        </div>
      </div>
    `;

    box.querySelectorAll(".chart-card img").forEach((img) => {
      img.addEventListener("click", () => openLightbox(img.src, img.alt));
    });
  } catch (err) {
    box.innerHTML = `<div class="card">Lỗi tải dữ liệu thực nghiệm: ${err}</div>`;
  }
}

/* ---------- Lightbox ---------- */
function openLightbox(src, caption) {
  $("#lightbox-img").src = src;
  $("#lightbox-caption").textContent = caption || "";
  $("#lightbox").classList.remove("hidden");
}
$("#lightbox-close").addEventListener("click", () => $("#lightbox").classList.add("hidden"));
$("#lightbox").addEventListener("click", (e) => {
  if (e.target.id === "lightbox") $("#lightbox").classList.add("hidden");
});

/* ---------- Init ---------- */
loadDatasets();
