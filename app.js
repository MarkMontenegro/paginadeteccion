const GUARD_MODEL_URL   = "guard_cnn_js/model.json";
const DISEASE_MODEL_URL = "model_cnn_potato_js/model.json";

const GUARD_SIZE = 160;
const DISEASE_SIZE = 256;

const DISEASE_LABELS = ["Hoja sana", "Tizón tardío"];

let THRESH_POTATO = 0.70;

const uploadInput = document.getElementById("imageUpload");
const previewImg  = document.getElementById("preview");
const fileNameEl  = document.getElementById("fileName");

const btnRun   = document.getElementById("btnRun");
const btnClear = document.getElementById("btnClear");

const threshSlider = document.getElementById("threshPotato");
const threshValue  = document.getElementById("threshValue");

const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");

const guardResultEl = document.getElementById("guardResult");
const guardProbEl   = document.getElementById("guardProb");
const guardBarEl    = document.getElementById("guardBar");

const diseaseResultEl = document.getElementById("diseaseResult");
const diseaseProbEl   = document.getElementById("diseaseProb");
const diseaseNoteEl   = document.getElementById("diseaseNote");

let guardModel = null;
let diseaseModel = null;

function log(msg) {
  const t = new Date().toLocaleTimeString();
  logEl.textContent += `[${t}] ${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

function setGuardUI(prob) {
  const pct = Math.max(0, Math.min(100, Math.round(prob * 100)));
  guardProbEl.textContent = prob.toFixed(4);
  guardBarEl.style.width = `${pct}%`;

  const passed = prob >= THRESH_POTATO;
  guardResultEl.textContent = passed ? "✅ POTATO_LEAF" : "❌ NOT_POTATO";
  guardResultEl.style.color = passed ? "var(--ok)" : "var(--bad)";
}

function setDiseaseUI(label, prob, note = "") {
  diseaseResultEl.textContent = label;
  diseaseProbEl.textContent = Number.isFinite(prob) ? prob.toFixed(4) : "—";
  diseaseNoteEl.textContent = note || "—";

  const isBad = String(label).toLowerCase().includes("tizón");
  diseaseResultEl.style.color = isBad ? "var(--warn)" : "var(--ok)";
}

function resetOutputs() {
  guardResultEl.textContent = "—";
  guardProbEl.textContent = "—";
  guardBarEl.style.width = "0%";
  diseaseResultEl.textContent = "—";
  diseaseProbEl.textContent = "—";
  diseaseNoteEl.textContent = "—";
}

function makeTensorFromImage(imgEl, size) {
  return tf.tidy(() => {
    const t = tf.browser.fromPixels(imgEl)
      .resizeBilinear([size, size])
      .toFloat();
    return t.expandDims(0);
  });
}

async function predictGraphModel(model, inputTensor) {
  const inputName = model.inputs?.[0]?.name;
  let out;


  try {
    if (inputName) {
      const feed = {};
      feed[inputName] = inputTensor;
      out = model.execute(feed);
    } else {
      out = model.execute(inputTensor);
    }
  } catch (e) {
    if (inputName) {
      const feed = {};
      feed[inputName] = inputTensor;
      out = await model.executeAsync(feed);
    } else {
      out = await model.executeAsync(inputTensor);
    }
  }

  let tensorOut = null;

  if (out instanceof tf.Tensor) {
    tensorOut = out;
  } else if (Array.isArray(out)) {
    tensorOut = out[0];
  } else if (out && typeof out === "object") {
    const firstKey = Object.keys(out)[0];
    tensorOut = out[firstKey];
  }

  if (!tensorOut) {
    throw new Error("No se pudo obtener Tensor de salida del modelo (GraphModel).");
  }

  const data = await tensorOut.data();

  if (out instanceof tf.Tensor) {
    out.dispose();
  } else if (Array.isArray(out)) {
    out.forEach(t => t.dispose && t.dispose());
  } else if (out && typeof out === "object") {
    Object.values(out).forEach(t => t.dispose && t.dispose());
  }

  return data;
}

async function loadModels() {
  setStatus("⏳ Cargando modelos (esto puede tardar un poco)…");
  log("Cargando modelos TFJS...");

  try {
    try {
      await tf.setBackend("webgl");
      await tf.ready();
    } catch {
      await tf.setBackend("cpu");
      await tf.ready();
    }
    log(`Backend TFJS: ${tf.getBackend()}`);

    guardModel = await tf.loadGraphModel(GUARD_MODEL_URL);
    log("✅ Modelo Guardia cargado");

    diseaseModel = await tf.loadGraphModel(DISEASE_MODEL_URL);
    log("✅ Modelo de Enfermedad cargado");
    setStatus("✅ Modelos listos. Sube una imagen y ejecuta.");
    btnRun.disabled = false;
    log("✅ Modelos listos");

  } catch (err) {
    console.error(err);
    log(`❌ Error cargando modelos: ${err.message || err}`);
    setStatus("❌ No se pudieron cargar los modelos. Revisa rutas y consola.");
  }
}

async function runPipeline() {
  if (!previewImg.src) return;

  btnRun.disabled = true;
  setStatus("🔎 Procesando imagen…");
  log("Ejecutando pipeline...");

  try {
    const tGuard = makeTensorFromImage(previewImg, GUARD_SIZE);
    const gData = await predictGraphModel(guardModel, tGuard);
    tGuard.dispose();

    const probPotato = gData[0]; 
    log(`Guard prob POTATO_LEAF = ${probPotato.toFixed(6)}`);
    setGuardUI(probPotato);

    if (probPotato < THRESH_POTATO) {
      setDiseaseUI("—", NaN, "El guardia rechazó la imagen. No se ejecuta el modelo de enfermedad.");
      setStatus("✅ Terminado: imagen rechazada por el guardia.");
      return;
    }

    const tDis = makeTensorFromImage(previewImg, DISEASE_SIZE);
    const dData = await predictGraphModel(diseaseModel, tDis);
    tDis.dispose();

    let label = "—";
    let conf = 0;

    if (dData.length >= 2) {
      const p0 = dData[0], p1 = dData[1];
      const ex0 = Math.exp(p0), ex1 = Math.exp(p1);
      const s0 = ex0 / (ex0 + ex1);
      const s1 = ex1 / (ex0 + ex1);

      const idx = s1 > s0 ? 1 : 0;
      label = DISEASE_LABELS[idx];
      conf = idx === 1 ? s1 : s0;

      log(`Disease raw=[${p0.toFixed(4)}, ${p1.toFixed(4)}] softmax=[${s0.toFixed(4)}, ${s1.toFixed(4)}]`);
    } else {
      const p = dData[0];
      label = (p >= 0.5) ? DISEASE_LABELS[1] : DISEASE_LABELS[0];
      conf = (p >= 0.5) ? p : (1 - p);
      log(`Disease prob (sigmoid) = ${p.toFixed(6)}`);
    }

    setDiseaseUI(label, conf, "El guardia aprobó la imagen. Clasificación realizada.");
    setStatus("✅ Terminado: guardia + enfermedad ejecutados.");

  } catch (err) {
    console.error(err);
    log(`❌ Error: ${err.message || err}`);
    setStatus("❌ Error al ejecutar. Revisa consola y rutas de modelos.");
  } finally {
    btnRun.disabled = false;
  }
}

uploadInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  fileNameEl.textContent = file.name;
  btnClear.disabled = false;

  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.style.display = "block";

  resetOutputs();
  log(`Imagen cargada: ${file.name}`);
});

btnRun.addEventListener("click", runPipeline);

btnClear.addEventListener("click", () => {
  uploadInput.value = "";
  previewImg.removeAttribute("src");
  previewImg.style.display = "none";
  fileNameEl.textContent = "Ningún archivo seleccionado";
  resetOutputs();
  btnClear.disabled = true;
  log("Limpieza OK.");
  setStatus("✅ Modelos listos. Sube una imagen y ejecuta.");
});

threshSlider.addEventListener("input", () => {
  THRESH_POTATO = parseFloat(threshSlider.value);
  threshValue.textContent = THRESH_POTATO.toFixed(2);
});

threshValue.textContent = parseFloat(threshSlider.value).toFixed(2);
loadModels();
