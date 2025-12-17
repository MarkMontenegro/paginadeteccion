// ==================================================
// CONFIGURACIÓN GENERAL
// ==================================================
const GUARD_MODEL_URL   = "./model_guard_js/model.json";
const DISEASE_MODEL_URL = "./model_cnn_potato_js/model.json";

const GUARD_LABELS = ["NO_LEAF", "OTHER_CROPS", "POTATO_LEAF"];
const CLASS_NAMES  = ["Hoja sana", "Tizón tardío"];

const IMG_SIZE = 256;
const THRESH_POTATO = 0.6;
const BRIGHTNESS = 1.10;
const CONTRAST   = 1.15;
const OK_INDEX   = 0;

// ==================================================
// VARIABLES GLOBALES
// ==================================================
let guardModel   = null;
let diseaseModel = null;
let processedImg = null;

// ==================================================
// REFERENCIAS DOM
// ==================================================
const uploadInput = document.getElementById("imageUpload");
const fileNameEl  = document.getElementById("fileName");
const imgOriginal = document.getElementById("imgOriginal");
const imgProcessed= document.getElementById("imgProcessed");
const predictBtn  = document.getElementById("predictBtn");
const statusEl    = document.getElementById("status");

const resultBox  = document.getElementById("resultBox");
const predLabel  = document.getElementById("predLabel");
const predBadge  = document.getElementById("predBadge");
const predConf   = document.getElementById("predConf");
const confBar    = document.getElementById("confBar");
const probList   = document.getElementById("probList");
const diagText   = document.getElementById("diagText");

// ==================================================
// CARGA INTELIGENTE DE MODELOS (Layers o Graph)
// ==================================================
async function smartLoadModel(url) {
  try {
    const model = await tf.loadLayersModel(url);
    console.log("✅ LayersModel cargado:", url);
    return model;
  } catch {
    const model = await tf.loadGraphModel(url);
    console.log("✅ GraphModel cargado:", url);
    return model;
  }
}

async function loadModels() {
  try {
    statusEl.textContent = "Cargando modelos...";
    await tf.ready();

    [guardModel, diseaseModel] = await Promise.all([
      smartLoadModel(GUARD_MODEL_URL),
      smartLoadModel(DISEASE_MODEL_URL)
    ]);

    statusEl.textContent = "✅ Modelos cargados. Sube una imagen.";
  } catch (e) {
    console.error(e);
    statusEl.textContent = "❌ Error cargando modelos. Usa Live Server.";
  }
}
window.addEventListener("load", loadModels);

// ==================================================
// PROCESAMIENTO DE IMAGEN (256×256)
// ==================================================
function buildProcessedImage(img) {
  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.filter = `brightness(${BRIGHTNESS}) contrast(${CONTRAST})`;
  ctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);

  return canvas.toDataURL("image/jpeg", 0.92);
}

// ==================================================
// FILTRO VERDE (VEGETACIÓN)
// ==================================================
function isGreenish(img) {
  const t = tf.browser.fromPixels(img).toFloat();
  const [r, g, b] = tf.split(t, 3, 2);

  const greenMask = tf.logicalAnd(
    tf.logicalAnd(g.greater(r.mul(1.05)), g.greater(b.mul(1.05))),
    g.greater(50)
  );

  const ratio = greenMask.cast("float32").mean().dataSync()[0];
  tf.dispose([t, r, g, b, greenMask]);

  console.log("🌿 Verde:", (ratio * 100).toFixed(2), "%");
  return ratio > 0.04;
}

// ==================================================
// GUARD: ¿ES HOJA DE PAPA?
// ==================================================
async function checkIfPotato(img) {
  const tensor = tf.browser.fromPixels(img)
    .resizeNearestNeighbor([160, 160])
    .toFloat()
    .div(127.5)
    .sub(1)
    .expandDims(0);

  const pred = await guardModel.predict(tensor).data();
  tensor.dispose();
  return Array.from(pred);
}

// ==================================================
// PREPROCESAMIENTO ENFERMEDAD
// ==================================================
function preprocessDisease(img) {
  return tf.browser.fromPixels(img)
    .resizeBilinear([IMG_SIZE, IMG_SIZE])
    .toFloat()
    .div(255.0)
    .expandDims(0);
}

function argmax(arr) {
  return arr.indexOf(Math.max(...arr));
}

// ==================================================
// MOSTRAR RESULTADOS
// ==================================================
function renderResult(data) {
  const idx = argmax(data);
  const prob = data[idx];

  predLabel.textContent = CLASS_NAMES[idx];
  predBadge.textContent = idx === OK_INDEX ? "OK" : "ALERTA";
  predBadge.className = "badge " + (idx === OK_INDEX ? "ok" : "bad");

  predConf.textContent = `Confianza: ${(prob * 100).toFixed(2)}%`;
  confBar.style.width = `${(prob * 100).toFixed(1)}%`;

  probList.innerHTML = "";
  data.forEach((p, i) => {
    const div = document.createElement("div");
    div.className = "prob-item";
    div.innerHTML = `<span>${CLASS_NAMES[i]}</span><b>${(p * 100).toFixed(2)}%</b>`;
    probList.appendChild(div);
  });

  resultBox.style.display = "block";
}

// ==================================================
// CARGA DE IMAGEN
// ==================================================
uploadInput.addEventListener("change", (e) => {
  const file = e.target.files[0];

  resultBox.style.display = "none";
  probList.innerHTML = "";
  confBar.style.width = "0%";
  diagText.textContent = "";
  processedImg = null;

  if (!file) return;

  fileNameEl.textContent = file.name;

  const reader = new FileReader();
  reader.onload = (ev) => {
    imgOriginal.src = ev.target.result;
    imgOriginal.style.display = "block";

    imgOriginal.onload = () => {
      const processedURL = buildProcessedImage(imgOriginal);
      imgProcessed.src = processedURL;
      imgProcessed.style.display = "block";

      processedImg = new Image();
      processedImg.src = processedURL;

      predictBtn.disabled = false;
      statusEl.textContent = "✅ Imagen lista.";
    };
  };
  reader.readAsDataURL(file);
});

// ==================================================
// PIPELINE COMPLETO
// ==================================================
predictBtn.addEventListener("click", async () => {
  predictBtn.disabled = true;
  statusEl.textContent = "Analizando...";
  resultBox.style.display = "none";

  try {
    // 1️⃣ Filtro verde
    if (!isGreenish(imgOriginal)) {
      statusEl.textContent = "❌ No se detecta vegetación.";
      return;
    }

    // 2️⃣ GUARD
    const [noLeaf, other, potato] = await checkIfPotato(processedImg);
    diagText.textContent =
      `Guard → NO_LEAF ${(noLeaf*100).toFixed(1)}% | ` +
      `OTHER ${(other*100).toFixed(1)}% | ` +
      `POTATO ${(potato*100).toFixed(1)}%`;

    if (potato < THRESH_POTATO || potato < noLeaf || potato < other) {
      statusEl.textContent = "❌ No es hoja de papa.";
      return;
    }

    // 3️⃣ ENFERMEDAD
    const input = preprocessDisease(processedImg);
    let out = diseaseModel.predict ? diseaseModel.predict(input) : diseaseModel.execute(input);
    if (Array.isArray(out)) out = out[0];

    const data = await out.data();
    tf.dispose([input, out]);

    renderResult(Array.from(data));
    statusEl.textContent = "✅ Clasificación completada.";
  } catch (e) {
    console.error(e);
    statusEl.textContent = "❌ Error en la predicción.";
  } finally {
    predictBtn.disabled = false;
  }
});
