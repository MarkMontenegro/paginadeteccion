// =====================
//  CONFIGURACIÓN DE MODELOS
// =====================

// Modelo guardia (LayersModel)
const GUARD_MODEL_URL = "model_guard_js/model.json";
// [NO_LEAF, OTHER_CROPS, POTATO_LEAF]
const GUARD_LABELS = ["NO_LEAF", "OTHER_CROPS", "POTATO_LEAF"];

// Modelo enfermedad CNN exportado desde Colab (GraphModel)
const DISEASE_MODEL_URL = "model_cnn_potato_js/model.json";
// Ajusta el orden al de class_names en tu entrenamiento
const CLASS_NAMES = ["Hoja sana", "Tizón tardío"];

// Umbral mínimo para aceptar que realmente es hoja de papa
const THRESH_POTATO = 0.51;

// Tamaño de entrada de tu CNN
const CNN_SIZE = 256;

let guardModel = null;   // LayersModel
let diseaseModel = null; // GraphModel

// Referencias a elementos del DOM
const uploadInput = document.getElementById("imageUpload");
const previewImg = document.getElementById("preview");
const fileNameSpan = document.getElementById("fileName");
const predictBtn = document.getElementById("predictBtn");
const statusDiv = document.getElementById("status");
const resultBox = document.getElementById("resultBox");
const resultLabel = document.getElementById("resultLabel");
const probText = document.getElementById("probText");
const diagDiv = document.getElementById("diagnostic");

// =========================
//  CARGA DE MODELOS
// =========================
async function loadModels() {
  try {
    statusDiv.textContent = "Cargando modelos...";

    // Guardia: LayersModel | Enfermedad: GraphModel
    const [guard, disease] = await Promise.all([
      tf.loadLayersModel(GUARD_MODEL_URL),
      tf.loadGraphModel(DISEASE_MODEL_URL),
    ]);

    guardModel = guard;
    diseaseModel = disease;

    statusDiv.textContent = "✅ Modelos cargados. Sube una imagen para clasificar.";
  } catch (err) {
    console.error("Error cargando los modelos:", err);
    statusDiv.textContent = "❌ Error al cargar los modelos. Revisa rutas y usa Live Server.";
  }
}

window.addEventListener("load", loadModels);

// =========================
//  MANEJO DE IMAGEN
// =========================
uploadInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  resultBox.classList.add("hidden");
  diagDiv.textContent = "";

  if (!file) {
    fileNameSpan.textContent = "Ningún archivo seleccionado";
    previewImg.style.display = "none";
    predictBtn.disabled = true;
    return;
  }

  fileNameSpan.textContent = file.name;

  const reader = new FileReader();
  reader.onload = (event) => {
    previewImg.src = event.target.result;
    previewImg.style.display = "block";
    predictBtn.disabled = false;
  };
  reader.readAsDataURL(file);
});

// =========================
//  FILTRO VERDE (protección por color)
//  (tu versión actual - promedio global)
// =========================
function isGreenish(img) {
  const tensor = tf.browser.fromPixels(img);
  const mean = tensor.mean(0).mean(0).dataSync();
  const [r, g, b] = mean;
  tensor.dispose();

  const greenRatio = g / ((r + b) / 2 + 1e-6);
  console.log(
    "RGB promedio:",
    r.toFixed(1),
    g.toFixed(1),
    b.toFixed(1),
    "→ ratio verde:",
    greenRatio.toFixed(2)
  );

  return greenRatio > 0.50;
}

// =========================
//  MODELO GUARDIA (protección forma/fondo)
// =========================
async function checkIfPotato(imgElement) {
  const tensor = tf.browser
    .fromPixels(imgElement)
    .resizeNearestNeighbor([160, 160])
    .toFloat()
    .div(127.5)
    .sub(1.0)
    .expandDims();

  const prediction = await guardModel.predict(tensor).data();
  tensor.dispose();

  console.log("Predicciones guardia:", prediction);
  return prediction;
}

// =========================
//  MODELO ENFERMEDAD (CNN GraphModel 256x256)
// =========================
async function classifyDisease(imgElement) {
  const input = tf.browser
    .fromPixels(imgElement)
    .resizeBilinear([CNN_SIZE, CNN_SIZE])
    .toFloat()
    .div(255.0)
    .expandDims(0); // [1,256,256,3]

  // GraphModel devuelve Tensor o Array<Tensor>
  let out = diseaseModel.predict(input);
  if (Array.isArray(out)) out = out[0];

  const data = await out.data();

  tf.dispose([input, out]);

  // argmax
  let maxIdx = 0;
  for (let i = 1; i < data.length; i++) {
    if (data[i] > data[maxIdx]) maxIdx = i;
  }

  const prob = data[maxIdx];
  return {
    index: maxIdx,
    label: CLASS_NAMES[maxIdx] || `Clase ${maxIdx}`,
    prob,
  };
}

// =========================
//  PIPELINE COMPLETO AL HACER CLICK
// =========================
predictBtn.addEventListener("click", async () => {
  if (!guardModel || !diseaseModel) {
    statusDiv.textContent = "⚠️ Los modelos aún no están cargados.";
    return;
  }
  if (!previewImg.src) {
    statusDiv.textContent = "⚠️ Primero selecciona una imagen.";
    return;
  }

  statusDiv.textContent = "Aplicando filtros de protección...";
  predictBtn.disabled = true;
  resultBox.classList.add("hidden");
  diagDiv.textContent = "";

  try {
    // 1) Filtro de color (verde)
    if (!isGreenish(previewImg)) {
      showNotPotato(
        "❌ La imagen no parece contener vegetación.",
        "Filtro verde: ❌ (sin predominancia de verde)"
      );
      statusDiv.textContent = "Imagen rechazada por filtro de color.";
      return;
    }

    // 2) Modelo guardia
    const predGuard = await checkIfPotato(previewImg);
    const noLeafProb = predGuard[0];
    const otherCropProb = predGuard[1];
    const potatoProb = predGuard[2];

    diagDiv.textContent =
      "🛡 Guardia (NO_LEAF, OTHER_CROPS, POTATO_LEAF):\n" +
      `- NO_LEAF: ${(noLeafProb * 100).toFixed(1)}%\n` +
      `- OTHER_CROPS: ${(otherCropProb * 100).toFixed(1)}%\n` +
      `- POTATO_LEAF: ${(potatoProb * 100).toFixed(1)}%\n`;

    if (
      potatoProb < THRESH_POTATO ||
      potatoProb < noLeafProb ||
      potatoProb < otherCropProb
    ) {
      showNotPotato(
        "❌ No parece una hoja de papa (otro objeto / cultivo detectado).",
        "\nClasificación guardia: ❌ No papa."
      );
      statusDiv.textContent = "Imagen rechazada por modelo guardia.";
      return;
    }

    // 3) Si pasó filtros → CNN
    statusDiv.textContent = "Clasificando enfermedad (CNN)...";
    const disease = await classifyDisease(previewImg);
    showDiseaseResult(disease);
    statusDiv.textContent = "Clasificación completada.";
  } catch (err) {
    console.error("Error en la clasificación:", err);
    statusDiv.textContent = "❌ Error al clasificar la imagen.";
  } finally {
    predictBtn.disabled = false;
  }
});

// =========================
//  FUNCIONES DE PRESENTACIÓN
// =========================
function showNotPotato(mainText, extraText = "") {
  resultBox.classList.remove("hidden", "result-ok", "result-bad");
  resultBox.classList.add("result-bad");
  resultLabel.textContent = "Imagen no válida";
  probText.textContent = mainText;
  if (extraText) diagDiv.textContent += extraText;
}

function showDiseaseResult({ index, label, prob }) {
  resultBox.classList.remove("hidden", "result-ok", "result-bad");

  if (index === 0) resultBox.classList.add("result-ok");
  else resultBox.classList.add("result-bad");

  resultLabel.textContent = label;
  probText.textContent = `Confianza aproximada: ${(prob * 100).toFixed(2)} %`;
}
