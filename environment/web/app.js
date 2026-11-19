// Umuhinzi 3D field — three.js front-end fed by the Python environment over SSE.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ------------------------------------------------------------------ setup
const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;

const scene = new THREE.Scene();
const SKY_CLEAR = new THREE.Color("#8ec5ff");
const SKY_RAIN = new THREE.Color("#5b6672");
scene.background = SKY_CLEAR.clone();
scene.fog = new THREE.Fog(scene.background, 60, 170);

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 500);
camera.position.set(22, 17, 26);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 1.2, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 10;
controls.maxDistance = 90;

const hemi = new THREE.HemisphereLight("#dff1ff", "#4b3a22", 0.9);
scene.add(hemi);
const sun = new THREE.DirectionalLight("#fff3d6", 2.2);
sun.position.set(25, 40, 15);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0008;
sun.shadow.normalBias = 0.03;
Object.assign(sun.shadow.camera, { left: -25, right: 25, top: 25, bottom: -25 });
scene.add(sun);

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

const rand = mulberry32(7);
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ------------------------------------------------------------ landscape
const grass = new THREE.Mesh(
  new THREE.CircleGeometry(220, 64),
  new THREE.MeshStandardMaterial({ color: "#5f8f3a", roughness: 1 })
);
grass.rotation.x = -Math.PI / 2;
grass.receiveShadow = true;
scene.add(grass);

// "Land of a thousand hills": terraced hills around the plot.
const hillMat = new THREE.MeshStandardMaterial({ color: "#4f7d30", roughness: 1, flatShading: true });
const terraceMat = new THREE.MeshStandardMaterial({ color: "#6d9a3c", roughness: 1, flatShading: true });
for (let i = 0; i < 16; i++) {
  const ang = (i / 16) * Math.PI * 2 + rand() * 0.3;
  const dist = 70 + rand() * 50;
  const r = 18 + rand() * 22, h = 10 + rand() * 18;
  const g = new THREE.Group();
  const steps = 5;
  for (let s = 0; s < steps; s++) {
    const k = 1 - s / steps;
    const ring = new THREE.Mesh(new THREE.CylinderGeometry(r * k * 0.82, r * k, h / steps, 20), s % 2 ? terraceMat : hillMat);
    ring.position.y = (s + 0.5) * (h / steps);
    ring.castShadow = false;
    ring.receiveShadow = true;
    g.add(ring);
  }
  g.position.set(Math.cos(ang) * dist, 0, Math.sin(ang) * dist);
  scene.add(g);
}

// Banana trees along the plot edge (very common next to Rwandan plots).
const trunkMat = new THREE.MeshStandardMaterial({ color: "#7a8b3a" });
const leafMat = new THREE.MeshLambertMaterial({ color: "#4a8f30", emissive: "#1f4214", side: THREE.DoubleSide });
for (let i = 0; i < 7; i++) {
  const t = new THREE.Group();
  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.35, 3.2, 8), trunkMat);
  trunk.position.y = 1.6;
  trunk.castShadow = true;
  t.add(trunk);
  for (let k = 0; k < 6; k++) {
    const leaf = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 3.0), leafMat);
    leaf.position.y = 3.3;
    leaf.rotation.set(-0.9, (k / 6) * Math.PI * 2, 0, "YXZ");
    leaf.translateY(1.2);
    leaf.castShadow = true;
    leaf.receiveShadow = false;
    t.add(leaf);
  }
  t.position.set(-14 + i * 4.5 + rand(), 0, -13.5 - rand() * 2);
  scene.add(t);
}

// House (inzu) next to the field.
const house = new THREE.Group();
const walls = new THREE.Mesh(new THREE.BoxGeometry(5, 3, 4), new THREE.MeshStandardMaterial({ color: "#b5651d" }));
walls.position.y = 1.5; walls.castShadow = true; house.add(walls);
const roof = new THREE.Mesh(new THREE.ConeGeometry(4.2, 2.2, 4), new THREE.MeshStandardMaterial({ color: "#8e9aa3", metalness: 0.4, roughness: 0.5 }));
roof.position.y = 4.1; roof.rotation.y = Math.PI / 4; roof.castShadow = true; house.add(roof);
house.position.set(-17, 0, 6);
house.rotation.y = 0.4;
scene.add(house);

// ----------------------------------------------------------------- field
const FIELD = 18;
const soilMat = new THREE.MeshStandardMaterial({ color: "#8a6a45", roughness: 1 });
const soil = new THREE.Mesh(new THREE.BoxGeometry(FIELD, 0.4, FIELD), soilMat);
soil.position.y = 0.2;
soil.receiveShadow = true;
scene.add(soil);
const DRY = new THREE.Color("#c4a372"), WET = new THREE.Color("#3e2a1a");

const ridgeMat = new THREE.MeshStandardMaterial({ color: "#6b4f33", roughness: 1 });
const ROWS = 14, COLS = 14, SP = FIELD / ROWS;
for (let r = 0; r < ROWS; r++) {
  const ridge = new THREE.Mesh(new THREE.BoxGeometry(FIELD - 0.4, 0.18, 0.5), ridgeMat);
  ridge.position.set(0, 0.45, -FIELD / 2 + SP * (r + 0.5));
  ridge.receiveShadow = true;
  scene.add(ridge);
}

// Maize plants as instanced stalks, leaves, tassels and cobs.
const N = ROWS * COLS;
const plants = [];
for (let r = 0; r < ROWS; r++)
  for (let c = 0; c < COLS; c++)
    plants.push({
      x: -FIELD / 2 + SP * (c + 0.5) + (rand() - 0.5) * 0.25,
      z: -FIELD / 2 + SP * (r + 0.5) + (rand() - 0.5) * 0.15,
      rot: rand() * Math.PI * 2,
      vigor: 0.85 + rand() * 0.3,
      frailty: rand(), // which plants show damage first
    });
const stalkGeo = new THREE.CylinderGeometry(0.05, 0.08, 1, 6);
stalkGeo.translate(0, 0.5, 0);
const leafGeo = new THREE.PlaneGeometry(0.16, 1.0, 1, 4);
leafGeo.translate(0, 0.5, 0);
{ // curve the leaf
  const p = leafGeo.attributes.position;
  for (let i = 0; i < p.count; i++) { const y = p.getY(i); p.setZ(i, -0.35 * y * y); }
  leafGeo.computeVertexNormals();
}
const tasselGeo = new THREE.ConeGeometry(0.08, 0.35, 5);
const cobGeo = new THREE.CylinderGeometry(0.07, 0.06, 0.32, 6);
const plantMat = new THREE.MeshStandardMaterial({ color: "#ffffff", roughness: 0.8, side: THREE.DoubleSide });
const LEAVES = 4;
const stalks = new THREE.InstancedMesh(stalkGeo, plantMat, N);
const leaves = new THREE.InstancedMesh(leafGeo, plantMat, N * LEAVES);
const tassels = new THREE.InstancedMesh(tasselGeo, new THREE.MeshStandardMaterial({ color: "#d9b84a" }), N);
const cobs = new THREE.InstancedMesh(cobGeo, new THREE.MeshStandardMaterial({ color: "#e8c547" }), N);
for (const m of [stalks, leaves, tassels, cobs]) { m.castShadow = true; m.instanceMatrix.setUsage(THREE.DynamicDrawUsage); scene.add(m); }

// Fall armyworm larvae, weeds, harvested sacks.
const larvaGeo = new THREE.CapsuleGeometry(0.035, 0.12, 2, 6);
const larvae = new THREE.InstancedMesh(larvaGeo, new THREE.MeshStandardMaterial({ color: "#c2410c", emissive: "#7c2d12", emissiveIntensity: 0.4 }), N * 3);
const weedGeo = new THREE.ConeGeometry(0.18, 0.35, 5);
const weeds = new THREE.InstancedMesh(weedGeo, new THREE.MeshStandardMaterial({ color: "#2f5d1e" }), 300);
const weedPos = Array.from({ length: 300 }, () => [(rand() - 0.5) * (FIELD - 1), (rand() - 0.5) * (FIELD - 1), rand()]);
const sackGeo = new THREE.CylinderGeometry(0.45, 0.55, 0.9, 10);
const sacks = new THREE.InstancedMesh(sackGeo, new THREE.MeshStandardMaterial({ color: "#e7dcc0" }), 20);
for (const m of [larvae, weeds, sacks]) { m.castShadow = true; m.count = 0; scene.add(m); }

// Farmer avatar.
const farmer = new THREE.Group();
const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.32, 0.9, 4, 10), new THREE.MeshStandardMaterial({ color: "#2563eb" }));
body.position.y = 1.0; farmer.add(body);
const head = new THREE.Mesh(new THREE.SphereGeometry(0.25, 16, 12), new THREE.MeshStandardMaterial({ color: "#5b3a24" }));
head.position.y = 1.85; farmer.add(head);
const hat = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 0.06, 16), new THREE.MeshStandardMaterial({ color: "#e9d8a6" }));
hat.position.y = 2.08; farmer.add(hat);
const tank = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.6, 0.25), new THREE.MeshStandardMaterial({ color: "#facc15" }));
tank.position.set(0, 1.15, -0.35); tank.visible = false; farmer.add(tank);
farmer.traverse((o) => (o.castShadow = true));
const HOME = new THREE.Vector3(-11.5, 0, 7);
farmer.position.copy(HOME);
scene.add(farmer);
let farmerPath = [];

// Particles: rain + spray/irrigation clouds.
const RAIN_N = 4000;
const rainGeo = new THREE.BufferGeometry();
const rainPos = new Float32Array(RAIN_N * 3);
for (let i = 0; i < RAIN_N; i++) { rainPos[i * 3] = (rand() - 0.5) * 70; rainPos[i * 3 + 1] = rand() * 30; rainPos[i * 3 + 2] = (rand() - 0.5) * 70; }
rainGeo.setAttribute("position", new THREE.BufferAttribute(rainPos, 3));
const rain = new THREE.Points(rainGeo, new THREE.PointsMaterial({ color: "#cfe3ff", size: 0.12, transparent: true, opacity: 0.75 }));
scene.add(rain);
let rainIntensity = 0;

const PUFF_N = 1500;
const puffGeo = new THREE.BufferGeometry();
const puffPos = new Float32Array(PUFF_N * 3);
const puffVel = new Float32Array(PUFF_N * 3);
puffGeo.setAttribute("position", new THREE.BufferAttribute(puffPos, 3));
const puffMat = new THREE.PointsMaterial({ color: "#ffffff", size: 0.22, transparent: true, opacity: 0 });
const puff = new THREE.Points(puffGeo, puffMat);
scene.add(puff);
let puffUntil = 0;
function emitPuff(color, upward) {
  puffMat.color.set(color);
  for (let i = 0; i < PUFF_N; i++) {
    puffPos[i * 3] = (rand() - 0.5) * FIELD;
    puffPos[i * 3 + 1] = upward ? 0.6 + rand() * 0.6 : 2.5 + rand() * 1.5;
    puffPos[i * 3 + 2] = (rand() - 0.5) * FIELD;
    puffVel[i * 3] = (rand() - 0.5) * 0.4;
    puffVel[i * 3 + 1] = upward ? 0.5 + rand() * 0.5 : -1.5 - rand();
    puffVel[i * 3 + 2] = (rand() - 0.5) * 0.4;
  }
  puffGeo.attributes.position.needsUpdate = true;
  puffUntil = performance.now() + 1600;
}

// ------------------------------------------------------------ state sync
const ACTIONS = ["WAIT", "PLANT", "IRRIGATE", "SPRAY_CHEAP", "SPRAY_BIO", "SCOUT", "WEED_FERTILIZE", "HARVEST"];
const SMS_TEXT = {
  PLANT: (s) => `Rains are established (${s.cum_rain.toFixed(0)} mm since 20 Aug, soil ${(s.soil_moisture * 100).toFixed(0)}%). Plant your maize this week.`,
  IRRIGATE: (s) => `Dry spell: soil moisture ${(s.soil_moisture * 100).toFixed(0)}%. Water your plot today from your rainwater tank.`,
  SPRAY_CHEAP: (s) => `Armyworm risk. Spray cypermethrin early morning, into the whorl.`,
  SPRAY_BIO: (s) => `Armyworm detected. Apply Bt / emamectin into the whorl (rotate products to avoid resistance).`,
  SCOUT: () => `Please inspect 20 plants in a W pattern and reply with how many show fresh armyworm damage.`,
  WEED_FERTILIZE: () => `Weed your plot and top-dress with manure/urea this week.`,
  HARVEST: (s) => `Your maize is mature. Harvest now to avoid field losses.`,
};
const SMS_CLASS = { SPRAY_CHEAP: "warn", SPRAY_BIO: "warn", IRRIGATE: "good", PLANT: "good", HARVEST: "good" };

const target = { stage: 0, health: 0, planted: false, pest: 0, weeds: 0, soil: 0.3, harvested: false, harvestKg: 0 };
const shown = { stage: 0, health: 0, pest: 0, weeds: 0, soil: 0.3 };
const history = [];
let lastEpisodeKey = null;

const $ = (id) => document.getElementById(id);
const pct = (v) => `${(v * 100).toFixed(0)}%`;
function meter(id, v, text) { $(`m-${id}`).style.width = `${Math.max(0, Math.min(1, v)) * 100}%`; $(`v-${id}`).textContent = text ?? pct(v); }

function onFrame(s) {
  const key = `${s.meta?.episode}-${s.year}`;
  if (key !== lastEpisodeKey || s.t === 0) {
    lastEpisodeKey = key;
    history.length = 0;
    $("sms-list").innerHTML = "";
    target.harvested = false;
    sacks.count = 0;
  }
  history.push(s);

  target.stage = s.stage; target.health = s.health; target.planted = s.planted;
  target.pest = s.pest_true; target.weeds = s.weeds; target.soil = s.soil_moisture;
  rainIntensity = Math.min(1, s.rain_today / 20);

  const act = s.last_action !== null && s.last_action !== undefined ? ACTIONS[s.last_action] : null;
  if (act && act !== "WAIT" && !s.event.startsWith("Invalid")) {
    const li = document.createElement("div");
    li.className = `sms ${SMS_CLASS[act] || ""}`;
    li.innerHTML = `<small>${s.date} · ${act.replace("_", " ")}</small>${SMS_TEXT[act](s)}`;
    $("sms-list").appendChild(li);
    while ($("sms-list").children.length > 7) $("sms-list").firstChild.remove();
    if (act === "SPRAY_CHEAP") { emitPuff("#e0f2fe", false); walkField(true); }
    if (act === "SPRAY_BIO") { emitPuff("#bbf7d0", false); walkField(true); }
    if (act === "IRRIGATE") { emitPuff("#60a5fa", false); walkField(false); }
    if (act === "SCOUT" || act === "WEED_FERTILIZE" || act === "PLANT") walkField(false);
    if (act === "HARVEST" || s.event.includes("forced harvest")) harvestNow(s.harvest_kg);
  }
  if (s.event.includes("forced harvest")) harvestNow(s.harvest_kg);

  // HUD
  const m = s.meta || {};
  $("chip-agent").textContent = `agent: ${m.agent ?? "–"}`;
  $("chip-episode").textContent = `episode ${m.episode ?? "–"} · year ${s.year}`;
  $("chip-weather").textContent = `${s.weather_source} weather${m.rain_scale && m.rain_scale !== 1 ? ` ×${m.rain_scale}` : ""}`;
  $("chip-status").textContent = "live"; $("chip-status").classList.add("on");
  $("date").textContent = s.date;
  $("day").textContent = `day ${s.day} / 198`;
  $("season-bar").style.width = `${(s.day / 198) * 100}%`;
  $("rain").textContent = `${s.rain_today.toFixed(1)} mm`;
  $("forecast").textContent = `${s.forecast_7d.toFixed(0)} mm`;
  $("cumrain").textContent = `${s.cum_rain.toFixed(0)} mm`;
  $("temp").textContent = `${s.temperature.toFixed(1)} °C`;
  meter("soil", s.soil_moisture);
  meter("tank", s.water_tank ?? 0);
  meter("stage", Math.min(s.stage, 1), s.planted ? pct(Math.min(s.stage, 1.2)) : "–");
  meter("health", s.health, s.planted ? pct(s.health) : "–");
  meter("weeds", s.weeds);
  meter("pest", s.pest_true);
  meter("pestest", s.pest_estimate);
  meter("res", s.resistance);
  meter("budget", s.budget / s.budget0, s.budget.toFixed(1));
  $("return").textContent = s.episode_return.toFixed(2);
  $("harvest").textContent = s.harvest_kg > 0 ? `${s.harvest_kg.toFixed(0)} kg` : "–";
  $("action-badge").textContent = s.last_action_label;
  $("event").textContent = s.event;
  $("reward").textContent = (s.step_reward ?? 0).toFixed(3);
  drawChart();
}

function walkField(spraying) {
  tank.visible = spraying;
  farmerPath = [];
  const z0 = FIELD / 2 - 0.5;
  for (let i = 0; i <= 4; i++) farmerPath.push(new THREE.Vector3(-FIELD / 2 + 1 + (i * (FIELD - 2)) / 4, 0.4, i % 2 ? -z0 + 1 : z0 - 1));
  farmerPath.push(HOME.clone());
}

function harvestNow(kg) {
  if (target.harvested) return;
  target.harvested = true;
  target.harvestKg = kg;
  const nSacks = Math.min(20, Math.round(kg / 25));
  const m = new THREE.Matrix4();
  for (let i = 0; i < nSacks; i++) {
    m.makeTranslation(-13 + (i % 5) * 1.1, 0.45, 1 + Math.floor(i / 5) * 1.1);
    sacks.setMatrixAt(i, m);
  }
  sacks.count = nSacks;
  sacks.instanceMatrix.needsUpdate = true;
}

// ------------------------------------------------------------ plant update
const tmpM = new THREE.Matrix4(), tmpQ = new THREE.Quaternion(), tmpS = new THREE.Vector3(), tmpP = new THREE.Vector3(), tmpE = new THREE.Euler();
const GREEN = new THREE.Color("#3f9b2f"), YELLOW = new THREE.Color("#c8b23c"), BROWN = new THREE.Color("#7a5a2a"), DRYLEAF = new THREE.Color("#c9b27a");
const col = new THREE.Color();

function updatePlants(time) {
  const visible = target.planted && !target.harvested;
  const stage = shown.stage, health = shown.health;
  const height = 0.08 + 2.6 * Math.min(stage, 1) ** 0.9;
  let li = 0, pi = 0;
  for (let i = 0; i < N; i++) {
    const p = plants[i];
    const damaged = p.frailty > health + 0.05;
    const h = visible ? height * p.vigor * (damaged ? 0.75 : 1) : 0.0001;
    const sway = Math.sin(time * 1.3 + p.x * 0.7 + p.z * 0.4) * 0.04 * (1 + rainIntensity * 2);
    const droop = damaged ? 0.35 : 0;
    tmpE.set(sway + droop * 0.5, p.rot, sway * 0.6);
    tmpQ.setFromEuler(tmpE);
    tmpM.compose(tmpP.set(p.x, 0.5, p.z), tmpQ, tmpS.set(1, h, 1));
    stalks.setMatrixAt(i, tmpM);
    if (stage > 1.0) col.copy(DRYLEAF); // mature crop dries down
    else col.copy(GREEN).lerp(YELLOW, Math.max(0, 1 - health) * 0.9);
    if (damaged) col.lerp(BROWN, 0.7);
    stalks.setColorAt(i, col);
    for (let k = 0; k < LEAVES; k++) {
      const ly = 0.5 + h * (0.25 + 0.18 * k);
      const ls = visible ? Math.min(1.0, 0.35 + stage * 1.6) * (1 - 0.15 * k) * p.vigor : 0.0001;
      tmpE.set(-0.9 - droop - sway, p.rot + k * 2.1, 0, "YXZ");
      tmpQ.setFromEuler(tmpE);
      tmpM.compose(tmpP.set(p.x, ly, p.z), tmpQ, tmpS.set(1, ls, 1));
      leaves.setMatrixAt(li, tmpM);
      leaves.setColorAt(li, col);
      li++;
    }
    const showT = visible && stage > 0.5;
    tmpM.compose(tmpP.set(p.x, 0.5 + h + 0.1, p.z), tmpQ.identity(), tmpS.setScalar(showT ? 1 : 0.0001));
    tassels.setMatrixAt(i, tmpM);
    const showC = visible && stage > 0.62 && !damaged;
    tmpE.set(0.4, p.rot, 0);
    tmpQ.setFromEuler(tmpE);
    tmpM.compose(tmpP.set(p.x + 0.08, 0.5 + h * 0.55, p.z), tmpQ, tmpS.setScalar(showC ? Math.min(1, (stage - 0.6) * 3) : 0.0001));
    cobs.setMatrixAt(i, tmpM);

    // larvae on infested plants
    if (visible && p.frailty < shown.pest * 1.6) {
      for (let k = 0; k < 2 && pi < larvae.count + 9999; k++) {
        const a = p.rot + k * 3 + time * 0.3;
        tmpM.compose(tmpP.set(p.x + Math.cos(a) * 0.12, 0.5 + h * (0.35 + 0.2 * k), p.z + Math.sin(a) * 0.12), tmpQ.setFromEuler(tmpE.set(0, a, Math.PI / 2)), tmpS.setScalar(1));
        larvae.setMatrixAt(pi++, tmpM);
      }
    }
  }
  larvae.count = Math.min(pi, N * 3);
  for (const m of [stalks, leaves, tassels, cobs, larvae]) { m.instanceMatrix.needsUpdate = true; }
  stalks.instanceColor.needsUpdate = true;
  leaves.instanceColor.needsUpdate = true;

  // weeds
  const nWeeds = visible ? Math.round(shown.weeds * 300) : 0;
  for (let i = 0; i < nWeeds; i++) {
    const [x, z, s] = weedPos[i];
    tmpM.compose(tmpP.set(x, 0.55, z), tmpQ.identity(), tmpS.setScalar(0.6 + s));
    weeds.setMatrixAt(i, tmpM);
  }
  weeds.count = nWeeds;
  weeds.instanceMatrix.needsUpdate = true;
}

// ----------------------------------------------------------------- chart
const chart = $("chart");
const ctx = chart.getContext("2d");
function drawChart() {
  const w = (chart.width = chart.clientWidth * devicePixelRatio);
  const h = (chart.height = 110 * devicePixelRatio);
  ctx.clearRect(0, 0, w, h);
  const maxT = history[0]?.max_steps || 66;
  const X = (t) => (t / maxT) * (w - 8) + 4;
  const Y = (v) => h - 6 - v * (h - 14);
  ctx.strokeStyle = "rgba(255,255,255,.08)";
  for (let g = 0; g <= 4; g++) { ctx.beginPath(); ctx.moveTo(0, Y(g / 4)); ctx.lineTo(w, Y(g / 4)); ctx.stroke(); }
  for (const s of history) {
    if (s.last_action > 0) {
      ctx.strokeStyle = "rgba(255,255,255,.35)";
      ctx.beginPath(); ctx.moveTo(X(s.t), h - 6); ctx.lineTo(X(s.t), h - 18); ctx.stroke();
    }
  }
  const series = [["health", "#4ade80"], ["soil_moisture", "#60a5fa"], ["pest_true", "#f97316"], ["stage", "#facc15"]];
  for (const [k, c] of series) {
    ctx.strokeStyle = c; ctx.lineWidth = 2 * devicePixelRatio; ctx.beginPath();
    history.forEach((s, i) => { const v = Math.min(1, s[k]); i ? ctx.lineTo(X(s.t), Y(v)) : ctx.moveTo(X(s.t), Y(v)); });
    ctx.stroke();
  }
}

// ------------------------------------------------------------ main loop
const clock = new THREE.Clock();
function animate() {
  const dt = Math.min(clock.getDelta(), 0.05);
  const time = clock.elapsedTime;
  const k = 1 - Math.exp(-dt * 4);
  for (const key of ["stage", "health", "pest", "weeds", "soil"]) shown[key] += (target[key] - shown[key]) * k;
  if (!target.planted) { shown.stage = target.stage; }

  soilMat.color.copy(DRY).lerp(WET, Math.min(1, shown.soil * 1.1));
  scene.background.copy(SKY_CLEAR).lerp(SKY_RAIN, rainIntensity);
  scene.fog.color.copy(scene.background);
  sun.intensity = 2.2 - 1.4 * rainIntensity;

  // rain
  rain.visible = rainIntensity > 0.02;
  if (rain.visible) {
    const n = Math.floor(RAIN_N * rainIntensity);
    rainGeo.setDrawRange(0, n);
    for (let i = 0; i < n; i++) {
      rainPos[i * 3 + 1] -= dt * 28;
      if (rainPos[i * 3 + 1] < 0) rainPos[i * 3 + 1] = 30;
    }
    rainGeo.attributes.position.needsUpdate = true;
  }
  // spray / irrigation puff
  const puffLife = (puffUntil - performance.now()) / 1000;
  if (puffLife > 0) {
    puffMat.opacity = Math.min(0.85, puffLife);
    for (let i = 0; i < PUFF_N; i++) {
      puffPos[i * 3] += puffVel[i * 3] * dt;
      puffPos[i * 3 + 1] = Math.max(0.5, puffPos[i * 3 + 1] + puffVel[i * 3 + 1] * dt);
      puffPos[i * 3 + 2] += puffVel[i * 3 + 2] * dt;
    }
    puffGeo.attributes.position.needsUpdate = true;
  } else puffMat.opacity = 0;

  // farmer walking
  if (farmerPath.length) {
    const tgt = farmerPath[0];
    const d = tgt.clone().sub(farmer.position);
    const step = dt * 14;
    if (d.length() < step) { farmer.position.copy(tgt); farmerPath.shift(); if (!farmerPath.length) tank.visible = false; }
    else { farmer.position.add(d.setLength(step)); farmer.rotation.y = Math.atan2(d.x, d.z); }
    body.position.y = 1.0 + Math.abs(Math.sin(time * 12)) * 0.08;
  }

  updatePlants(time);
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();

// ------------------------------------------------------------ data feed
function connect() {
  const es = new EventSource("/events");
  es.onmessage = (e) => onFrame(JSON.parse(e.data));
  es.onerror = () => {
    $("chip-status").textContent = "reconnecting…";
    $("chip-status").classList.remove("on");
  };
}
connect();
window.__umuhinzi = { onFrame }; // handy for debugging from the console
