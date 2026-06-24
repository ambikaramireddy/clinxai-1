// three-scene.js - Interactive Background

// 1. Scene Setup
const canvas = document.getElementById('bg-canvas');
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0f172a, 0.002); // Add depth fog

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 20;

const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Cap pixel ratio for performance

// 2. Objects: Neural Orb
// We'll create a Points cloud forming a sphere
const geometry = new THREE.BufferGeometry();
const count = 1500;
const positions = new Float32Array(count * 3);
const colors = new Float32Array(count * 3);

// Initialize as a sphere
const colorBase = new THREE.Color('#3b82f6');
for (let i = 0; i < count; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos((Math.random() * 2) - 1);
    const radius = 10 + (Math.random() * 0.5); // Slight thickness

    const x = radius * Math.sin(phi) * Math.cos(theta);
    const y = radius * Math.sin(phi) * Math.sin(theta);
    const z = radius * Math.cos(phi);

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    colors[i * 3] = colorBase.r;
    colors[i * 3 + 1] = colorBase.g;
    colors[i * 3 + 2] = colorBase.b;
}

geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

const material = new THREE.PointsMaterial({
    size: 0.15,
    vertexColors: true,
    transparent: true,
    opacity: 0.8,
    blending: THREE.AdditiveBlending
});

const orb = new THREE.Points(geometry, material);
scene.add(orb);

// Add some ambient background particles
const starGeo = new THREE.BufferGeometry();
const starCount = 500;
const starPos = new Float32Array(starCount * 3);
for (let i = 0; i < starCount; i++) {
    starPos[i * 3] = (Math.random() - 0.5) * 100;
    starPos[i * 3 + 1] = (Math.random() - 0.5) * 100;
    starPos[i * 3 + 2] = (Math.random() - 0.5) * 100 - 20; // Push back
}
starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
const starMat = new THREE.PointsMaterial({ color: 0x64748b, size: 0.1, transparent: true, opacity: 0.5 });
const stars = new THREE.Points(starGeo, starMat);
scene.add(stars);


// 3. State Machine
let currentState = 'idle';
let targetColor = new THREE.Color('#3b82f6');
let rotationSpeed = 0.001;
let pulseFactor = 0;

// Export Global Hook
window.setBackgroundState = function (state) {
    currentState = state;
    if (state === 'idle') {
        targetColor.set('#3b82f6'); // Blue
        rotationSpeed = 0.001;
    } else if (state === 'processing') {
        targetColor.set('#f59e0b'); // Warm/Yellow
        rotationSpeed = 0.02; // Fast spin
    } else if (state === 'result') {
        // Logic inside render loop handles checking UI result class if needed, or pass explicit color?
        // For simplification, let ui.js set result logic implicitly or we default to a check.
        // Let's rely on standard result color (e.g., White/Neutral) then UI can override?
        // No, requirement said "Color locks to risk level".
        // Let's look for element in UI logic.
        const diagnosisEl = document.getElementById('res-diagnosis');
        if (diagnosisEl && diagnosisEl.textContent.includes('PNEUMONIA')) {
            targetColor.set('#ef4444'); // Red
        } else if (diagnosisEl) {
            targetColor.set('#22c55e'); // Green
        }
        rotationSpeed = 0.002;
    }
};

// 4. Interaction (Mouse / Parallax)
let mouseX = 0;
let mouseY = 0;
let targetX = 0;
let targetY = 0;

const windowHalfX = window.innerWidth / 2;
const windowHalfY = window.innerHeight / 2;

document.addEventListener('mousemove', (event) => {
    mouseX = (event.clientX - windowHalfX);
    mouseY = (event.clientY - windowHalfY);
});

// 5. Animation Loop
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);

    const time = clock.getElapsedTime();

    // Parallax Easing
    targetX = mouseX * 0.001;
    targetY = mouseY * 0.001;

    // Smooth camera movement
    orb.rotation.y += rotationSpeed;
    orb.rotation.x += rotationSpeed * 0.5;

    // Add mouse influence to rotation
    orb.rotation.y += 0.05 * (targetX - orb.rotation.y);
    orb.rotation.x += 0.05 * (targetY - orb.rotation.x);

    stars.rotation.y += 0.0002;

    // Color Transition (Lerp)
    orb.material.color.lerp(targetColor, 0.05);

    // Pulse Effect (Processing State)
    if (currentState === 'processing') {
        const scale = 1 + Math.sin(time * 10) * 0.05;
        orb.scale.set(scale, scale, scale);
    } else {
        orb.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);
    }

    renderer.render(scene, camera);
}

// Disable vertexColors to use material color tinting for state efficiency
orb.material.vertexColors = false;
orb.material.color.set('#3b82f6');

animate();

// 5. Responsive Resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    // Check mobile
    const isMobile = window.innerWidth < 768;
    orb.material.size = isMobile ? 0.2 : 0.15;
});
