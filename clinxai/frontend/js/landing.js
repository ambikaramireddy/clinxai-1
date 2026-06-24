/* ================================================
   ClinXAI Landing Page - Interactive Logic
   Handcrafted, scroll-driven, premium experience
   ================================================ */

'use strict';

// ================================================
// Three.js Dynamic Interactive Background
// ================================================

let scene, camera, renderer, particles, linesMesh, medicalGrid, ecgLine;
let mouseX = 0, mouseY = 0;
let targetMouseX = 0, targetMouseY = 0;
let rotationSpeed = 0.0003;
let heartbeatPhase = 0; // For medical heartbeat pulse effect

/**
 * Initialize Three.js particle background with medical-themed dynamics
 */
function initThreeBackground() {
    const container = document.getElementById('three-background');
    if (!container) return;

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(
        60,
        window.innerWidth / window.innerHeight,
        1,
        2000
    );
    camera.position.z = 900;

    // Particle system
    const particleCount = 850;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const velocities = new Float32Array(particleCount * 3);
    const scales = new Float32Array(particleCount); // For heartbeat pulse

    // Vibrant medical color palette
    const color1 = new THREE.Color(0x6ba3d4);  // Brighter blue
    const color2 = new THREE.Color(0x4d8ab8);  // Medium blue  
    const color3 = new THREE.Color(0xa8c5da);  // Light blue-gray
    const colorMedical = new THREE.Color(0x5dd9c1); // Medical cyan-green (vibrant)
    const colorAccent = new THREE.Color(0x7ec8e3); // Sky blue accent

    for (let i = 0; i < particleCount; i++) {
        const i3 = i * 3;

        positions[i3] = (Math.random() - 0.5) * 1800;
        positions[i3 + 1] = (Math.random() - 0.5) * 1800;
        positions[i3 + 2] = (Math.random() - 0.5) * 1200;

        velocities[i3] = (Math.random() - 0.5) * 0.3;
        velocities[i3 + 1] = (Math.random() - 0.5) * 0.3;
        velocities[i3 + 2] = (Math.random() - 0.5) * 0.3;

        // Initial scale for heartbeat effect
        scales[i] = 1.0;

        // Color distribution with vibrant medical accents
        const colorChoice = Math.random();
        const chosenColor = colorChoice < 0.25 ? color1 :
            colorChoice < 0.50 ? color2 :
                colorChoice < 0.75 ? color3 :
                    colorChoice < 0.90 ? colorMedical : colorAccent;

        colors[i3] = chosenColor.r;
        colors[i3 + 1] = chosenColor.g;
        colors[i3 + 2] = chosenColor.b;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));

    const material = new THREE.PointsMaterial({
        size: 4.2,  // Larger particles for visibility
        vertexColors: true,
        transparent: true,
        opacity: 0.85,  // More opaque
        blending: THREE.AdditiveBlending,  // Additive for glow effect
        depthWrite: false
    });

    particles = new THREE.Points(geometry, material);
    scene.add(particles);

    // Connection lines
    const maxConnections = particleCount * 2;
    const linePositions = new Float32Array(maxConnections * 3);
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));

    const lineMaterial = new THREE.LineBasicMaterial({
        color: 0x5dd9c1,  // Vibrant cyan-green
        transparent: true,
        opacity: 0.25,  // More visible connections
        blending: THREE.AdditiveBlending
    });

    linesMesh = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(linesMesh);

    // Medical grid overlay (subtle cross pattern)
    createMedicalGrid();

    // ECG-like wave visualization
    createECGWave();

    renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: false
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x0d1117, 1);

    container.appendChild(renderer.domElement);

    document.addEventListener('mousemove', handleMouseMove, false);
    window.addEventListener('resize', handleResize, false);

    animate();
}

/**
 * Create medical cross grid overlay (enhanced visibility)
 */
function createMedicalGrid() {
    const gridGeometry = new THREE.BufferGeometry();
    const gridPositions = [];
    const gridCount = 8;
    const gridSpacing = 300;

    // Create visible cross markers across the scene
    for (let x = -gridCount; x <= gridCount; x++) {
        for (let y = -gridCount; y <= gridCount; y++) {
            const posX = x * gridSpacing;
            const posY = y * gridSpacing;
            const crossSize = 14;  // Slightly larger crosses

            // Horizontal line of cross
            gridPositions.push(posX - crossSize, posY, -100);
            gridPositions.push(posX + crossSize, posY, -100);

            // Vertical line of cross
            gridPositions.push(posX, posY - crossSize, -100);
            gridPositions.push(posX, posY + crossSize, -100);
        }
    }

    gridGeometry.setAttribute('position',
        new THREE.Float32BufferAttribute(gridPositions, 3));

    const gridMaterial = new THREE.LineBasicMaterial({
        color: 0x5dd9c1,  // Vibrant cyan
        transparent: true,
        opacity: 0.12,  // More visible
        blending: THREE.AdditiveBlending
    });

    medicalGrid = new THREE.LineSegments(gridGeometry, gridMaterial);
    scene.add(medicalGrid);
}

/**
 * Create ECG wave line (heartbeat visualization)
 */
function createECGWave() {
    const wavePoints = 200;
    const wavePositions = new Float32Array(wavePoints * 3);

    for (let i = 0; i < wavePoints; i++) {
        const x = (i / wavePoints) * 2000 - 1000;
        wavePositions[i * 3] = x;
        wavePositions[i * 3 + 1] = 0;
        wavePositions[i * 3 + 2] = -200;
    }

    const waveGeometry = new THREE.BufferGeometry();
    waveGeometry.setAttribute('position', new THREE.BufferAttribute(wavePositions, 3));

    const waveMaterial = new THREE.LineBasicMaterial({
        color: 0x5dd9c1,  // Vibrant cyan
        transparent: true,
        opacity: 0.22,  // Much more visible
        linewidth: 3
    });

    ecgLine = new THREE.Line(waveGeometry, waveMaterial);
    scene.add(ecgLine);
}

/**
 * Handle mouse movement for parallax
 */
function handleMouseMove(event) {
    targetMouseX = (event.clientX / window.innerWidth) * 2 - 1;
    targetMouseY = -(event.clientY / window.innerHeight) * 2 + 1;
}

/**
 * Animation loop with medical-themed dynamics
 */
function animate() {
    requestAnimationFrame(animate);

    // Camera parallax
    mouseX += (targetMouseX * 120 - mouseX) * 0.03;
    mouseY += (targetMouseY * 120 - mouseY) * 0.03;

    camera.position.x = mouseX;
    camera.position.y = mouseY;
    camera.lookAt(scene.position);

    particles.rotation.x += rotationSpeed;
    particles.rotation.y += rotationSpeed * 1.3;

    // Medical heartbeat pulse effect (60 BPM rhythm)
    heartbeatPhase += 0.08;
    const heartbeat = Math.sin(heartbeatPhase) * 0.5 + 0.5; // 0 to 1
    const pulseIntensity = heartbeat > 0.85 ? (heartbeat - 0.85) * 8 : 0; // Sharp pulse

    // Apply heartbeat to particle sizes (more visible pulse)
    const material = particles.material;
    material.size = 4.2 + pulseIntensity * 1.2;

    // Particle motion
    const positions = particles.geometry.attributes.position.array;
    const velocities = particles.geometry.attributes.velocity.array;

    for (let i = 0; i < positions.length; i += 3) {
        positions[i] += velocities[i];
        positions[i + 1] += velocities[i + 1];
        positions[i + 2] += velocities[i + 2];

        if (Math.abs(positions[i]) > 900) velocities[i] *= -1;
        if (Math.abs(positions[i + 1]) > 900) velocities[i + 1] *= -1;
        if (Math.abs(positions[i + 2]) > 600) velocities[i + 2] *= -1;
    }

    particles.geometry.attributes.position.needsUpdate = true;

    // Update ECG wave to simulate heartbeat
    updateECGWave(heartbeatPhase);

    // Enhanced breathing effect on medical grid
    if (medicalGrid) {
        medicalGrid.material.opacity = 0.12 + Math.sin(heartbeatPhase * 0.3) * 0.04;
    }

    updateParticleConnections();

    renderer.render(scene, camera);
}

/**
 * Update ECG wave to show heartbeat pattern
 */
function updateECGWave(phase) {
    if (!ecgLine) return;

    const positions = ecgLine.geometry.attributes.position.array;
    const wavePoints = positions.length / 3;

    for (let i = 0; i < wavePoints; i++) {
        const x = (i / wavePoints) * 2000 - 1000;
        const normalizedX = (i / wavePoints + phase * 0.05) % 1;

        // ECG heartbeat pattern (simplified)
        let y = 0;
        if (normalizedX > 0.1 && normalizedX < 0.15) {
            // P wave
            y = Math.sin((normalizedX - 0.1) / 0.05 * Math.PI) * 15;
        } else if (normalizedX > 0.2 && normalizedX < 0.25) {
            // QRS complex (sharp spike)
            y = Math.sin((normalizedX - 0.2) / 0.05 * Math.PI) * 60;
        } else if (normalizedX > 0.3 && normalizedX < 0.38) {
            // T wave
            y = Math.sin((normalizedX - 0.3) / 0.08 * Math.PI) * 20;
        }

        positions[i * 3] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = -200;
    }

    ecgLine.geometry.attributes.position.needsUpdate = true;
}

/**
 * Draw connection lines between nearby particles
 */
function updateParticleConnections() {
    const positions = particles.geometry.attributes.position.array;
    const linePositions = linesMesh.geometry.attributes.position.array;

    let lineIndex = 0;
    const maxDistance = 180;

    for (let i = 0; i < positions.length && lineIndex < linePositions.length - 6; i += 3) {
        const x1 = positions[i];
        const y1 = positions[i + 1];
        const z1 = positions[i + 2];

        for (let j = i + 3; j < positions.length && lineIndex < linePositions.length - 6; j += 3) {
            const x2 = positions[j];
            const y2 = positions[j + 1];
            const z2 = positions[j + 2];

            const dx = x1 - x2;
            const dy = y1 - y2;
            const dz = z1 - z2;
            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

            if (distance < maxDistance) {
                linePositions[lineIndex++] = x1;
                linePositions[lineIndex++] = y1;
                linePositions[lineIndex++] = z1;
                linePositions[lineIndex++] = x2;
                linePositions[lineIndex++] = y2;
                linePositions[lineIndex++] = z2;
            }
        }
    }

    for (let i = lineIndex; i < linePositions.length; i++) {
        linePositions[i] = 0;
    }

    linesMesh.geometry.attributes.position.needsUpdate = true;
    linesMesh.geometry.setDrawRange(0, lineIndex / 3);
}

/**
 * Handle window resize
 */
function handleResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

/**
 * Reduce background motion when content sections are in view
 * Keep it visible but calmer
 */
function quietBackground() {
    rotationSpeed = 0.00008;
    if (linesMesh) linesMesh.material.opacity = 0.15; // Keep connections visible
    document.body.classList.add('content-in-view');
}

/**
 * Restore background intensity
 */
function restoreBackground() {
    rotationSpeed = 0.0003;
    if (linesMesh) linesMesh.material.opacity = 0.25; // Full visibility
    document.body.classList.remove('content-in-view');
}

// ================================================
// Scroll-Driven Reveal System
// ================================================

/**
 * Initialize scroll-based reveal for all sections
 */
function initScrollReveal() {
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (prefersReducedMotion) {
        revealAllElements();
        return;
    }

    const observerOptions = {
        root: null,
        rootMargin: '-60px 0px -120px 0px',
        threshold: 0.15
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    const trackedElements = document.querySelectorAll('[data-scroll-track]');
    trackedElements.forEach(el => observer.observe(el));
}

/**
 * Reveal all elements instantly (reduced motion)
 */
function revealAllElements() {
    const elements = document.querySelectorAll('[data-scroll-track]');
    elements.forEach(el => el.classList.add('visible'));
}

// ================================================
// Section-Based Background Control
// ================================================

/**
 * Monitor content sections to control background intensity
 */
function observeContentSections() {
    const contentSections = [
        document.querySelector('.overview-section'),
        document.querySelector('.features-section'),
        document.querySelector('.team-section'),
        document.querySelector('.cta-section')
    ];

    const observer = new IntersectionObserver(
        (entries) => {
            const anyContentVisible = entries.some(entry => entry.isIntersecting);
            if (anyContentVisible) {
                quietBackground();
            } else {
                restoreBackground();
            }
        },
        { threshold: 0.2 }
    );

    contentSections.forEach(section => {
        if (section) observer.observe(section);
    });
}

// ================================================
// Hero "Get Started" Button Smooth Scroll
// ================================================

/**
 * Smooth scroll to overview section when "Get Started" is clicked
 */
function initHeroButton() {
    const getStartedBtn = document.getElementById('get-started-btn');
    const overviewSection = document.getElementById('overview');

    if (getStartedBtn && overviewSection) {
        getStartedBtn.addEventListener('click', () => {
            overviewSection.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        });
    }
}

// ================================================
// Launch Button Interaction
// ================================================

/**
 * Handle launch system button click
 */
function initLaunchButton() {
    const launchBtn = document.getElementById('launch-btn');

    if (launchBtn) {
        launchBtn.addEventListener('click', () => {
            // Navigate to main application
            sessionStorage.setItem('from_landing', 'true');
            window.location.href = 'app.html';
        });
    }
}

// ================================================
// Performance Optimizations
// ================================================

/**
 * Pause rendering when page is hidden
 */
function handleVisibilityChange() {
    if (document.hidden) {
        cancelAnimationFrame(animate);
    } else {
        animate();
    }
}

// ================================================
// Initialization
// ================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Three.js background
    initThreeBackground();

    // Initialize scroll reveal system
    initScrollReveal();

    // Observe content sections for background control
    observeContentSections();

    // Initialize button interactions
    initHeroButton();
    initLaunchButton();

    // Performance: pause when tab hidden
    document.addEventListener('visibilitychange', handleVisibilityChange);

    console.log('✓ ClinXAI Landing Page Initialized');
});
