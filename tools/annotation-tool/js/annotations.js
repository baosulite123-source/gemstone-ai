// annotations.js — Color zone polygon annotation on images

const Annotations = (() => {
  let active = false;
  let currentPoints = [];
  let allAnnotations = {}; // { stoneId: { imageIdx: [{ points, color, label }] } }
  const COLORS = ['#e94560', '#4a9eff', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];
  let colorIdx = 0;

  function isActive() { return active; }

  function toggle() {
    active = !active;
    currentPoints = [];
    document.querySelectorAll('.annotation-canvas').forEach(c => {
      c.style.display = active ? 'block' : 'none';
    });
    document.querySelectorAll('.annotation-mode-badge').forEach(b => {
      b.style.display = active ? 'block' : 'none';
    });
    return active;
  }

  function attachToCard(card, stoneId, imageIdx) {
    const container = card.querySelector('.img-container');
    const canvas = document.createElement('canvas');
    canvas.className = 'annotation-canvas';
    canvas.style.display = active ? 'block' : 'none';
    container.appendChild(canvas);

    const badge = document.createElement('div');
    badge.className = 'annotation-mode-badge';
    badge.textContent = '✏️ Annotate';
    badge.style.display = active ? 'block' : 'none';
    container.appendChild(badge);

    function resize() {
      canvas.width = container.offsetWidth;
      canvas.height = container.offsetHeight;
      redraw(canvas, stoneId, imageIdx);
    }

    new ResizeObserver(resize).observe(container);
    setTimeout(resize, 100);

    canvas.addEventListener('click', e => {
      if (!active) return;
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;
      currentPoints.push({ x, y });

      if (currentPoints.length >= 3) {
        // Draw in-progress polygon
        drawPolygon(canvas, currentPoints, COLORS[colorIdx % COLORS.length], false);
      }
    });

    canvas.addEventListener('dblclick', e => {
      if (!active || currentPoints.length < 3) return;
      e.preventDefault();
      const color = COLORS[colorIdx % COLORS.length];
      const label = prompt('Color zone label (e.g., "primary red", "blue zone"):') || `Zone ${(getAnns(stoneId, imageIdx).length + 1)}`;
      if (!allAnnotations[stoneId]) allAnnotations[stoneId] = {};
      if (!allAnnotations[stoneId][imageIdx]) allAnnotations[stoneId][imageIdx] = [];
      allAnnotations[stoneId][imageIdx].push({ points: [...currentPoints], color, label });
      currentPoints = [];
      colorIdx++;
      redraw(canvas, stoneId, imageIdx);
    });
  }

  function getAnns(stoneId, imageIdx) {
    return (allAnnotations[stoneId] && allAnnotations[stoneId][imageIdx]) || [];
  }

  function redraw(canvas, stoneId, imageIdx) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const anns = getAnns(stoneId, imageIdx);
    anns.forEach(ann => drawPolygon(canvas, ann.points, ann.color, true, ann.label));
    if (currentPoints.length > 0) {
      drawPolygon(canvas, currentPoints, COLORS[colorIdx % COLORS.length], false);
    }
  }

  function drawPolygon(canvas, points, color, closed, label) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.beginPath();
    ctx.moveTo(points[0].x * w, points[0].y * h);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x * w, points[i].y * h);
    }
    if (closed) ctx.closePath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    if (closed) {
      ctx.fillStyle = color + '33';
      ctx.fill();
      if (label) {
        const cx = points.reduce((s, p) => s + p.x, 0) / points.length * w;
        const cy = points.reduce((s, p) => s + p.y, 0) / points.length * h;
        ctx.fillStyle = '#fff';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(label, cx, cy);
      }
    }
    // Draw points
    points.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    });
  }

  function clearCurrentPoints() { currentPoints = []; }

  return { isActive, toggle, attachToCard, clearCurrentPoints };
})();
