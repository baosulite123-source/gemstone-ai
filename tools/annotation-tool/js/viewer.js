// viewer.js — Multi-image grid viewer + lightbox

const Viewer = (() => {
  const grid = document.getElementById('image-grid');
  const lightbox = document.getElementById('lightbox');
  let currentStone = null;

  const LIGHTING_ICONS = {
    D65: '☀️', transmitted: '💡', cross_polarized: '🔬',
    darkfield: '🌑', UV: '🟣', fluorescent: '💫'
  };

  function render(stone) {
    currentStone = stone;
    grid.innerHTML = '';

    if (!stone) {
      grid.innerHTML = '<div class="empty-state"><div class="icon">💎</div><p>Select a stone to view images</p></div>';
      return;
    }

    if (!stone.images || stone.images.length === 0) {
      grid.innerHTML = '<div class="empty-state"><div class="icon">📷</div><p>No images available for this stone</p></div>';
      return;
    }

    stone.images.forEach((img, idx) => {
      const card = document.createElement('div');
      card.className = 'image-card';
      card.dataset.index = idx;

      const icon = LIGHTING_ICONS[img.lighting] || '📸';
      const hasUrl = img.url && img.url.trim() !== '';

      card.innerHTML = `
        <div class="img-container" id="img-container-${idx}">
          ${hasUrl
            ? `<img src="${img.url}" alt="${img.lighting} ${img.angle}" loading="lazy">`
            : `<div class="img-placeholder">${icon}</div>`
          }
        </div>
        <div class="img-label">
          <span class="lighting">${icon} ${img.lighting.replace('_', ' ')}</span>
          <span class="angle">${img.angle}</span>
        </div>
      `;

      card.addEventListener('click', (e) => {
        // Don't open lightbox if annotation mode is on
        if (Annotations && Annotations.isActive()) return;
        openLightbox(img, icon);
      });

      grid.appendChild(card);

      // Let annotations module attach canvas if needed
      if (typeof Annotations !== 'undefined') {
        Annotations.attachToCard(card, stone.id, idx);
      }
    });
  }

  function openLightbox(img, icon) {
    const hasUrl = img.url && img.url.trim() !== '';
    lightbox.innerHTML = `
      ${hasUrl
        ? `<img src="${img.url}" alt="${img.lighting}">`
        : `<div class="lb-placeholder">${icon}</div>`
      }
      <div class="lb-info">${img.lighting.replace('_', ' ').toUpperCase()} — ${img.angle} view</div>
    `;
    lightbox.classList.add('active');
  }

  lightbox.addEventListener('click', () => lightbox.classList.remove('active'));

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') lightbox.classList.remove('active');
  });

  return { render };
})();
