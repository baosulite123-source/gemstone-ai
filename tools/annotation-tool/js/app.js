// app.js — Main application logic

const App = (() => {
  let stones = [];
  let filteredStones = [];
  let currentIndex = -1;
  let filter = 'all';
  let searchQuery = '';

  async function init() {
    Grading.init();
    stones = await Storage.loadSampleStones();
    applyFilter();
    renderStoneList();
    updateProgress();
    bindEvents();
    if (filteredStones.length > 0) selectStone(0);
  }

  function bindEvents() {
    // Import
    document.getElementById('btn-import').addEventListener('click', () => document.getElementById('file-import').click());
    document.getElementById('file-import').addEventListener('change', async e => {
      if (!e.target.files[0]) return;
      try {
        stones = await Storage.importStones(e.target.files[0]);
        applyFilter();
        renderStoneList();
        updateProgress();
        if (filteredStones.length > 0) selectStone(0);
        toast('✅ Imported ' + stones.length + ' stones');
      } catch (err) { toast('❌ Import failed: ' + err.message); }
    });

    // Export
    document.getElementById('btn-export').addEventListener('click', () => {
      Storage.exportGrades(stones);
      toast('📦 Grades exported');
    });

    // Search
    document.getElementById('search-input').addEventListener('input', e => {
      searchQuery = e.target.value.toLowerCase();
      applyFilter();
      renderStoneList();
    });

    // Filter chips
    document.querySelectorAll('.filter-row .chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('.filter-row .chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        filter = chip.dataset.filter;
        applyFilter();
        renderStoneList();
        if (filteredStones.length > 0) selectStone(0);
      });
    });

    // Save
    document.getElementById('btn-save').addEventListener('click', saveGrade);

    // Skip
    document.getElementById('btn-skip').addEventListener('click', () => navigateStone(1));

    // Nav
    document.getElementById('btn-prev').addEventListener('click', () => navigateStone(-1));
    document.getElementById('btn-next').addEventListener('click', () => navigateStone(1));

    // Annotate toggle
    document.getElementById('btn-annotate').addEventListener('click', () => {
      const on = Annotations.toggle();
      document.getElementById('btn-annotate').classList.toggle('btn-primary', on);
      toast(on ? '✏️ Annotation mode ON — click to place points, double-click to close polygon' : '✏️ Annotation mode OFF');
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); navigateStone(-1); }
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); navigateStone(1); }
      if (e.key === 'Enter') { e.preventDefault(); saveGrade(); }
      if (e.key === 's' && !e.ctrlKey) { e.preventDefault(); navigateStone(1); } // skip
      if (e.key === 'Escape') { Annotations.clearCurrentPoints(); }
    });
  }

  function applyFilter() {
    filteredStones = stones.filter(s => {
      if (filter !== 'all' && s.variety !== filter) return false;
      if (searchQuery && !s.id.toLowerCase().includes(searchQuery) && !(s.origin||'').toLowerCase().includes(searchQuery)) return false;
      return true;
    });
  }

  function renderStoneList() {
    const list = document.getElementById('stone-list');
    list.innerHTML = '';
    filteredStones.forEach((stone, idx) => {
      const div = document.createElement('div');
      div.className = 'stone-item' + (idx === currentIndex ? ' active' : '');
      const graded = Storage.isGraded(stone.id);
      div.innerHTML = `
        <div>
          <div class="stone-id">${stone.id}</div>
          <div class="stone-meta">${stone.weight} ct · ${stone.shape}</div>
        </div>
        <div style="display:flex;align-items:center;">
          <span class="badge ${stone.variety}">${stone.variety === 'blue_sapphire' ? 'BS' : 'RB'}</span>
          ${graded ? '<span class="graded-check">✓</span>' : ''}
        </div>
      `;
      div.addEventListener('click', () => selectStone(idx));
      list.appendChild(div);
    });
  }

  function selectStone(idx) {
    if (idx < 0 || idx >= filteredStones.length) return;
    currentIndex = idx;
    const stone = filteredStones[idx];

    // Update list selection
    document.querySelectorAll('.stone-item').forEach((el, i) => {
      el.classList.toggle('active', i === idx);
    });

    // Metadata
    renderMetadata(stone);

    // Viewer
    Viewer.render(stone);

    // Grading form
    Grading.loadForStone(stone.id);
  }

  function renderMetadata(stone) {
    const el = document.getElementById('stone-metadata');
    const varClass = stone.variety === 'blue_sapphire' ? 'blue_sapphire' : 'ruby';
    const varLabel = stone.variety === 'blue_sapphire' ? 'Blue Sapphire' : 'Ruby';
    el.innerHTML = `
      <span class="variety-tag ${varClass}">${varLabel}</span>
      <span class="meta-item"><span class="label">ID:</span><span class="value">${stone.id}</span></span>
      <span class="meta-item"><span class="label">Lot:</span><span class="value">${stone.lotId}</span></span>
      <span class="meta-item"><span class="label">Weight:</span><span class="value">${stone.weight} ct</span></span>
      <span class="meta-item"><span class="label">Dims:</span><span class="value">${stone.dimensions} mm</span></span>
      <span class="meta-item"><span class="label">Shape:</span><span class="value">${stone.shape}</span></span>
      <span class="meta-item"><span class="label">Origin:</span><span class="value">${stone.origin}</span></span>
      <span class="meta-item"><span class="label">Treatment:</span><span class="value">${stone.treatment}</span></span>
    `;
  }

  function navigateStone(dir) {
    const next = currentIndex + dir;
    if (next >= 0 && next < filteredStones.length) selectStone(next);
  }

  function saveGrade() {
    if (Grading.saveCurrentGrade()) {
      toast('✅ Grade saved for ' + filteredStones[currentIndex].id);
      updateProgress();
      renderStoneList();
      // Auto-advance
      navigateStone(1);
    }
  }

  function updateProgress() {
    const total = stones.length;
    const graded = Storage.gradedCount(stones.map(s => s.id));
    document.getElementById('progress-text').innerHTML = `<span class="count">${graded}</span> / ${total} graded`;
  }

  return { init };
})();

// Toast helper
function toast(msg) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 2500);
}

// Boot
document.addEventListener('DOMContentLoaded', () => App.init());
