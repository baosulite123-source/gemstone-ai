// grading.js — Grading form logic

const Grading = (() => {
  let currentStoneId = null;

  const SAT_LABELS = ['', 'Grayish', 'Slightly grayish', 'Very slightly grayish', 'Moderately strong', 'Strong', 'Vivid'];
  const TONE_LABELS = ['', 'Very light', 'Light', 'Medium light', 'Medium', 'Medium dark', 'Dark', 'Very dark'];
  const DENSITY_LABELS = ['', 'Insignificant', 'Minor', 'Moderate', 'Heavy', 'Severe'];

  function init() {
    // Slider value displays
    document.getElementById('saturation').addEventListener('input', e => {
      document.getElementById('sat-value').textContent = `${e.target.value} — ${SAT_LABELS[e.target.value]}`;
    });
    document.getElementById('tone').addEventListener('input', e => {
      document.getElementById('tone-value').textContent = `${e.target.value} — ${TONE_LABELS[e.target.value]}`;
    });
    document.getElementById('inclusion-density').addEventListener('input', e => {
      document.getElementById('density-value').textContent = `${e.target.value} — ${DENSITY_LABELS[e.target.value]}`;
    });
  }

  function loadForStone(stoneId) {
    currentStoneId = stoneId;
    const grade = Storage.getGrade(stoneId);
    if (grade) {
      setFormValues(grade);
    } else {
      resetForm();
    }
  }

  function setFormValues(g) {
    document.getElementById('hue').value = g.hue || '';
    const sat = document.getElementById('saturation'); sat.value = g.saturation || 3; sat.dispatchEvent(new Event('input'));
    const tone = document.getElementById('tone'); tone.value = g.tone || 4; tone.dispatchEvent(new Event('input'));
    setRadio('color-distribution', g.colorDistribution || '');
    setRadio('special-designation', g.specialDesignation || 'none');
    setRadio('transparency', g.transparency || '');
    setCheckboxes('inclusion-types', g.inclusionTypes || []);
    const den = document.getElementById('inclusion-density'); den.value = g.inclusionDensity || 1; den.dispatchEvent(new Event('input'));
    setRadio('inclusion-location', g.inclusionLocation || '');
    setRadio('fracture-risk', g.fractureRisk || '');
    document.getElementById('grading-notes').value = g.notes || '';
  }

  function resetForm() {
    document.getElementById('hue').value = '';
    const sat = document.getElementById('saturation'); sat.value = 3; sat.dispatchEvent(new Event('input'));
    const tone = document.getElementById('tone'); tone.value = 4; tone.dispatchEvent(new Event('input'));
    clearRadios('color-distribution');
    setRadio('special-designation', 'none');
    clearRadios('transparency');
    clearCheckboxes('inclusion-types');
    const den = document.getElementById('inclusion-density'); den.value = 1; den.dispatchEvent(new Event('input'));
    clearRadios('inclusion-location');
    clearRadios('fracture-risk');
    document.getElementById('grading-notes').value = '';
  }

  function collectGrade() {
    return {
      hue: document.getElementById('hue').value,
      saturation: parseInt(document.getElementById('saturation').value),
      tone: parseInt(document.getElementById('tone').value),
      colorDistribution: getRadio('color-distribution'),
      specialDesignation: getRadio('special-designation') || 'none',
      transparency: getRadio('transparency'),
      inclusionTypes: getCheckboxes('inclusion-types'),
      inclusionDensity: parseInt(document.getElementById('inclusion-density').value),
      inclusionLocation: getRadio('inclusion-location'),
      fractureRisk: getRadio('fracture-risk'),
      notes: document.getElementById('grading-notes').value.trim()
    };
  }

  function saveCurrentGrade() {
    if (!currentStoneId) return false;
    const grade = collectGrade();
    Storage.saveGrade(currentStoneId, grade);
    return true;
  }

  // Radio helpers
  function setRadio(name, value) {
    const el = document.querySelector(`input[name="${name}"][value="${value}"]`);
    if (el) el.checked = true;
  }
  function getRadio(name) {
    const el = document.querySelector(`input[name="${name}"]:checked`);
    return el ? el.value : '';
  }
  function clearRadios(name) {
    document.querySelectorAll(`input[name="${name}"]`).forEach(el => el.checked = false);
  }

  // Checkbox helpers
  function setCheckboxes(name, values) {
    document.querySelectorAll(`input[name="${name}"]`).forEach(el => {
      el.checked = values.includes(el.value);
    });
  }
  function getCheckboxes(name) {
    return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map(el => el.value);
  }
  function clearCheckboxes(name) {
    document.querySelectorAll(`input[name="${name}"]`).forEach(el => el.checked = false);
  }

  return { init, loadForStone, saveCurrentGrade, resetForm, collectGrade };
})();
