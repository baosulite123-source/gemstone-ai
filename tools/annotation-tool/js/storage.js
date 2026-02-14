// storage.js — localStorage persistence + JSON import/export

const Storage = (() => {
  const GRADES_KEY = 'gemgrade_grades';
  const STONES_KEY = 'gemgrade_stones';

  function getGrades() {
    try { return JSON.parse(localStorage.getItem(GRADES_KEY)) || {}; } catch { return {}; }
  }

  function saveGrade(stoneId, grade) {
    const grades = getGrades();
    grades[stoneId] = { ...grade, timestamp: new Date().toISOString() };
    localStorage.setItem(GRADES_KEY, JSON.stringify(grades));
  }

  function getGrade(stoneId) {
    return getGrades()[stoneId] || null;
  }

  function isGraded(stoneId) {
    return !!getGrades()[stoneId];
  }

  function gradedCount(stoneIds) {
    const grades = getGrades();
    return stoneIds.filter(id => grades[id]).length;
  }

  function exportGrades(stones) {
    const grades = getGrades();
    const data = {
      exportDate: new Date().toISOString(),
      version: '1.0',
      grades: stones.map(s => ({
        stoneId: s.id,
        lotId: s.lotId,
        variety: s.variety,
        weight: s.weight,
        origin: s.origin,
        grade: grades[s.id] || null
      }))
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `gemstone-grades-${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function importStones(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = e => {
        try {
          const data = JSON.parse(e.target.result);
          // Accept array directly or {stones: [...]}
          const stones = Array.isArray(data) ? data : data.stones;
          if (!Array.isArray(stones)) throw new Error('Invalid format');
          resolve(stones);
        } catch (err) { reject(err); }
      };
      reader.onerror = reject;
      reader.readAsText(file);
    });
  }

  async function loadSampleStones() {
    const resp = await fetch('data/sample-stones.json');
    return resp.json();
  }

  return { getGrades, saveGrade, getGrade, isGraded, gradedCount, exportGrades, importStones, loadSampleStones };
})();
