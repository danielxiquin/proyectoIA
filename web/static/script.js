/* ══════════════════════════════════════════════════════════
   IntelliDesk — Support Console · script.js
   ══════════════════════════════════════════════════════════ */

// ── State ─────────────────────────────────────────────────
const tickets       = [];
let selectedPriority = 'normal';
let currentFilter   = 'all';

// ── Category helpers — lee desde config.js ────────────────
// CATEGORIES, CAT_COLOR y CAT_DISPLAY vienen de /static/config.js
function getCatColor(key) {
  return CAT_COLOR[key] || CAT_COLOR[CAT_DISPLAY[key]] || '#888888';
}
function getCatDisplay(key) {
  return CAT_DISPLAY[key] || key;
}

// ── Init ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  generateTicketId();
  document.getElementById('current-date').textContent = formatDate(new Date());

  document.getElementById('description').addEventListener('input', function () {
    document.getElementById('char-count').textContent = this.value.length;
  });
});

function generateTicketId() {
  const id = 'TKT-' + String(Math.floor(Math.random() * 90000 + 10000));
  document.getElementById('current-ticket-id').textContent = id;
  document.getElementById('ti-id').textContent = id;
  return id;
}

// ── Navigation ────────────────────────────────────────────
function navigate(view) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('view-' + view).classList.add('active');
  document.querySelector(`[data-view="${view}"]`).classList.add('active');
  if (view === 'dashboard') refreshDashboard();
  if (view === 'history')   renderHistory();
  if (view === 'model')     loadModelStats();
}

// ── AI Model Stats ────────────────────────────────────────
let modelStatsLoaded = false;

async function loadModelStats() {
  if (modelStatsLoaded) return;   // only fetch once
  try {
    const res  = await fetch('/model-stats');
    const data = await res.json();

    document.getElementById('model-loading').classList.add('hidden');

    if (!data.has_eval) {
      document.getElementById('model-no-eval').classList.remove('hidden');
      return;
    }

    // Fill global cards
    document.getElementById('mc-accuracy').textContent = (data.accuracy * 100).toFixed(2) + '%';
    document.getElementById('mc-f1').textContent       = (data.macro_f1 * 100).toFixed(2) + '%';
    document.getElementById('mc-docs').textContent     = data.n_docs.toLocaleString();
    document.getElementById('mc-vocab').textContent    = data.vocab_size.toLocaleString();
    document.getElementById('mc-classes').textContent  = data.classes.length;
    document.getElementById('mc-k').textContent        = `${data.k_folds}-fold cross validation`;

    // Per-class table
    const tbody = document.getElementById('model-table-rows');
    const sorted = data.classes.slice().sort((a, b) =>
      (data.per_class[b]?.f1 || 0) - (data.per_class[a]?.f1 || 0)
    );
    tbody.innerHTML = sorted.map(cls => {
      const m   = data.per_class[cls] || { precision: 0, recall: 0, f1: 0 };
      const name = data.category_display?.[cls] || cls;
      const f1pct = (m.f1 * 100).toFixed(1);
      return `
        <div class="model-table-row">
          <span style="font-weight:500">${name}</span>
          <span style="color:var(--neon-cyan)">${(m.precision * 100).toFixed(1)}%</span>
          <span style="color:var(--muted)">${(m.recall * 100).toFixed(1)}%</span>
          <span style="color:var(--neon-green);font-weight:600">${f1pct}%</span>
          <div style="display:flex;align-items:center;gap:.5rem">
            <div class="model-metric-bar-bg" style="flex:1">
              <div class="model-metric-bar-fill" style="width:${f1pct}%"></div>
            </div>
          </div>
        </div>`;
    }).join('');

    // Training distribution bars
    const distEl    = document.getElementById('model-dist-bars');
    const maxDocs   = Math.max(...Object.values(data.docs_per_class));
    distEl.innerHTML = sorted.map(cls => {
      const name  = data.category_display?.[cls] || cls;
      const count = data.docs_per_class[cls] || 0;
      const pct   = maxDocs ? (count / maxDocs * 100).toFixed(1) : 0;
      return `
        <div class="model-dist-item">
          <div class="model-dist-row">
            <span class="model-dist-name">${name}</span>
            <span>${count.toLocaleString()} docs</span>
          </div>
          <div class="model-dist-bar-bg">
            <div class="model-dist-bar-fill" style="width:${pct}%"></div>
          </div>
        </div>`;
    }).join('');

    document.getElementById('model-content').classList.remove('hidden');
    modelStatsLoaded = true;

  } catch (e) {
    document.getElementById('model-loading').textContent = 'Failed to load model stats.';
    console.error(e);
  }
}

// ── Priority selector ─────────────────────────────────────
function setPriority(btn) {
  document.querySelectorAll('.prio-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  selectedPriority = btn.dataset.prio;
}

// ── Example chips ─────────────────────────────────────────
function fillExample(text) {
  document.getElementById('description').value = text;
  document.getElementById('char-count').textContent = text.length;
}

// ── Filter (history) ──────────────────────────────────────
function setFilter(btn) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentFilter = btn.dataset.filter;
  renderHistory();
}

// ── Auto-classify only (no save) ──────────────────────────
async function classifyOnly() {
  const description = document.getElementById('description').value.trim();
  const subject     = document.getElementById('subject').value.trim();
  if (!description) { alert('Please describe your issue first.'); return; }
  await doClassify(subject, description, false);
}

// ── Submit + save ticket ──────────────────────────────────
async function submitTicket() {
  const description = document.getElementById('description').value.trim();
  const subject     = document.getElementById('subject').value.trim();
  if (!description) { alert('Please describe your issue.'); return; }
  await doClassify(subject, description, true);
}

// ── Core classify function ────────────────────────────────
async function doClassify(subject, description, save) {
  setLoading(true);
  try {
    const res = await fetch('/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subject, text: description })
    });
    const data = await res.json();
    if (!res.ok) { alert(data.error || 'Unknown error'); return; }

    showClassification(data);

    if (save) {
      const ticketId = document.getElementById('current-ticket-id').textContent;
      const ticket = {
        ...data,
        ticket_id: ticketId,
        subject:   subject || '(no subject)',
        priority:  selectedPriority,
        status:    selectedPriority === 'urgent' ? 'in-progress' : 'open',
        description,
        savedAt:   new Date()
      };
      tickets.unshift(ticket);
      updateSidebarCount();
      // Mostrar estado "Submitted" en el panel, sin borrar resultados
      showSubmittedState(ticketId);
    }
  } catch (e) {
    alert('Connection error. Is Flask running on port 5000?');
    console.error(e);
  } finally {
    setLoading(false);
  }
}

// ── Show classification result ────────────────────────────
function showClassification(data) {
  const color = data.color || getCatColor(data.predicted_class);

  // Top category pill
  document.getElementById('top-cat-dot').style.background  = color;
  document.getElementById('top-cat-name').textContent       = data.display_name;

  const probs      = Object.values(data.probabilities);
  const topConf    = probs.length ? probs[0] : 0;
  document.getElementById('top-cat-conf').textContent = `${topConf.toFixed(1)}% confidence`;

  // Probability bars
  const list = document.getElementById('prob-bars');
  list.innerHTML = '';
  Object.entries(data.probabilities).forEach(([name, pct]) => {
    const isTop    = (name === data.display_name);
    const barColor = isTop ? color : '#334155';
    list.innerHTML += `
      <div class="prob-item">
        <div class="prob-row">
          <span class="prob-name">${name}</span>
          <span>${pct.toFixed(1)}%</span>
        </div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" style="width:${pct}%;background:${barColor}"></div>
        </div>
      </div>`;
  });

  // Ticket info block
  document.getElementById('ti-date').textContent = formatDate(new Date());
  document.getElementById('classify-ticket-info').classList.remove('hidden');

  // Toggle panels
  document.getElementById('classify-empty').classList.add('hidden');
  document.getElementById('classify-result').classList.remove('hidden');
}

// ── Submitted state (ticket saved, results visible) ───────
function showSubmittedState(ticketId) {
  const info = document.getElementById('classify-ticket-info');
  info.classList.remove('hidden');
  // Change status from Draft to Open
  const rows = info.querySelectorAll('.ti-row');
  rows.forEach(row => {
    const label = row.querySelector('.ti-label');
    if (label && label.textContent === 'STATUS') {
      const val = row.querySelector('.ti-value');
      val.className = 'ti-value';
      val.style.color = '#39FF14';
      val.textContent = selectedPriority === 'urgent' ? 'In Progress' : 'Open';
    }
  });
  // Show new ticket button in the panel
  let newBtn = document.getElementById('new-ticket-btn');
  if (!newBtn) {
    newBtn = document.createElement('button');
    newBtn.id = 'new-ticket-btn';
    newBtn.className = 'btn-new-ticket';
    newBtn.textContent = '+ New Ticket';
    newBtn.onclick = resetForm;
    document.getElementById('classify-ticket-info').after(newBtn);
  }
  newBtn.classList.remove('hidden');
}

// ── Reset form ────────────────────────────────────────────
function resetForm() {
  document.getElementById('description').value = '';
  document.getElementById('subject').value     = '';
  document.getElementById('char-count').textContent = '0';
  document.getElementById('classify-empty').classList.remove('hidden');
  document.getElementById('classify-result').classList.add('hidden');
  document.getElementById('classify-ticket-info').classList.add('hidden');
  const newBtn = document.getElementById('new-ticket-btn');
  if (newBtn) newBtn.classList.add('hidden');
  generateTicketId();
  document.getElementById('current-date').textContent = formatDate(new Date());
}

// ── History table ─────────────────────────────────────────
function renderHistory() {
  const rows = document.getElementById('history-rows');
  const filtered = currentFilter === 'all'
    ? tickets
    : tickets.filter(t => t.status === currentFilter);

  if (!filtered.length) {
    rows.innerHTML = `<div class="table-empty">${
      tickets.length ? 'No tickets match that filter.' : 'No tickets yet in this session'
    }</div>`;
    document.getElementById('table-count').textContent = '';
    return;
  }

  const statusBadge = {
    'open':        '<span class="badge-status badge-open">Open</span>',
    'in-progress': '<span class="badge-status badge-progress">In Progress</span>',
    'resolved':    '<span class="badge-status badge-resolved">Resolved</span>',
  };

  rows.innerHTML = filtered.map(t => {
    const color = t.color || getCatColor(t.predicted_class);
    return `
      <div class="table-row">
        <span style="font-weight:600">${t.ticket_id}</span>
        <span style="color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding-right:.5rem">${t.subject}</span>
        <span>
          <span class="badge-category" style="background:${color}22;color:${color}">${t.display_name}</span>
        </span>
        <span>${statusBadge[t.status] || ''}</span>
        <span style="color:var(--dim)">${formatShortDate(t.savedAt)}</span>
      </div>`;
  }).join('');

  document.getElementById('table-count').textContent =
    `${filtered.length} ticket${filtered.length !== 1 ? 's' : ''}`;
}

// ── Dashboard ─────────────────────────────────────────────
function refreshDashboard() {
  const total    = tickets.length;
  const open     = tickets.filter(t => t.status === 'open').length;
  const progress = tickets.filter(t => t.status === 'in-progress').length;
  const resolved = tickets.filter(t => t.status === 'resolved').length;

  document.getElementById('dash-total').textContent    = total;
  document.getElementById('dash-open').textContent     = open;
  document.getElementById('dash-progress').textContent = progress;
  document.getElementById('dash-resolved').textContent = resolved;
  updateSidebarCount();

  // Category counts
  const cats = {};
  tickets.forEach(t => { cats[t.display_name] = (cats[t.display_name] || 0) + 1; });
  const catEl = document.getElementById('dash-categories');
  catEl.innerHTML = Object.keys(cats).length
    ? Object.entries(cats).sort((a, b) => b[1] - a[1]).map(([name, cnt]) => {
        const color = getCatColor(name);
        return `<span class="badge-category" style="background:${color}22;color:${color};padding:.3rem .75rem;border-radius:6px;font-size:.78rem">
          ${name} <strong style="margin-left:.3rem">${cnt}</strong>
        </span>`;
      }).join('')
    : '<p class="empty-text">No tickets yet.</p>';

  // Recent list
  const recentEl = document.getElementById('dash-recent');
  recentEl.innerHTML = tickets.length
    ? tickets.slice(0, 5).map(t => {
        const color = t.color || getCatColor(t.predicted_class);
        return `<div class="dash-recent-item">
          <span style="font-weight:600;font-size:.8rem;flex-shrink:0">${t.ticket_id}</span>
          <span style="color:var(--muted);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${t.subject}</span>
          <span style="color:${color};font-size:.75rem;font-weight:600;flex-shrink:0">${t.display_name}</span>
        </div>`;
      }).join('')
    : '<p class="empty-text">No tickets yet.</p>';
}

// ── Sidebar counter ───────────────────────────────────────
function updateSidebarCount() {
  const open   = tickets.filter(t => t.status === 'open' || t.status === 'in-progress').length;
  const urgent = tickets.filter(t => t.priority === 'urgent').length;
  document.getElementById('open-count').textContent = String(open).padStart(2, '0');
  const urgentLine = document.getElementById('urgent-line');
  if (urgent > 0) {
    document.getElementById('urgent-count').textContent = urgent;
    urgentLine.classList.remove('hidden');
  } else {
    urgentLine.classList.add('hidden');
  }
}

// ── Loading state ─────────────────────────────────────────
function setLoading(on) {
  const btn = document.getElementById('submit-btn');
  btn.disabled = on;
  document.getElementById('btn-text').classList.toggle('hidden', on);
  document.getElementById('btn-loading').classList.toggle('hidden', !on);
}

// ── Date helpers ──────────────────────────────────────────
function formatDate(d) {
  return d.toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit'
  });
}
function formatShortDate(d) {
  return new Date(d).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}
