// Visual-only enhancement: real Gradio controls retain every inference event.
// Position the existing panels near their triggers and provide dialog-like dismissal.
(() => {
  const anchors = {
    settings: 'Settings', variants: 'Variants', size: 'Size',
    model: 'Model', swap: 'Replace', bfs: 'BFS weight',
  };
  const panelIds = Object.keys(anchors).map((name) => `studio-panel-${name}`);
  let activePanel = null;
  let detailsVisible = false;
  let raf = 0;

  function shell() {
    return document.querySelector('gradio-app')?.shadowRoot ?? document;
  }

  function findTrigger(name) {
    const toolbar = shell().querySelector('#studio-prompt-toolbar');
    if (!toolbar) return null;
    const label = anchors[name];
    return Array.from(toolbar.querySelectorAll('button')).find((button) => {
      const text = (button.textContent || '').trim();
      return name === 'model' ? text.includes('▾') : text === label;
    });
  }

  function visible(panel) {
    return panel && panel.getClientRects().length > 0 &&
      getComputedStyle(panel).display !== 'none';
  }

  function positionDetails(root) {
    const details = root.querySelector('#studio-run-details');
    const trigger = root.querySelector('#studio-details-trigger button, button#studio-details-trigger');
    const open = visible(details);
    if (details && trigger && open) {
      const bounds = trigger.getBoundingClientRect();
      const width = Math.min(380, window.innerWidth - 24);
      const left = Math.min(Math.max(12, bounds.right - width), window.innerWidth - width - 12);
      const top = Math.min(bounds.bottom + 10, Math.max(12, window.innerHeight - Math.min(details.scrollHeight, 610) - 12));
      if (details.style.getPropertyValue('--studio-details-left') !== `${left}px`) details.style.setProperty('--studio-details-left', `${left}px`);
      if (details.style.getPropertyValue('--studio-details-top') !== `${top}px`) details.style.setProperty('--studio-details-top', `${top}px`);
    }
    if (trigger && trigger.getAttribute('aria-expanded') !== String(Boolean(open))) {
      trigger.setAttribute('aria-expanded', String(Boolean(open)));
    }
    if (trigger?.getAttribute('aria-controls') !== 'studio-run-details') {
      trigger?.setAttribute('aria-controls', 'studio-run-details');
    }
    detailsVisible = Boolean(open);
  }

  function positionPanel() {
    const root = shell();
    positionDetails(root);
    const open = panelIds.map((id) => root.querySelector(`#${id}`)).find(visible);
    panelIds.forEach((id) => {
      const panel = root.querySelector(`#${id}`);
      if (panel && panel !== open && panel.classList.contains('studio-popover-active')) {
        panel.classList.remove('studio-popover-active');
      }
    });
    const next = open?.id?.replace('studio-panel-', '') ?? null;
    if (!open || !next) {
      Object.keys(anchors).forEach((name) => {
        const trigger = findTrigger(name);
        if (trigger?.getAttribute('aria-expanded') !== 'false') {
          trigger?.setAttribute('aria-expanded', 'false');
        }
      });
      activePanel = null;
      return;
    }
    activePanel = next;
    const trigger = findTrigger(next);
    if (!trigger) return;
    if (!open.classList.contains('studio-popover-active')) open.classList.add('studio-popover-active');
    const box = trigger.getBoundingClientRect();
    const width = Math.min(420, window.innerWidth - 24);
    const x = Math.min(window.innerWidth - width - 12, Math.max(12, box.left));
    const below = box.bottom + 10;
    const height = Math.min(open.scrollHeight, window.innerHeight * .65);
    const top = below + height > window.innerHeight - 12
      ? Math.max(12, box.top - height - 10) : below;
    if (open.style.getPropertyValue('--studio-popup-left') !== `${x}px`) {
      open.style.setProperty('--studio-popup-left', `${x}px`);
    }
    if (open.style.getPropertyValue('--studio-popup-top') !== `${top}px`) {
      open.style.setProperty('--studio-popup-top', `${top}px`);
    }
    if (trigger.getAttribute('aria-expanded') !== 'true') trigger.setAttribute('aria-expanded', 'true');
    if (trigger.getAttribute('aria-controls') !== open.id) trigger.setAttribute('aria-controls', open.id);
    Object.keys(anchors).filter((name) => name !== next).forEach((name) => {
      const other = findTrigger(name);
      if (other?.getAttribute('aria-expanded') !== 'false') other?.setAttribute('aria-expanded', 'false');
    });
  }

  function refresh() {
    if (raf) return;
    raf = requestAnimationFrame(() => { raf = 0; positionPanel(); });
  }

  const observer = new MutationObserver(refresh);
  const start = () => {
    const root = shell();
    observer.observe(root, {childList: true, subtree: true, attributes: true,
      attributeFilter: ['class', 'style', 'hidden']});
    root.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape') return;
      const trigger = detailsVisible
        ? root.querySelector('#studio-details-trigger button, button#studio-details-trigger')
        : activePanel ? findTrigger(activePanel) : null;
      if (trigger) { trigger.click(); trigger.focus(); event.preventDefault(); }
    });
    root.addEventListener('click', (event) => {
      const details = root.querySelector('#studio-run-details');
      const detailsTrigger = root.querySelector('#studio-details-trigger button, button#studio-details-trigger');
      if (detailsVisible && details && detailsTrigger &&
          !details.contains(event.target) && !detailsTrigger.contains(event.target)) {
        detailsTrigger.click();
      }
      if (!activePanel) return;
      const panel = root.querySelector(`#studio-panel-${activePanel}`);
      const trigger = findTrigger(activePanel);
      if (panel && trigger && !panel.contains(event.target) &&
          !trigger.contains(event.target) &&
          !event.target.closest('#studio-prompt-toolbar')) {
        trigger.click();
      }
    });
    window.addEventListener('resize', refresh);
    window.addEventListener('scroll', refresh, {passive: true});
    refresh();
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, {once: true});
  else start();
})();