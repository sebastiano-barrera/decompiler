import { h, render } from 'https://esm.sh/preact?dev';
import { useState, useEffect } from 'https://esm.sh/preact/hooks?dev';
import htm from 'https://esm.sh/htm';

// Initialize htm with Preact
const html = htm.bind(h);

class Debouncer {
  constructor(delay) {
    this.delay = delay;
    this.timeoutId = null;
  }

  debounce(func) {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
    }
    this.timeoutId = setTimeout(() => {
      func();
      this.timeoutId = null;
    }, this.delay);
  }
}

function FunctionList({ exe }) {
  useEffect(() => {
    document.title = `${exe.name} - Functions - decompiler`;
  }, [exe.name]);

  const [filterQuery, setFilterQuery_] = useState('');
  const setFilterQuery = value => setFilterQuery_(value.toLowerCase());

  const [countLimit, setCountLimit] = useState(200);

  const filteredElements = []
  let visibleCount = 0
  for (const funcName of exe.functions) {
    const isVisible = visibleCount < countLimit && funcName.toLowerCase().includes(filterQuery);
    if (isVisible) { visibleCount++; }
    filteredElements.push(html`
      <li key="${funcName}" style="display: ${isVisible ? 'block' : 'none'}">
        <a href="/functions/${encodeURIComponent(funcName)}">${funcName}</a>
      </li>
    `);
  }

  return html`
    <div class="function-list">
      <input 
        id="query"
        type="text"
        placeholder="Search function by name..."
        onInput="${(event) => setFilterQuery(event.target.value)}" />
      <div>
        <span>${filteredElements.length} / ${exe.functions.length} functions</span>
      </div>
      <ul>
        ${filteredElements}
        <a
          class="big-button"
          onClick="${e => setCountLimit(Infinity)}"
        >
          More ➫
        </a>
      </ul>
      
    </div>
  `;
}

function MainArea() {
  const [exe, setExe] = useState({
    name: "???",
    functions: [],
  });

  useEffect(async () => {
      const res = await fetch("/exe");
      const data = await res.json();
      setExe(data);
  }, []);

  return html`
    <div class="topbar">
      ${exe.name} 🢒 Functions
    </div>
    <div id="main-area">
      <${FunctionList} exe=${exe} />
    </div>
  `;
}

render(html`<${MainArea} />`, document.getElementById("app"));