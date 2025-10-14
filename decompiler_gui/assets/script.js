import { LocationProvider, Router, useLocation, useRoute } from "preact-iso";
import { h, render } from "preact";
import { useState, useEffect } from "preact/hooks";
import htm from "htm";

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

  const [filterQuery, setFilterQuery_] = useState("");
  const setFilterQuery = (value) => setFilterQuery_(value.toLowerCase());

  const [countLimit, setCountLimit] = useState(200);

  const filteredElements = [];
  let visibleCount = 0;
  for (const funcName of exe.functions) {
    const isVisible =
      visibleCount < countLimit && funcName.toLowerCase().includes(filterQuery);
    if (isVisible) {
      visibleCount++;
    }
    filteredElements.push(html`
      <li key="${funcName}" style="display: ${isVisible ? "block" : "none"}">
        <a href="/p/functions/${encodeURIComponent(funcName)}">${funcName}</a>
      </li>
    `);
  }

  return html`
    <div class="function-list">
      <input
        id="query"
        type="text"
        placeholder="Search function by name..."
        onInput="${(event) => setFilterQuery(event.target.value)}"
      />
      <div>
        <span
          >${filteredElements.length} / ${exe.functions.length} functions</span
        >
      </div>
      <ul>
        ${filteredElements}
        <a class="big-button" onClick="${(e) => setCountLimit(Infinity)}">
          More ➫
        </a>
      </ul>
    </div>
  `;
}

function Page_FunctionList({ exe }) {
  const topBar = html`${exe.name} 🢒 Functions`;
  const mainArea = html`<${FunctionList} exe=${exe} />`;

  return html`<${ToplevelLayout} topBar="${topBar}" mainArea="${mainArea}" />`;
}

function Page_Function({}) {
  const { params } = useRoute();
  const functionName = params.name;

  return html`
    <div>Function here!</div>
    <div>
      function:
      <pre>${functionName}</pre>
    </div>
  `;
}

function Page_NotFound() {
  const { url } = useLocation();
  return html`<div>Not found: ${url}</div>`;
}

function Page_Initial() {
  const { route } = useLocation();
  route("/p/functions/");
}

function ToplevelLayout({ topBar, mainArea }) {
  return html`
    <div class="topbar">${topBar}</div>
    <div id="main-area">${mainArea}</div>
  `;
}

function App() {
  const [exe, setExe] = useState({
    name: "???",
    functions: [],
  });

  useEffect(async () => {
    const res = await fetch("/exe");
    const data = await res.json();
    setExe(data);
  }, []);

  return html`<${LocationProvider}>
    <${Router}>
      <${Page_Initial} path="/p/" />
      <${Page_Function} path="/p/functions/:name" exe=${exe} />
      <${Page_FunctionList} path="/p/functions" exe=${exe} />
      <${Page_NotFound} default />
    <//>
  <//>`;
}

render(html`<${App} />`, document.getElementById("app"));
