
const baseUrl = location.pathname.endsWith("/") ? location.pathname : location.pathname + "/";
const chatLog = document.getElementById("chatLog");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const articleSelect = document.getElementById("articleSelect");
const deleteBtn = document.getElementById("deleteBtn");
let lang = "en";

const I18N = {
  en: {
    tagline: "a self-growing knowledge base · iris graphrag",
    welcomeHtml: "Ask me anything. If I don’t know the answer yet, I’ll <strong>learn it from Wikipedia</strong> — " +
                 "one article at a time — and weave it into my knowledge graph before replying.",
    placeholder: "Tell me about...",
    ask: "Ask",
    manageLabel: "Manage knowledge:",
    deleteBtn: "Delete article",
    noArticles: "— no articles learned yet —",
    statsHtml: (s) => "knowledge graph &middot; <b>" + (s.documents || 0) + "</b> articles &middot; <b>" +
      (s.chunks || 0) + "</b> chunks &middot; <b>" + (s.entities || 0) + "</b> entities &middot; <b>" +
      (s.relationships || 0) + "</b> relations",
    optionLabel: (doc) => doc.title + "  (" + doc.lang + " · " + doc.chunks + " chunks · " + doc.entities + " entities)",
    confirmDelete: (t) => "Delete “" + t + "” and all of its chunks, entities and relationships?\n\nThis cannot be undone.",
    removed: (t) => "🗑️ Removed “" + t + "” and everything derived from it.",
    deleteError: (m) => "⚠️ Could not delete the article: " + m,
    genericError: "Something went wrong.",
    thisArticle: "this article",
    menuDocs: "Documentation",
    menuManage: "Manage knowledge",
    docsTitle: "How WikiGraph works",
    manageTitle: "Manage knowledge",
    docsHtml:
      "<h3>What is this?</h3>" +
      "<p>WikiGraph is a chatbot backed by a <strong>self-growing knowledge base</strong>. Ask a question — if the answer isn’t in its knowledge graph yet, it learns the matching Wikipedia article on the spot, then answers from what it just learned.</p>" +
      "<h3>How an answer is built</h3>" +
      "<ol>" +
      "<li><strong>Search</strong> — your question is matched against the knowledge graph (vector + graph search inside IRIS).</li>" +
      "<li><strong>Learn</strong> — if nothing relevant is found, the single best Wikipedia page is fetched (no linked pages), split into chunks, embedded, and turned into entities &amp; relationships.</li>" +
      "<li><strong>Answer</strong> — the most relevant chunks and graph facts are handed to the LLM, which replies in your selected language.</li>" +
      "</ol>" +
      "<h3>Under the hood</h3>" +
      "<ul>" +
      "<li><strong>IRIS</strong> stores everything — documents, chunks, vectors and the graph.</li>" +
      "<li>Embeddings use the IRIS <code>EMBEDDING</code> type, computed by an <strong>Ollama</strong> model.</li>" +
      "<li>The whole pipeline runs as a traceable <strong>interoperability production</strong> (pyprod).</li>" +
      "<li>The LLM (<strong>gemma</strong> via Ollama) writes the final answer.</li>" +
      "</ul>" +
      "<h3>Tips</h3>" +
      "<ul>" +
      "<li>Toggle <strong>EN / PT</strong> to ask and get answers in English or Portuguese.</li>" +
      "<li>Use <strong>Manage knowledge</strong> to review or delete learned articles (and everything derived from them).</li>" +
      "</ul>",
    status: { pending: "Thinking", searching: "Searching my knowledge graph",
              learning: "Learning from Wikipedia", answering: "Writing the answer" }
  },
  pt: {
    tagline: "a self-growing knowledge base · iris graphrag",
    welcomeHtml: "Pergunte o que quiser. Se eu ainda não souber a resposta, vou <strong>aprender na Wikipédia</strong> — " +
                 "um artigo de cada vez — e integrá-la ao meu grafo de conhecimento antes de responder.",
    placeholder: "Me fale sobre...",
    ask: "Perguntar",
    manageLabel: "Gerenciar conhecimento:",
    deleteBtn: "Excluir artigo",
    noArticles: "— nenhum artigo aprendido ainda —",
    statsHtml: (s) => "grafo de conhecimento &middot; <b>" + (s.documents || 0) + "</b> artigos &middot; <b>" +
      (s.chunks || 0) + "</b> trechos &middot; <b>" + (s.entities || 0) + "</b> entidades &middot; <b>" +
      (s.relationships || 0) + "</b> relações",
    optionLabel: (doc) => doc.title + "  (" + doc.lang + " · " + doc.chunks + " trechos · " + doc.entities + " entidades)",
    confirmDelete: (t) => "Excluir “" + t + "” e todos os seus trechos, entidades e relações?\n\nIsso não pode ser desfeito.",
    removed: (t) => "🗑️ Removido “" + t + "” e tudo que derivou dele.",
    deleteError: (m) => "⚠️ Não foi possível excluir o artigo: " + m,
    genericError: "Algo deu errado.",
    thisArticle: "este artigo",
    menuDocs: "Documentação",
    menuManage: "Gerenciar conhecimento",
    docsTitle: "Como o WikiGraph funciona",
    manageTitle: "Gerenciar conhecimento",
    docsHtml:
      "<h3>O que é isto?</h3>" +
      "<p>O WikiGraph é um chatbot apoiado por uma <strong>base de conhecimento que cresce sozinha</strong>. Faça uma pergunta — se a resposta ainda não estiver no grafo de conhecimento, ele aprende o artigo correspondente da Wikipédia na hora e responde com o que acabou de aprender.</p>" +
      "<h3>Como uma resposta é construída</h3>" +
      "<ol>" +
      "<li><strong>Buscar</strong> — sua pergunta é comparada com o grafo de conhecimento (busca vetorial + grafo dentro do IRIS).</li>" +
      "<li><strong>Aprender</strong> — se nada relevante for encontrado, a melhor página da Wikipédia é baixada (sem seguir links), dividida em trechos, vetorizada e transformada em entidades e relações.</li>" +
      "<li><strong>Responder</strong> — os trechos e fatos do grafo mais relevantes são enviados ao LLM, que responde no idioma selecionado.</li>" +
      "</ol>" +
      "<h3>Nos bastidores</h3>" +
      "<ul>" +
      "<li>O <strong>IRIS</strong> guarda tudo — documentos, trechos, vetores e o grafo.</li>" +
      "<li>Os embeddings usam o tipo <code>EMBEDDING</code> do IRIS, calculado por um modelo no <strong>Ollama</strong>.</li>" +
      "<li>Todo o fluxo roda como uma <strong>produção de interoperabilidade</strong> rastreável (pyprod).</li>" +
      "<li>O LLM (<strong>gemma</strong> via Ollama) escreve a resposta final.</li>" +
      "</ul>" +
      "<h3>Dicas</h3>" +
      "<ul>" +
      "<li>Alterne <strong>EN / PT</strong> para perguntar e receber respostas em inglês ou português.</li>" +
      "<li>Use <strong>Gerenciar conhecimento</strong> para revisar ou excluir artigos aprendidos (e tudo derivado deles).</li>" +
      "</ul>",
    status: { pending: "Pensando", searching: "Pesquisando no meu grafo de conhecimento",
              learning: "Aprendendo com a Wikipédia", answering: "Escrevendo a resposta" }
  }
};
const STATUS_ICON = { pending: "✨", searching: "🔍", learning: "📚", answering: "✍️" };
const t = () => I18N[lang];

function setLang(newLang) {
  lang = newLang;
  document.documentElement.lang = lang;
  document.getElementById("langEn").classList.toggle("active", lang === "en");
  document.getElementById("langPt").classList.toggle("active", lang === "pt");
  const dict = t();
  questionInput.placeholder = dict.placeholder;
  document.getElementById("tagline").textContent = dict.tagline;
  document.getElementById("manageLabel").textContent = dict.manageLabel;
  deleteBtn.textContent = dict.deleteBtn;
  sendBtn.textContent = dict.ask;
  document.getElementById("menuDocs").textContent = dict.menuDocs;
  document.getElementById("menuManage").textContent = dict.menuManage;
  document.getElementById("panelDocs").innerHTML = dict.docsHtml;
  // keep the open side-panel's title in sync with the language
  if (sidePanel.classList.contains("open")) {
    document.getElementById("sidePanelTitle").textContent =
      sidePanel.dataset.panel === "manage" ? dict.manageTitle : dict.docsTitle;
  }
  const welcome = document.getElementById("welcome");
  if (welcome) welcome.innerHTML = dict.welcomeHtml;
  // re-render the server-fed sections in the new language
  refreshStats();
  refreshDocuments();
}

const sidePanel = document.getElementById("sidePanel");
const menuBtn = document.getElementById("menuBtn");
const menuDropdown = document.getElementById("menuDropdown");

function toggleMenu(event) {
  if (event) event.stopPropagation();
  const open = menuDropdown.classList.toggle("open");
  menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
}

function closeMenu() {
  menuDropdown.classList.remove("open");
  menuBtn.setAttribute("aria-expanded", "false");
}

function openPanel(which) {
  closeMenu();
  sidePanel.dataset.panel = which;
  document.getElementById("panelDocs").hidden = which !== "docs";
  document.getElementById("panelManage").hidden = which !== "manage";
  document.getElementById("sidePanelTitle").textContent =
    which === "manage" ? t().manageTitle : t().docsTitle;
  sidePanel.classList.add("open");
  sidePanel.setAttribute("aria-hidden", "false");
  if (which === "manage") { refreshStats(); refreshDocuments(); }
}

function closePanel() {
  sidePanel.classList.remove("open");
  sidePanel.setAttribute("aria-hidden", "true");
}

// dismiss the dropdown / panel when clicking elsewhere or pressing Escape
document.addEventListener("click", (event) => {
  if (!menuDropdown.contains(event.target) && event.target !== menuBtn) closeMenu();
  if (sidePanel.classList.contains("open") &&
      !sidePanel.contains(event.target) && !menuDropdown.contains(event.target) && event.target !== menuBtn) {
    closePanel();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") { closeMenu(); closePanel(); }
});

function appendMessage(role, text, extraClass) {
  const wrapper = document.createElement("div");
  wrapper.className = "msg " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble" + (extraClass ? " " + extraClass : "");
  bubble.textContent = text;
  wrapper.appendChild(bubble);
  chatLog.appendChild(wrapper);
  wrapper.scrollIntoView({ behavior: "smooth", block: "end" });
  return bubble;
}

function appendStatusBubble() {
  const bubble = appendMessage("bot", "", "statusBubble");
  bubble.innerHTML = '<span class="icon"></span><span><span class="detail"></span><span class="dots"></span></span>';
  return bubble;
}

function updateStatusBubble(bubble, status, statusDetail) {
  bubble.classList.toggle("learning", status === "learning");
  bubble.querySelector(".icon").textContent = STATUS_ICON[status] || STATUS_ICON.pending;
  const localized = t().status[status] || t().status.pending;
  // the server's statusDetail is English-only, so only show it in EN; PT uses the localized text
  bubble.querySelector(".detail").textContent = (lang === "en" && statusDetail) ? statusDetail : localized;
}

async function refreshStats() {
  try {
    const stats = await (await fetch(baseUrl + "api/stats")).json();
    document.getElementById("kbStats").innerHTML = t().statsHtml(stats);
  } catch (e) { /* stats are decorative */ }
}

async function refreshDocuments() {
  try {
    const data = await (await fetch(baseUrl + "api/documents")).json();
    const docs = data.documents || [];
    articleSelect.innerHTML = "";
    if (docs.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = t().noArticles;
      articleSelect.appendChild(opt);
      articleSelect.disabled = true;
      deleteBtn.disabled = true;
      return;
    }
    articleSelect.disabled = false;
    deleteBtn.disabled = false;
    docs.forEach((doc) => {
      const opt = document.createElement("option");
      opt.value = doc.id;
      opt.dataset.title = doc.title;
      opt.textContent = t().optionLabel(doc);
      articleSelect.appendChild(opt);
    });
  } catch (e) { /* leave the dropdown as-is on a transient error */ }
}

deleteBtn.addEventListener("click", async () => {
  const id = articleSelect.value;
  if (!id) return;
  const title = articleSelect.options[articleSelect.selectedIndex].dataset.title || t().thisArticle;
  if (!confirm(t().confirmDelete(title))) return;
  deleteBtn.disabled = true;
  articleSelect.disabled = true;
  try {
    const response = await fetch(baseUrl + "api/documents/" + id, { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    const welcome = document.getElementById("welcome");
    if (welcome) welcome.remove();
    appendMessage("bot", t().removed(data.title));
  } catch (e) {
    appendMessage("bot", t().deleteError(e.message), "error");
  }
  await refreshDocuments();
  await refreshStats();
});

function pollAnswer(requestId, statusBubble) {
  const timer = setInterval(async () => {
    try {
      const response = await fetch(baseUrl + "api/chat/" + requestId);
      if (!response.ok) return;
      const data = await response.json();
      if (data.status === "done") {
        clearInterval(timer);
        statusBubble.closest(".msg").remove();
        appendMessage("bot", data.answer);
        sendBtn.disabled = false;
        refreshStats();
        refreshDocuments();
      } else if (data.status === "error") {
        clearInterval(timer);
        statusBubble.closest(".msg").remove();
        appendMessage("bot", "⚠️ " + (data.statusDetail || t().genericError), "error");
        sendBtn.disabled = false;
      } else {
        updateStatusBubble(statusBubble, data.status, data.statusDetail);
      }
    } catch (e) { /* transient network error - keep polling */ }
  }, 1500);
}

document.getElementById("composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || sendBtn.disabled) return;
  const welcome = document.getElementById("welcome");
  if (welcome) welcome.remove();

  appendMessage("user", question);
  questionInput.value = "";
  sendBtn.disabled = true;
  const statusBubble = appendStatusBubble();
  updateStatusBubble(statusBubble, "pending", "");

  try {
    const response = await fetch(baseUrl + "api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, lang: lang })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    pollAnswer(data.requestId, statusBubble);
  } catch (e) {
    statusBubble.closest(".msg").remove();
    appendMessage("bot", "⚠️ " + e.message, "error");
    sendBtn.disabled = false;
  }
});

// populate the documentation panel for the initial language
document.getElementById("panelDocs").innerHTML = t().docsHtml;
refreshStats();
refreshDocuments();
