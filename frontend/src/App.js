import React, { useState, useEffect, useRef, useCallback } from "react";
import "./App.css";

const API = process.env.REACT_APP_API_URL || "http://localhost:8501";
const STORAGE_KEY = "ml-assistant-conversations";

const EXAMPLES = [
  { text: "What is a transformer architecture?", tag: "Architecture" },
  { text: "Explain the difference between LSTM and GRU", tag: "Comparison" },
  { text: "How does backpropagation work?", tag: "Training" },
  { text: "What is transfer learning?", tag: "Technique" },
  { text: "Explain attention mechanisms in neural networks", tag: "Architecture" },
  { text: "What are the differences between CNN and RNN?", tag: "Comparison" },
  { text: "How does gradient descent optimize a model?", tag: "Training" },
  { text: "What is the vanishing gradient problem?", tag: "Concepts" },
];

function loadConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveConversations(convos) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(convos));
}

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function getTitle(messages) {
  const first = messages.find(m => m.role === "user");
  if (!first) return "New Chat";
  return first.text.length > 34 ? first.text.slice(0, 34) + "..." : first.text;
}

function timeAgo(dateStr) {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

function Logo({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none">
      <path d="M16 3 Q22 10 16 16 Q10 10 16 3Z" fill="currentColor" />
      <path d="M29 16 Q22 22 16 16 Q22 10 29 16Z" fill="currentColor" opacity="0.75" />
      <path d="M16 29 Q10 22 16 16 Q22 22 16 29Z" fill="currentColor" opacity="0.5" />
      <path d="M3 16 Q10 10 16 16 Q10 22 3 16Z" fill="currentColor" opacity="0.35" />
    </svg>
  );
}

function useSimulatedProgress(active) {
  const [progress, setProgress] = useState(0);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (active) {
      setProgress(0);
      let p = 0;
      intervalRef.current = setInterval(() => {
        p += (95 - p) * 0.04;
        setProgress(Math.min(Math.round(p), 95));
      }, 300);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (progress > 0) {
        setProgress(100);
        const t = setTimeout(() => setProgress(0), 400);
        return () => clearTimeout(t);
      }
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [active]);

  return progress;
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <path d="M12 4L12 20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M5 11L12 4L19 11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function StreamingText({ text, onDone }) {
  const [displayed, setDisplayed] = useState("");
  const indexRef = useRef(0);

  useEffect(() => {
    if (!text) { setDisplayed(""); return; }
    indexRef.current = 0;
    setDisplayed("");
    const words = text.split(" ");
    let current = 0;

    const interval = setInterval(() => {
      current++;
      if (current <= words.length) {
        setDisplayed(words.slice(0, current).join(" "));
      } else {
        clearInterval(interval);
        if (onDone) onDone();
      }
    }, 30);

    return () => clearInterval(interval);
  }, [text, onDone]);

  return <>{displayed}<span className="cursor" /></>;
}

function App() {
  const [conversations, setConversations] = useState(() => loadConversations());
  const [activeId, setActiveId] = useState(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [streamingIdx, setStreamingIdx] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [showPanel, setShowPanel] = useState(false);
  const [showExperiments, setShowExperiments] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const chatEnd = useRef(null);
  const textareaRef = useRef(null);
  const progress = useSimulatedProgress(loading);

  const activeConvo = conversations.find(c => c.id === activeId);
  const messages = activeConvo ? activeConvo.messages : [];

  const updateConversations = useCallback((updater) => {
    setConversations(prev => {
      const next = typeof updater === "function" ? updater(prev) : updater;
      saveConversations(next);
      return next;
    });
  }, []);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, streamingIdx]);

  useEffect(() => {
    fetch(`${API}/model-info`).then(r => r.json()).then(setModelInfo).catch(() => {});
    fetch(`${API}/metrics`).then(r => r.json()).then(setMetrics).catch(() => {});
  }, []);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 160) + "px";
    }
  }, [input]);

  const startNewChat = () => {
    setActiveId(null);
    setInput("");
    setStreamingIdx(null);
  };

  const deleteConversation = (e, id) => {
    e.stopPropagation();
    updateConversations(prev => prev.filter(c => c.id !== id));
    if (activeId === id) {
      setActiveId(null);
      setStreamingIdx(null);
    }
  };

  const sendMessage = async (text) => {
    const msg = text || input.trim();
    if (!msg || loading) return;
    setInput("");
    setStreamingIdx(null);

    let convoId = activeId;
    const userMsg = { role: "user", text: msg };

    if (!convoId) {
      convoId = generateId();
      const newConvo = { id: convoId, messages: [userMsg], createdAt: new Date().toISOString() };
      updateConversations(prev => [newConvo, ...prev]);
      setActiveId(convoId);
    } else {
      updateConversations(prev =>
        prev.map(c => c.id === convoId ? { ...c, messages: [...c.messages, userMsg] } : c)
      );
    }

    setLoading(true);
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      });
      const data = await res.json();
      const assistantMsg = { role: "assistant", text: data.response };
      updateConversations(prev => {
        const updated = prev.map(c => c.id === convoId ? { ...c, messages: [...c.messages, assistantMsg] } : c);
        const convo = updated.find(c => c.id === convoId);
        if (convo) setStreamingIdx(convo.messages.length - 1);
        return updated;
      });
    } catch {
      const errorMsg = { role: "assistant", text: "Could not reach the API server. Please make sure the backend is running." };
      updateConversations(prev =>
        prev.map(c => c.id === convoId ? { ...c, messages: [...c.messages, errorMsg] } : c)
      );
    }
    setLoading(false);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const msgCount = (convo) => {
    const n = convo.messages.filter(m => m.role === "user").length;
    return n === 1 ? "1 message" : `${n} messages`;
  };

  return (
    <div className="app">
      <div className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-top">
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M3 6h18M3 12h18M3 18h18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>
          {sidebarOpen && (
            <button className="new-chat-btn" onClick={startNewChat}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              New Chat
            </button>
          )}
        </div>

        {sidebarOpen && (
          <div className="sidebar-conversations">
            {conversations.length === 0 && (
              <div className="sidebar-empty">
                <div className="sidebar-empty-icon"><Logo size={24} /></div>
                No conversations yet
              </div>
            )}
            {conversations.map(convo => (
              <div
                key={convo.id}
                className={`convo-item ${convo.id === activeId ? "active" : ""}`}
                onClick={() => { setActiveId(convo.id); setStreamingIdx(null); }}
              >
                <div className="convo-content">
                  <span className="convo-title">{getTitle(convo.messages)}</span>
                  <span className="convo-meta">{msgCount(convo)} · {timeAgo(convo.createdAt)}</span>
                </div>
                <button
                  className="convo-delete"
                  onClick={(e) => deleteConversation(e, convo.id)}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                    <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        )}

        {sidebarOpen && (
          <div className="sidebar-bottom">
            <div className="sidebar-badge">
              <Logo size={14} />
              <span>QLoRA Fine-tuned</span>
            </div>
          </div>
        )}
      </div>

      <div className="main">
        <header className="top-bar">
          <div className="top-bar-left">
            <div className="top-bar-logo"><Logo size={22} /></div>
            <span className="top-bar-title">ML/AI Learning Assistant</span>
            <span className="top-bar-tag">Beta</span>
          </div>
          <div className="top-bar-right">
            <button className="header-btn" onClick={() => setShowExperiments(!showExperiments)}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
                <path d="M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
              </svg>
              Experiments
            </button>
            <button className="header-btn" onClick={() => setShowPanel(!showPanel)}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2" />
                <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              {showPanel ? "Hide Info" : "Model Info"}
            </button>
          </div>
        </header>

        <div className="chat-area">
          {messages.length === 0 && !activeId && (
            <div className="empty-state">
              <div className="empty-logo"><Logo size={44} /></div>
              <h1>What would you like to learn?</h1>
              <p>Explore machine learning concepts, neural network architectures, training techniques, and modern AI systems.</p>
              <div className="examples-grid">
                {EXAMPLES.map((q, i) => (
                  <button key={i} className="example-btn" style={{ animationDelay: `${i * 0.05}s` }} onClick={() => sendMessage(q.text)}>
                    <span className="example-tag">{q.tag}</span>
                    <span className="example-text">{q.text}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`message-row ${msg.role} fade-in`}>
              {msg.role === "assistant" && (
                <div className="message-avatar">
                  <Logo size={16} />
                </div>
              )}
              <div className={`message-bubble ${msg.role}`}>
                {msg.role === "assistant" && i === streamingIdx && msg.text ? (
                  <StreamingText text={msg.text} onDone={() => setStreamingIdx(null)} />
                ) : (msg.text || "")}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message-row assistant fade-in">
              <div className="loading-icon">
                <Logo size={24} />
              </div>
              <div className="message-bubble assistant">
                <div className="thinking-indicator">
                  <span className="thinking-text">Generating response · {progress}%</span>
                  <div className="thinking-bar">
                    <div className="thinking-progress" style={{ width: `${progress}%` }}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEnd} />
        </div>

        <div className="input-wrapper">
          <div className="input-container">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Ask about machine learning..."
              rows={1}
              disabled={loading}
            />
            <button
              className="send-btn"
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
            >
              <SendIcon />
            </button>
          </div>
          <div className="input-footer">
            Fine-tuned TinyLlama-1.1B with QLoRA · Responses may not always be accurate
          </div>
        </div>
      </div>

      {showPanel && (
        <div className="info-panel slide-in-right">
          <div className="panel-header">
            <h3>Model Information</h3>
            <button className="panel-close" onClick={() => setShowPanel(false)}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </button>
          </div>

          <div className="panel-card">
            <div className="panel-card-header">
              <Logo size={18} />
              <span>TinyLlama-1.1B</span>
            </div>
            {modelInfo ? (
              <div className="info-grid">
                <div className="info-item"><span>Quantization</span><span>{modelInfo.quantization}</span></div>
                <div className="info-item"><span>Total Parameters</span><span>{modelInfo.total_params}</span></div>
                <div className="info-item"><span>Trainable Parameters</span><span>{modelInfo.trainable_params}</span></div>
                <div className="info-item"><span>VRAM Used</span><span>{modelInfo.vram_used}</span></div>
              </div>
            ) : <p className="panel-note">Connecting to server...</p>}
          </div>

          <h3 className="panel-section">Evaluation Metrics</h3>
          {metrics?.comparison ? (
            <table className="metrics-table">
              <thead>
                <tr><th>Metric</th><th>Base</th><th>Fine-tuned</th></tr>
              </thead>
              <tbody>
                {metrics.comparison.map((row, i) => (
                  <tr key={i}>
                    <td>{row.metric}</td>
                    <td>{row.base_model}</td>
                    <td className="metric-highlight">{row.fine_tuned}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <p className="panel-note">No metrics available yet</p>}

          {metrics?.perplexity && (
            <>
              <h3 className="panel-section">Perplexity</h3>
              <div className="perplexity-cards">
                <div className="perp-card">
                  <span className="perp-label">Base</span>
                  <span className="perp-value">{metrics.perplexity.base}</span>
                </div>
                <div className="perp-card improved">
                  <span className="perp-label">Fine-tuned</span>
                  <span className="perp-value">{metrics.perplexity.fine_tuned}</span>
                </div>
              </div>
            </>
          )}

        </div>
      )}

      {showExperiments && (
        <div className="modal-overlay" onClick={() => setShowExperiments(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Training Experiments</h2>
              <button className="panel-close" onClick={() => setShowExperiments(false)}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                </svg>
              </button>
            </div>
            {metrics?.experiments ? (
              <table className="metrics-table experiments-table">
                <thead>
                  <tr>
                    <th>Experiment</th>
                    <th>Learning Rate</th>
                    <th>LoRA Rank</th>
                    <th>LoRA Alpha</th>
                    <th>Batch Size</th>
                    <th>Epochs</th>
                    <th>Train Loss</th>
                    <th>Eval Loss</th>
                    <th>Peak VRAM</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.experiments.map((run, i) => {
                    const best = run.best_eval_loss === Math.min(...metrics.experiments.map(r => r.best_eval_loss));
                    return (
                      <tr key={i} style={best ? {background: "#f8faf3"} : {}}>
                        <td style={best ? {fontWeight: 600, color: "#5a6e3a"} : {}}>
                          {run.experiment.replace(/_/g, " ")}
                          {best && " *"}
                        </td>
                        <td>{run.learning_rate}</td>
                        <td>{run.lora_r}</td>
                        <td>{run.lora_alpha}</td>
                        <td>{run.batch_size}</td>
                        <td>{run.epochs}</td>
                        <td>{run.train_loss}</td>
                        <td style={best ? {fontWeight: 600, color: "#5a6e3a"} : {}}>{run.best_eval_loss}</td>
                        <td>{run.peak_vram_gb} GB</td>
                        <td>{Math.round(run.training_time_min)} min</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            ) : <p className="panel-note">Loading experiments...</p>}
            <p className="modal-footnote">* Best experiment (lowest eval loss) — deployed to production</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
