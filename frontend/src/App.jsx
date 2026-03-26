import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Upload, FileText, Loader2, Send, ChevronRight, ChevronDown, CheckCircle2, AlertCircle, BookOpen, Clock, Zap, FileSearch } from 'lucide-react';

const API_BASE = "http://localhost:8000";

// --- Helpers ---
const getConfidenceColor = (score) => {
  if (score > 0.65) return 'var(--success)';
  if (score > 0.45) return 'var(--warning)';
  return 'var(--error)';
};

const getConfidenceText = (score) => {
  if (score > 0.65) return 'High Match';
  if (score > 0.45) return 'Partial Match';
  return 'Marginal Match';
};

const highlightRelevantSentences = (text, query) => {
  if (!text || !query) return text;
  
  const queryWords = query.toLowerCase().split(/\s+/).filter(w => w.length > 3);
  if (queryWords.length === 0) return text;

  // Simple contextual highlighter: highlight sentences containing strong query matches
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  
  return sentences.map((sentence, idx) => {
    const sLower = sentence.toLowerCase();
    const hasMatch = queryWords.some(w => sLower.includes(w));
    
    if (hasMatch) {
      // Bold the specific matching words within the sentence
      let highlightedSentence = sentence;
      queryWords.forEach(word => {
        const regex = new RegExp(`(${word})`, 'gi');
        highlightedSentence = highlightedSentence.replace(regex, '<span style="color: var(--accent); font-weight: 500;">$1</span>');
      });
      return `<mark style="background: var(--accent-softer); color: var(--text-primary); border-radius: 2px;">${highlightedSentence}</mark>`;
    }
    return `<span style="opacity: 0.7">${sentence}</span>`;
  }).join(' ');
};

// --- Components ---

const SourceHighlight = ({ sources, cached, query }) => {
  if (!sources || sources.length === 0) return null;

  const [expanded, setExpanded] = useState(false);
  const uniquePages = new Set(sources.map(s => s.page));

  return (
    <div className="source-highlight animate-fade-in" style={{ marginTop: '2rem', borderTop: '1px solid var(--border)', paddingTop: '1.5rem' }}>
      <div 
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', userSelect: 'none' }}
        onClick={() => setExpanded(!expanded)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <h3 style={{ fontSize: '0.875rem', fontFamily: 'var(--font-mono)', letterSpacing: '0.05em', textTransform: 'uppercase', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <FileSearch size={14} /> Reasoning Context
          </h3>
          <span style={{ borderLeft: '1px solid var(--border)', paddingLeft: '1rem', display: 'flex', gap: '1rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              Synthesized from <strong style={{color: 'var(--text-primary)'}}>{sources.length} chunks</strong> across <strong style={{color: 'var(--text-primary)'}}>{uniquePages.size} pages</strong>
            </span>
          </span>
        </div>
        <div>
          {expanded ? <ChevronDown size={18} color="var(--text-muted)" /> : <ChevronRight size={18} color="var(--text-muted)" />}
        </div>
      </div>

      {expanded && (
        <div style={{ marginTop: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {sources.map((source, idx) => {
            const isRank1 = source.rank === 1 || idx === 0;
            const confColor = getConfidenceColor(source.score);
            const confText = getConfidenceText(source.score);
            
            return (
              <div id={`source-${source.rank || idx + 1}`} key={idx} style={{ 
                background: isRank1 ? 'var(--bg-tertiary)' : 'var(--bg-secondary)', 
                borderLeft: `${isRank1 ? '4px' : '2px'} solid ${isRank1 ? 'var(--accent)' : 'var(--border)'}`,
                padding: '1.25rem',
                borderRadius: '0 var(--radius-sm) var(--radius-sm) 0',
                opacity: isRank1 ? 1 : 0.85,
                transition: 'opacity 0.2s ease',
              }}
              onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
              onMouseLeave={(e) => e.currentTarget.style.opacity = isRank1 ? '1' : '0.85'}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <span style={{ 
                      fontFamily: 'var(--font-mono)', 
                      fontSize: '0.75rem', 
                      color: isRank1 ? 'var(--accent)' : 'var(--text-secondary)', 
                      fontWeight: isRank1 ? 700 : 500,
                      background: isRank1 ? 'var(--accent-soft)' : 'transparent',
                      padding: isRank1 ? '0.1rem 0.5rem' : '0',
                      borderRadius: '2px'
                    }}>
                      [RANK {source.rank || (idx + 1).toString().padStart(2, '0')}]
                    </span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      PAGE {source.page}
                    </span>
                    {source.score !== undefined && (
                      <span style={{ 
                        fontFamily: 'var(--font-mono)', 
                        fontSize: '0.7rem', 
                        color: confColor,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.25rem'
                      }}>
                        <Zap size={10} /> {confText} {(source.score).toFixed(2)}
                      </span>
                    )}
                  </div>
                </div>
                
                <details open={isRank1} style={{ fontFamily: 'var(--font-body)', fontSize: '0.875rem', color: 'var(--text-primary)', lineHeight: 1.6 }}>
                  <summary style={{ cursor: 'pointer', outline: 'none', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                    <span style={{ fontStyle: 'italic' }}>Relevant excerpt</span>
                  </summary>
                  <div 
                    style={{ padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', marginTop: '0.5rem', whiteSpace: 'pre-wrap', border: '1px solid var(--border)' }}
                    dangerouslySetInnerHTML={{ __html: highlightRelevantSentences(source.text, query) || "Backend payload missing text extract." }}
                  />
                </details>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

const TranscriptItem = ({ message }) => {
  if (message.role === 'user') {
    return (
      <div className="animate-slide-up" style={{ marginBottom: '3rem', marginTop: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', color: 'var(--accent)' }}>
          <div style={{ width: '12px', height: '2px', background: 'var(--accent)' }}></div>
          <span className="mono-text" style={{ textTransform: 'uppercase', fontSize: '0.7rem', letterSpacing: '0.1em' }}>Query Focus</span>
        </div>
        <h2 style={{ fontSize: '2rem', color: 'var(--text-primary)', marginLeft: '1rem', lineHeight: 1.3, paddingLeft: '1rem', borderLeft: '1px solid var(--border)' }}>
          {message.text}
        </h2>
      </div>
    );
  }

  // Parse text for citations [1], [2] and turn them into links
  const renderTextWithCitations = (text) => {
    // Splits by [1], [2], etc.
    const parts = text.split(/(\[\d+\])/g);
    return parts.map((part, i) => {
      const match = part.match(/\[(\d+)\]/);
      if (match) {
        const id = match[1];
        return (
          <a key={i} href={`#source-${id}`} style={{
            color: 'var(--accent)',
            textDecoration: 'none',
            fontSize: '0.75em',
            verticalAlign: 'super',
            cursor: 'pointer',
            padding: '0 0.1em'
          }} title={`Jump to Source Rank ${id}`}>
            {part}
          </a>
        );
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <div className="animate-slide-up" style={{ marginBottom: '4rem', marginLeft: '2rem' }}>
      
      {message.isFailure ? (
        <div style={{ borderLeft: '3px solid var(--error)', padding: '1.5rem', background: 'var(--bg-tertiary)', borderRadius: '0 var(--radius-sm) var(--radius-sm) 0' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--error)', marginBottom: '0.5rem', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
            <AlertCircle size={14} /> NO CONTEXTUAL MATCH
          </div>
          <p style={{ color: 'var(--text-primary)', lineHeight: 1.6 }}>{message.text}</p>
        </div>
      ) : (
        <div className="answer-content reading-measure">
          {message.text.split('\n').filter(p => p.trim() !== '').map((paragraph, idx) => (
            <p key={idx}>{renderTextWithCitations(paragraph)}</p>
          ))}
        </div>
      )}

      {message.sources && message.sources.length > 0 && (
        <SourceHighlight sources={message.sources} cached={message.cached} query={message.originalQuery} />
      )}
    </div>
  );
};

// --- Main App ---

function App() {
  const [session, setSession] = useState({ id: null, documentName: null });
  const [status, setStatus] = useState('idle'); // idle, uploading, ready
  const [uploadStages, setUploadStages] = useState([]);
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  
  const bottomRef = useRef(null);

  useEffect(() => {
    if (messages.length > 0) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isQuerying]);

  const addUploadStage = (msg, isCurrent = false, isDone = false) => {
    setUploadStages(prev => [...prev.filter(s => s.msg !== msg), { msg, isCurrent, isDone }]);
  };

  const handleUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    setStatus('uploading');
    setUploadStages([]);
    
    // UI Simulation for UX Feedback
    addUploadStage('Initializing upload...', true, false);

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      setTimeout(() => addUploadStage('Initializing upload...', false, true), 300);
      setTimeout(() => addUploadStage('Extracting text context...', true, false), 400);
      
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');
      const data = await response.json();
      
      addUploadStage('Extracting text context...', false, true);
      addUploadStage(`Chunking & Embedding (${data.pages || '?'} pages)...`, false, true);
      addUploadStage('Indexing vectors...', false, true);
      
      setTimeout(() => {
        setSession({ id: data.file_id || Date.now(), documentName: uploadedFile.name });
        setStatus('ready');
      }, 800);
      
    } catch (err) {
      console.error(err);
      setStatus('idle');
      alert('Failed to upload document. Please ensure backend is running.');
    }
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim() || isQuerying) return;

    const userQuery = query.trim();
    setQuery('');
    setMessages(prev => [...prev, { role: 'user', text: userQuery }]);
    setIsQuerying(true);

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userQuery }),
      });

      const data = await response.json();
      
      // Handle Failure State - No context / ambiguous query
      if (data.answer.toLowerCase().includes("i don't know") || data.answer.toLowerCase().includes("does not contain")) {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          text: "The document does not contain definitive information regarding this query. Ensure the topic is covered in the provided text, or try using alternate phrasing.", 
          sources: [],
          cached: data.cached,
          isFailure: true,
          originalQuery: userQuery
        }]);
      } else {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          text: data.answer, 
          sources: data.sources,
          cached: data.cached,
          originalQuery: userQuery
        }]);
      }
    } catch (err) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        text: 'The system encountered an error connecting to the retrieval pipeline. Please ensure the backend service is operational.',
        isFailure: true,
        originalQuery: userQuery
      }]);
    } finally {
      setIsQuerying(false);
    }
  };

  const placeholders = useMemo(() => [
    "Query the text dynamics...",
    "Extract specific definitions...",
    "What core methodology is proposed?",
    "Summarize the conclusion..."
  ], []);
  
  const [phIndex, setPhIndex] = useState(0);

  useEffect(() => {
    if (status === 'ready' && !query) {
      const interval = setInterval(() => setPhIndex(p => (p + 1) % placeholders.length), 3500);
      return () => clearInterval(interval);
    }
  }, [status, query, placeholders]);

  return (
    <div className="container" style={{ position: 'relative', paddingBottom: '8rem' }}>
      
      {/* Editorial Header */}
      <header className="animate-fade-in" style={{ marginBottom: '4rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 style={{ marginBottom: '0.25rem' }}>liteRAG.</h1>
          <p className="mono-text" style={{ color: 'var(--text-muted)' }}>RESEARCH ENGINE PIPELINE</p>
        </div>
        
        {/* Session Context */}
        {session.documentName && (
          <div style={{ textAlign: 'right', display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)' }}>
              <BookOpen size={14} />
              <span className="mono-text">{session.documentName}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)' }}>
              <Clock size={12} />
              <span className="mono-text" style={{ fontSize: '0.7rem' }}>Session Active</span>
            </div>
          </div>
        )}
      </header>

      {/* Upload State */}
      {(status === 'idle' || status === 'uploading') && (
        <div className={`file-dropzone animate-fade-in ${status === 'uploading' ? 'pulse-border' : ''}`} style={{ padding: '4rem 3rem', borderRadius: 'var(--radius-sm)' }}>
          <input type="file" id="pdf-upload" accept=".pdf" onChange={handleUpload} style={{ display: 'none' }} disabled={status === 'uploading'} />
          
          <label htmlFor="pdf-upload" style={{ cursor: status === 'uploading' ? 'default' : 'pointer', display: 'block' }}>
            {status === 'idle' ? (
              <div style={{ textAlign: 'center' }}>
                <div style={{ marginBottom: '1.5rem', opacity: 0.8 }}>
                  <FileText size={48} strokeWidth={1} color="var(--text-primary)" />
                </div>
                <h2 style={{ marginBottom: '1rem' }}>Ingest Knowledge Base</h2>
                <p style={{ color: 'var(--text-secondary)', maxWidth: '400px', margin: '0 auto', fontFamily: 'var(--font-body)' }}>
                  Upload a PDF document. The system will extract, chunk, and embed the contents into the local vector space for immediate retrieval.
                </p>
                <div style={{ marginTop: '2rem' }}>
                  <span className="btn-primary">Select Document</span>
                </div>
              </div>
            ) : (
              <div style={{ maxWidth: '400px', margin: '0 auto' }}>
                <h2 style={{ marginBottom: '2rem', textAlign: 'center' }}>Processing...</h2>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {uploadStages.map((stage, i) => (
                    <div key={i} className="animate-fade-in" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                      {stage.isDone ? (
                        <CheckCircle2 size={18} color="var(--success)" />
                      ) : stage.isCurrent ? (
                        <Loader2 size={18} className="animate-spin" color="var(--accent)" />
                      ) : (
                        <div style={{ width: '18px', height: '18px', border: '1px solid var(--border)', borderRadius: '50%' }} />
                      )}
                      <span className="mono-text" style={{ 
                        color: stage.isCurrent ? 'var(--text-primary)' : (stage.isDone ? 'var(--text-secondary)' : 'var(--text-muted)'),
                        opacity: stage.isDone ? 0.7 : 1
                      }}>
                        {stage.msg}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </label>
        </div>
      )}

      {/* Research Transcript State */}
      {status === 'ready' && (
        <div className="transcript-container animate-fade-in">
          {messages.length === 0 ? (
            <div style={{ padding: '4rem 0', opacity: 0.5, maxWidth: '600px' }}>
              <h2 style={{ marginBottom: '1rem' }}>Document Indexed.</h2>
              <p style={{ fontSize: '1.125rem' }}>The vector store is active. You may now query the document semantics below.</p>
            </div>
          ) : (
            <div className="messages" style={{ paddingBottom: '2rem' }}>
              {messages.map((msg, i) => <TranscriptItem key={i} message={msg} />)}
              
              {isQuerying && (
                <div className="animate-fade-in" style={{ marginLeft: '2rem', display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--accent)', marginTop: '2rem' }}>
                  <Loader2 size={16} className="animate-spin" />
                  <span className="mono-text" style={{ fontSize: '0.8rem', letterSpacing: '0.05em' }}>Retrieving Context & Generating Synthesis...</span>
                </div>
              )}
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      )}

      {/* Fixed Query Input bottom */}
      {status === 'ready' && (
        <div style={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          background: 'linear-gradient(to top, var(--bg-primary) 80%, transparent)',
          padding: '2rem',
          display: 'flex',
          justifyContent: 'center'
        }}>
          <div style={{ width: '100%', maxWidth: '800px', margin: '0 auto', position: 'relative' }}>
            <form onSubmit={handleQuery} style={{ position: 'relative' }}>
              <input
                type="text"
                className="editorial-input"
                placeholder={placeholders[phIndex]}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                autoFocus
              />
              <button 
                type="submit" 
                disabled={!query.trim() || isQuerying}
                style={{
                  position: 'absolute',
                  right: 0,
                  bottom: '0.75rem',
                  background: 'none',
                  border: 'none',
                  color: query.trim() ? 'var(--text-primary)' : 'var(--text-muted)',
                  cursor: query.trim() && !isQuerying ? 'pointer' : 'not-allowed',
                  transition: 'color 0.2s ease'
                }}
              >
                <Send size={24} />
              </button>
            </form>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;
