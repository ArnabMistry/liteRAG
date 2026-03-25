import React, { useState, useEffect } from 'react';
import { Upload, MessageSquare, Shield, FileText, Loader2, Send, CheckCircle2, ChevronRight } from 'lucide-react';

const API_BASE = "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, uploading, indexing, ready
  const [stage, setStage] = useState(''); // Extracting, Chunking, etc.
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);

  const handleUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    setFile(uploadedFile);
    setStatus('uploading');
    setStage('Starting upload...');

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      // In a real app, we'd use WebSockets for progress, here we emulate stages
      setStage('Extracting text...');
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');
      
      setStatus('ready');
      setStage('PDF Indexed');
      setMessages([{ role: 'assistant', text: `Successfully indexed "${uploadedFile.name}". You can now ask questions about it.` }]);
    } catch (err) {
      console.error(err);
      setStatus('idle');
      setStage('Upload failed');
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
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        text: data.answer, 
        sources: data.sources,
        cached: data.cached 
      }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', text: 'Sorry, I encountered an error processing your request.' }]);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="container animate-fade-in">
      <header style={{ marginBottom: '3rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <div style={{ background: 'var(--accent)', padding: '0.5rem', borderRadius: '0.5rem' }}>
          <Shield size={24} color="white" />
        </div>
        <div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700 }}>liteRAG</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>Private • Efficient • Grounded</p>
        </div>
      </header>

      {status === 'idle' || status === 'uploading' ? (
        <div className="upload-section" style={{ 
          background: 'var(--bg-secondary)', 
          border: '2px dashed var(--border)',
          borderRadius: '1rem',
          padding: '4rem 2rem',
          textAlign: 'center',
          transition: 'border-color 0.2s'
        }}>
          <input 
            type="file" 
            id="pdf-upload" 
            accept=".pdf" 
            onChange={handleUpload} 
            style={{ display: 'none' }} 
          />
          <label htmlFor="pdf-upload" style={{ cursor: 'pointer' }}>
            <div style={{ 
              width: '64px', height: '64px', 
              background: 'var(--bg-tertiary)', 
              borderRadius: '50%', 
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              margin: '0 auto 1.5rem'
            }}>
              {status === 'uploading' ? <Loader2 className="animate-pulse" color="var(--accent)" /> : <Upload color="var(--text-secondary)" />}
            </div>
            <h2 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>
              {status === 'uploading' ? 'Analyzing Document' : 'Upload your PDF'}
            </h2>
            <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
              {status === 'uploading' ? stage : 'Drag and drop or click to browse'}
            </p>
          </label>
        </div>
      ) : (
        <div className="chat-section" style={{ display: 'flex', flexDirection: 'column', height: '70vh' }}>
          <div className="messages" style={{ flex: 1, overflowY: 'auto', marginBottom: '1.5rem', paddingRight: '0.5rem' }}>
            {messages.map((msg, i) => (
              <div key={i} className="animate-fade-in" style={{ 
                marginBottom: '1.5rem',
                display: 'flex',
                flexDirection: 'column',
                alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start'
              }}>
                <div style={{ 
                  background: msg.role === 'user' ? 'var(--accent)' : 'var(--bg-secondary)',
                  padding: '1rem 1.25rem',
                  borderRadius: '1rem',
                  borderBottomRightRadius: msg.role === 'user' ? '0.25rem' : '1rem',
                  borderBottomLeftRadius: msg.role === 'user' ? '1rem' : '0.25rem',
                  maxWidth: '85%',
                  fontSize: '0.925rem',
                  lineHeight: 1.6
                }}>
                  {msg.text}
                </div>
                {msg.sources && msg.sources.length > 0 && (
                  <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    {msg.sources.map((s, si) => (
                      <div key={si} style={{ 
                        fontSize: '0.75rem', 
                        color: 'var(--text-muted)', 
                        background: 'var(--bg-tertiary)', 
                        padding: '0.25rem 0.6rem', 
                        borderRadius: '0.5rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.25rem'
                      }}>
                        <FileText size={12} /> Page {s.page}
                      </div>
                    ))}
                    {msg.cached && <div style={{ fontSize: '0.75rem', color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}><CheckCircle2 size={12} /> Cached</div>}
                  </div>
                )}
              </div>
            ))}
            {isQuerying && (
              <div style={{ display: 'flex', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                <Loader2 size={16} className="animate-spin" /> Thinking...
              </div>
            )}
          </div>

          <form onSubmit={handleQuery} style={{ 
            position: 'relative',
            background: 'var(--bg-secondary)',
            borderRadius: '1rem',
            padding: '0.5rem',
            border: '1px solid var(--border)'
          }}>
            <input 
              type="text" 
              value={query} 
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask anything about the document..."
              style={{
                width: '100%',
                background: 'transparent',
                border: 'none',
                color: 'white',
                padding: '0.75rem 3rem 0.75rem 1rem',
                outline: 'none',
              }}
            />
            <button type="submit" disabled={isQuerying || !query.trim()} style={{
              position: 'absolute',
              right: '0.75rem',
              top: '50%',
              transform: 'translateY(-50%)',
              background: query.trim() ? 'var(--accent)' : 'var(--bg-tertiary)',
              color: 'white',
              padding: '0.4rem',
              borderRadius: '0.5rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Send size={18} />
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

export default App;
