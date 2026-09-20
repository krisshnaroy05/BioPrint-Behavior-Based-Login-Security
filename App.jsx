import React, { useState, useEffect } from 'react';
import { useBiometrics } from './hooks/useBiometrics';

export default function App() {
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false); // Visibility toggle state
  const [deviceContext, setDeviceContext] = useState('desktop-mouse');
  const [dashboardData, setDashboardData] = useState(null);
  const [enrollBuffer, setEnrollBuffer] = useState([]);
  const [activeNonce, setActiveNonce] = useState(''); // Pre-fetched nonce state
  
  const { bindInput, bindContainer, getTelemetryPayload, resetTelemetry } = useBiometrics();

  // Pre-fetch the cryptographic challenge to eliminate network latency during login
  const refreshNonce = async () => {
    try {
      const res = await fetch('http://localhost:8000/challenge');
      const data = await res.json();
      setActiveNonce(data.nonce);
    } catch (err) {
      console.error("Failed to fetch nonce:", err);
    }
  };

  // Fetch initial nonce on component mount
  useEffect(() => {
    refreshNonce();
  }, []);

  const handleEnroll = async (e) => {
    e.preventDefault();
    if (!password) return;

    if (enrollBuffer.length > 0 && password !== enrollBuffer[0].password) {
      alert('Password does not match your first attempt! Please try again.');
      resetTelemetry(); 
      setPassword(''); 
      return; 
    }

    const payload = { 
      ...getTelemetryPayload(userId), 
      password, 
      nonce: activeNonce,
      client_timestamp: Date.now(),
      device_context: deviceContext
    };
    
    const newBuffer = [...enrollBuffer, payload];
    setEnrollBuffer(newBuffer);
    resetTelemetry(); 
    setPassword(''); // Cleans the password bar immediately
    refreshNonce();  // Queue up the next nonce in the background

    if (newBuffer.length === 3) {
      await fetch('http://localhost:8000/enroll', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, samples: newBuffer })
      });
      alert(`Profile Baseline Established for ${userId} on ${deviceContext}!`);
      setEnrollBuffer([]);
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    if (!password) return;

    const payload = { 
      ...getTelemetryPayload(userId), 
      password,
      nonce: activeNonce,
      client_timestamp: Date.now(),
      device_context: deviceContext
    };
    
    // Send the telemetry payload
    const res = await fetch('http://localhost:8000/authenticate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const data = await res.json();
    setDashboardData(data);
    
    // Clean the password bar and reset telemetry for the next attempt
    resetTelemetry(); 
    setPassword(''); 
    refreshNonce(); // Pre-fetch the next nonce instantly
    
    if (data.session_token) {
      sessionStorage.setItem("bio_session", data.session_token);
    }
  };

  return (
    <div {...bindContainer} style={{ display: 'flex', height: '100vh', backgroundColor: '#0f172a', color: 'white', fontFamily: 'sans-serif' }}>
      <div style={{ width: '50%', padding: '40px', borderRight: '1px solid #334155', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        <h1 style={{ color: '#38bdf8', marginBottom: '5px' }}>BioPrint Security</h1>
        <p style={{ color: '#94a3b8', marginBottom: '30px' }}>Cryptographically secured behavioral footprinting.</p>
        
        <select 
          value={deviceContext} 
          onChange={(e) => setDeviceContext(e.target.value)}
          style={{ marginBottom: '15px', padding: '12px', width: '100%', backgroundColor: '#1e293b', color: 'white', border: '1px solid #334155', borderRadius: '4px' }}
        >
          <option value="desktop-mouse">Desktop / External Mouse</option>
          <option value="laptop-touchpad">Laptop / Touchpad</option>
        </select>

        <input 
          value={userId} 
          onChange={(e) => setUserId(e.target.value)} 
          placeholder="Username" 
          style={{ marginBottom: '15px', padding: '12px', width: '100%', boxSizing: 'border-box', backgroundColor: '#1e293b', color: 'white', border: '1px solid #334155', borderRadius: '4px' }}
        />
        
        {/* Password Input with Visibility Toggle */}
        <div style={{ position: 'relative', marginBottom: '20px', width: '100%' }}>
          <input 
            type={showPassword ? "text" : "password"} 
            {...bindInput} 
            value={password} 
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password" 
            style={{ padding: '12px', width: '100%', boxSizing: 'border-box', backgroundColor: '#1e293b', color: 'white', border: '1px solid #334155', borderRadius: '4px', paddingRight: '70px' }}
          />
          <button 
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            style={{ position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '12px', fontWeight: 'bold' }}
            tabIndex="-1" // Prevents the toggle from interfering with keyboard telemetry
          >
            {showPassword ? "HIDE" : "SHOW"}
          </button>
        </div>
        
        <div style={{ display: 'flex', gap: '10px' }}>
          <button onClick={handleEnroll} disabled={enrollBuffer.length >= 3 || !userId} style={{ flex: 1, padding: '12px', backgroundColor: '#334155', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
            Enroll ({enrollBuffer.length}/3)
          </button>
          <button onClick={handleLogin} disabled={!userId} style={{ flex: 1, padding: '12px', backgroundColor: '#0ea5e9', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>
            Authenticate
          </button>
        </div>
      </div>

      <div style={{ width: '50%', padding: '40px', backgroundColor: '#020617', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        <h2 style={{ color: '#f8fafc' }}>Live Telemetry Dashboard</h2>
        {dashboardData ? (
          <div>
            <div style={{ padding: '20px', borderRadius: '6px', backgroundColor: dashboardData.authenticated ? '#064e3b' : '#7f1d1d', border: `1px solid ${dashboardData.authenticated ? '#059669' : '#dc2626'}` }}>
              <h3 style={{ margin: '0 0 10px 0' }}>{dashboardData.summary}</h3>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#cbd5e1', fontSize: '14px' }}>
                <span>Inference Latency: <strong>{dashboardData.latency_ms} ms</strong></span>
                <span>Trust Score: <strong>{dashboardData.confidence_score}%</strong></span>
              </div>
            </div>
            {dashboardData.flagged_signals?.length > 0 && (
              <div style={{ marginTop: '20px', padding: '15px', backgroundColor: '#1e293b', borderRadius: '6px' }}>
                <span style={{ fontSize: '12px', color: '#f59e0b', fontWeight: 'bold' }}>ANOMALIES DETECTED</span>
                <ul style={{ color: '#cbd5e1', marginTop: '10px', paddingLeft: '20px', fontSize: '14px' }}>
                  {dashboardData.flagged_signals.map((sig, i) => <li key={i}>{sig}</li>)}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <p style={{ color: '#64748b' }}>Awaiting authentication sequence...</p>
        )}
      </div>
    </div>
  );
}