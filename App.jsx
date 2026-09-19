import React, { useState } from 'react';
import { useBiometrics } from './hooks/useBiometrics';

export default function App() {
  const [userId, setUserId] = useState('user1');
  const [password, setPassword] = useState('');
  const [dashboardData, setDashboardData] = useState(null);
  const [enrollmentSamples, setEnrollmentSamples] = useState([]);
  
  const { bindInput, bindContainer, getTelemetryPayload, resetTelemetry } = useBiometrics();

  const handleEnrollSubmit = async (e) => {
    e.preventDefault();
    const payload = getTelemetryPayload(userId);
    const updated = [...enrollmentSamples, payload];
    setEnrollmentSamples(updated);
    resetTelemetry();
    setPassword('');

    if (updated.length === 5) {
      await fetch('http://localhost:8000/enroll', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, samples: updated })
      });
      alert('Enrollment complete! Profile trained.');
    }
  };

  const handleAuthSubmit = async (e) => {
    e.preventDefault();
    const payload = getTelemetryPayload(userId);
    
    const res = await fetch('http://localhost:8000/authenticate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const data = await res.json();
    setDashboardData(data);
    resetTelemetry();
    setPassword('');
  };

  return (
    <div {...bindContainer} className="min-h-screen bg-slate-900 text-white flex">
      {/* Left Pane: Credential Form */}
      <div className="w-1/2 p-12 border-r border-slate-700 flex flex-col justify-center">
        <h1 className="text-3xl font-bold mb-6 text-cyan-400">BioPrint Gateway</h1>
        <div className="mb-4">
          <label className="block text-sm mb-2 text-slate-300">Target User ID</label>
          <input 
            type="text" 
            value={userId} 
            onChange={(e) => setUserId(e.target.value)}
            className="w-full p-3 rounded bg-slate-800 border border-slate-600 focus:outline-none focus:border-cyan-400"
          />
        </div>
        <div className="mb-6">
          <label className="block text-sm mb-2 text-slate-300">
            Password (Execute Intentional Typo Signature)
          </label>
          <input 
            type="password" 
            {...bindInput}
            value={password} 
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Type secret pattern..."
            className="w-full p-3 rounded bg-slate-800 border border-slate-600 focus:outline-none focus:border-cyan-400"
          />
        </div>
        <div className="flex gap-4">
          <button 
            onClick={handleEnrollSubmit} 
            disabled={enrollmentSamples.length >= 5}
            className="flex-1 bg-slate-700 hover:bg-slate-600 py-3 rounded font-medium disabled:opacity-40"
          >
            Enroll Attempt ({enrollmentSamples.length}/5)
          </button>
          <button 
            onClick={handleAuthSubmit} 
            className="flex-1 bg-cyan-600 hover:bg-cyan-500 py-3 rounded font-medium"
          >
            Authenticate
          </button>
        </div>
      </div>

      {/* Right Pane: Explainability Dashboard */}
      <div className="w-1/2 p-12 bg-slate-950 flex flex-col justify-center">
        <h2 className="text-2xl font-semibold mb-6 text-slate-200">Explainability Telemetry</h2>
        {dashboardData ? (
          <div className="space-y-4">
            <div className={`p-4 rounded border ${dashboardData.authenticated ? 'bg-emerald-950/40 border-emerald-500' : 'bg-rose-950/40 border-rose-500'}`}>
              <div className="text-lg font-bold">{dashboardData.summary}</div>
              <div className="text-sm mt-1 text-slate-400">Response Latency: {dashboardData.latency_ms} ms</div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-900 p-4 rounded border border-slate-800">
                <span className="text-xs text-slate-400 uppercase">Confidence Score</span>
                <div className="text-2xl font-bold mt-1 text-cyan-400">{dashboardData.confidence_score}%</div>
              </div>
              <div className="bg-slate-900 p-4 rounded border border-slate-800">
                <span className="text-xs text-slate-400 uppercase">Bot/Script Detected</span>
                <div className={`text-2xl font-bold mt-1 ${dashboardData.is_bot ? 'text-rose-500' : 'text-slate-300'}`}>
                  {dashboardData.is_bot ? 'YES' : 'NO'}
                </div>
              </div>
            </div>

            {dashboardData.flagged_signals && dashboardData.flagged_signals.length > 0 && (
              <div className="bg-slate-900 p-4 rounded border border-slate-800">
                <span className="text-xs text-amber-400 font-semibold uppercase">Flagged Structural Outliers</span>
                <ul className="mt-2 space-y-1 text-sm text-slate-300 list-disc list-inside">
                  {dashboardData.flagged_signals.map((sig, i) => (
                    <li key={i}>{sig}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <div className="text-slate-500 text-center py-20 border border-dashed border-slate-800 rounded">
            Waiting for live authentication input...
          </div>
        )}
      </div>
    </div>
  );
}
