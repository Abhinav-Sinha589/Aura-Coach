import React, { useEffect, useState } from "react";
import "./SessionHistoryPage.css";
import bg from "../../assets/background.png";
import { resolveBackendUrl } from "../../lib/backend";

const dash = "—";
const fmtDate = (ts) => {
  const d = ts ? new Date(ts) : null;
  return d && !Number.isNaN(d.getTime()) ? d.toLocaleString() : dash;
};

export default function SessionHistoryPage({ onBack, onOpenReport, backendUrl }) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const BACKEND_URL = backendUrl || resolveBackendUrl();

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/sessions`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setSessions(data.sessions || []);
      } catch (err) {
        console.error("Error fetching sessions:", err);
        setError("Couldn't load sessions. Check that the backend is running.");
      } finally {
        setLoading(false);
      }
    };
    fetchSessions();
  }, [BACKEND_URL]);

  if (loading) return <div className="loading glass">Loading sessions...</div>;

  return (
    <div className="history-root" style={{ backgroundImage: `url(${bg})` }}>
      <header className="topbar glass">
        <div className="brand">Aura Coach</div>
        <button className="home-link" onClick={onBack}>
          ← Back
        </button>
      </header>

      <div className="history-container glass">
        <h2>Previous Sessions</h2>

        {error ? (
          <p>{error}</p>
        ) : sessions.length === 0 ? (
          <p>No saved sessions yet.</p>
        ) : (
          <table className="session-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>WPM</th>
                <th>Posture</th>
                <th>Emotion</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.session_id}>
                  <td>{fmtDate(s.timestamp)}</td>
                  <td>{s.wpm ?? dash}</td>
                  <td>{s.posture === null || s.posture === undefined ? dash : `${s.posture}%`}</td>
                  <td>{s.emotion ?? dash}</td>
                  <td>
                    <button
                      className="btn small"
                      onClick={() => onOpenReport(s.session_id)}
                    >
                      View Report
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
