import React, { useEffect, useRef, useState } from "react";
import Chart from "chart.js/auto";
import "./ReportPage.css";
import bg from "../../assets/background.png";
import { resolveBackendUrl, formatClock } from "../../lib/backend";

const dash = "—";
const show = (v, fmt = (x) => x) => (v === null || v === undefined ? dash : fmt(v));

// Fetches the report for exactly `sessionId` (live or from History) — it never
// reuses metrics from the previous live session.
export default function ReportPage({ sessionId, onBack, backendUrl }) {
  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);
  const [report, setReport] = useState(null);
  const [status, setStatus] = useState(sessionId ? "loading" : "missing"); // loading | ok | missing | error
  const [errorText, setErrorText] = useState(null);
  const [attempt, setAttempt] = useState(0);

  const BACKEND_URL = backendUrl || resolveBackendUrl();

  // 🧠 Fetch report from FastAPI
  useEffect(() => {
    if (!sessionId) {
      setStatus("missing");
      return undefined;
    }
    const controller = new AbortController();
    setStatus("loading");
    (async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/report/${encodeURIComponent(sessionId)}`, {
          signal: controller.signal,
        });
        if (response.status === 404) {
          setStatus("missing");
          return;
        }
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const result = await response.json();
        if (!result || !result.report) throw new Error("Malformed report");
        setReport(result.report);
        setStatus("ok");
      } catch (error) {
        if (controller.signal.aborted) return;
        console.error("❌ Error fetching report:", error);
        setErrorText(error.message);
        setStatus("error");
      }
    })();
    return () => controller.abort();
  }, [sessionId, BACKEND_URL, attempt]);

  // 📈 Draw the recorded trend (never simulated)
  const trend = report && report.trend;
  useEffect(() => {
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
      chartInstanceRef.current = null;
    }
    if (!trend || !chartRef.current) return undefined;

    const datasets = [];
    if (trend.posture) {
      datasets.push({
        label: "Confidence (Posture)",
        data: trend.posture,
        borderColor: "rgba(125, 82, 245, 0.9)",
        backgroundColor: "rgba(110, 92, 226, 0.25)",
        tension: 0.4,
        fill: true,
        borderWidth: 2.5,
        spanGaps: true,
      });
    }
    if (trend.wpm) {
      datasets.push({
        label: "Pacing (WPM)",
        data: trend.wpm,
        borderColor: "rgba(34,197,94,0.9)",
        backgroundColor: "rgba(74,222,128,0.2)",
        tension: 0.4,
        fill: true,
        borderWidth: 2.5,
        spanGaps: true,
      });
    }

    chartInstanceRef.current = new Chart(chartRef.current.getContext("2d"), {
      type: "line",
      data: { labels: trend.t.map(formatClock), datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 1200, easing: "easeInOutQuart" },
        scales: {
          y: {
            beginAtZero: true,
            suggestedMax: 100,
            ticks: { color: "#4B5563", font: { size: 13 } },
          },
          x: {
            title: { display: true, text: "Session time (m:ss)", color: "#6B7280" },
            ticks: { color: "#6B7280", font: { size: 12 } },
          },
        },
        plugins: {
          legend: {
            position: "top",
            labels: { color: "#4B5563", font: { size: 13, weight: 600 } },
          },
        },
      },
    });
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [trend]);

  if (status === "loading") return <p className="loading glass">Loading report...</p>;

  if (status !== "ok" || !report)
    return (
      <div className="report-root" style={{ backgroundImage: `url(${bg})` }}>
        <header className="topbar glass">
          <div className="brand">Aura Coach</div>
        </header>
        <div className="report-container glass">
          <h2>{status === "error" ? "Couldn't load this report 😕" : "No report found 😕"}</h2>
          {status === "error" && <p>{errorText}. Check that the backend is running.</p>}
          {status === "error" && (
            <button className="btn start" onClick={() => setAttempt((n) => n + 1)}>↻ Retry</button>
          )}
          <button className="btn start" onClick={onBack}>
            ← Back to Home
          </button>
        </div>
      </div>
    );

  const m = report.metrics || {};
  const eyeText = m.eye_contact
    ? `${m.eye_contact}${m.eye_contact_ratio !== null && m.eye_contact_ratio !== undefined ? ` (${Math.round(m.eye_contact_ratio * 100)}%)` : ""}`
    : dash;

  return (
    <div className="report-root" style={{ backgroundImage: `url(${bg})` }}>
      <header className="topbar glass">
        <div className="brand">Aura Coach</div>
      </header>

      <div className="report-container">
        {/* Summary */}
        <div className="glass overview-box">
          <h2>Session Summary</h2>
          <p>{report.summary}</p>
          {report.ml_performance_score !== null && report.ml_performance_score !== undefined && (
            <div className="ml-score-badge">
              <span className="ml-score-label">🧠 ML Performance Score</span>
              <span className="ml-score-value">{report.ml_performance_score}/100</span>
            </div>
          )}
          {report.ml_score_note && <p className="ml-score-note"><small>{report.ml_score_note}</small></p>}
        </div>

        {/* Stats */}
        <div className="stats-row">
          <div className="glass box">
            <h3>WPM</h3>
            <p className="big">{show(m.wpm)}</p>
            <span>Words per Minute</span>
          </div>
          <div className="glass box">
            <h3>Posture</h3>
            <p className="big">{show(m.posture, (x) => `${x}%`)}</p>
            <span>Body Alignment</span>
          </div>
          <div className="glass box" title="Estimated camera-facing gaze; not a medical eye-tracker.">
            <h3>Eye Contact</h3>
            <p className="big">{eyeText}</p>
            <span>Engagement</span>
          </div>
          <div className="glass box">
            <h3>Volume</h3>
            <p className="big">{show(m.volume, (x) => Number(x).toFixed(1))}</p>
            <span>Voice Power</span>
          </div>
          <div className="glass box">
            <h3>Emotion</h3>
            <p className="big">{show(m.emotion)}</p>
            <span>Expression</span>
          </div>
        </div>

        {/* Chart + Insights */}
        <div className="middle-row">
          <div className="glass chart-box" style={{ height: "320px" }}>
            <h2>Pacing & Confidence Trend</h2>
            {trend ? (
              <canvas ref={chartRef}></canvas>
            ) : (
              <p>Trend not recorded for this session.</p>
            )}
          </div>

          <div className="glass feedback-box">
            <h2>Insights</h2>
            <ul>
              <li>{report.insights.posture}</li>
              {report.insights.head_tilt && <li>{report.insights.head_tilt}</li>}
              <li>{report.insights.eye_contact}</li>
              <li>{report.insights.emotion}</li>
              {report.insights.tone && <li>{report.insights.tone}</li>}
            </ul>
          </div>
        </div>

        {/* Recommendations */}
        <div className="glass feedback-box">
          <h2>Recommendations</h2>
          <ul>
            {report.recommendations.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>

        <div className="actions">
          <button className="btn start" onClick={onBack}>
            ← Back to Home
          </button>
        </div>
      </div>
    </div>
  );
}
