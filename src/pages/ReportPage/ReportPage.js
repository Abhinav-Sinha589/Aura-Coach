import React, { useEffect, useMemo, useRef, useState } from "react";
import Chart from "chart.js/auto";
import "./ReportPage.css";
import bg from "../../assets/background.png";

// Accept backendUrl as an optional prop (recommended for flexibility)
export default function ReportPage({ data, sessionId, onBack, backendUrl }) {
  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState(null);

  const BACKEND_URL =
    backendUrl ||
    process.env.REACT_APP_BACKEND_URL ||
    (typeof window !== "undefined" &&
    window.location.hostname !== "localhost" &&
    window.location.hostname !== "127.0.0.1"
      ? window.location.origin
      : "http://127.0.0.1:8010");

  // 🧠 Fetch report from FastAPI with AbortController / stale-response guard
  useEffect(() => {
    let active = true;
    const controller = typeof AbortController !== "undefined" ? new AbortController() : null;

    if (!sessionId) {
      setReport(null);
      setFetchError("No session ID provided.");
      setLoading(false);
      return undefined;
    }

    setLoading(true);
    setFetchError(null);
    setReport(null);

    const fetchReport = async () => {
      try {
        const response = await fetch(
          `${BACKEND_URL}/api/report/${encodeURIComponent(sessionId)}`,
          controller ? { signal: controller.signal } : undefined
        );
        let result = null;
        try {
          result = await response.json();
        } catch (_e) {
          result = null;
        }
        if (!active) return;

        if (!response.ok || !result || !result.report) {
          const msg =
            response.status === 404
              ? "Session report not found (404)."
              : result?.error || `Failed to load report (HTTP ${response.status}).`;
          setFetchError(msg);
          setReport(null);
          return;
        }
        setReport(result.report);
      } catch (error) {
        if (!active || error?.name === "AbortError") return;
        setFetchError("Unable to reach backend to load report.");
        setReport(null);
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    fetchReport();

    return () => {
      active = false;
      if (controller) controller.abort();
    };
  }, [sessionId, BACKEND_URL]);

  const timeline = useMemo(
    () =>
      Array.isArray(report?.timeline)
        ? report.timeline.filter(
            (pt) =>
              pt &&
              typeof pt.t === "number" &&
              Number.isFinite(pt.t) &&
              ((typeof pt.posture === "number" && Number.isFinite(pt.posture)) ||
                (typeof pt.wpm === "number" && Number.isFinite(pt.wpm)))
          )
        : [],
    [report]
  );
  const hasRealTimeline = timeline.length >= 2;

  // 📈 Draw chart ONLY when real timestamped checkpoints exist; clean up on unmount
  useEffect(() => {
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
      chartInstanceRef.current = null;
    }
    if (!hasRealTimeline || !chartRef.current) return undefined;

    const ctx = chartRef.current.getContext("2d");
    const labels = timeline.map((pt) => `${Math.round(pt.t)}s`);
    const postureSeries = timeline.map((pt) =>
      typeof pt.posture === "number" && Number.isFinite(pt.posture) ? pt.posture : null
    );
    const wpmSeries = timeline.map((pt) =>
      typeof pt.wpm === "number" && Number.isFinite(pt.wpm) ? pt.wpm : null
    );

    chartInstanceRef.current = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Posture (%)",
            data: postureSeries,
            yAxisID: "yPosture",
            borderColor: "rgba(125, 82, 245, 0.9)",
            backgroundColor: "rgba(110, 92, 226, 0.25)",
            tension: 0.3,
            spanGaps: true,
            borderWidth: 2.5,
          },
          {
            label: "Speaking Pace (WPM)",
            data: wpmSeries,
            yAxisID: "yWpm",
            borderColor: "rgba(34, 197, 94, 0.9)",
            backgroundColor: "rgba(74, 222, 128, 0.2)",
            tension: 0.3,
            spanGaps: true,
            borderWidth: 2.5,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          yPosture: {
            type: "linear",
            position: "left",
            min: 0,
            max: 100,
            title: { display: true, text: "Posture (%)" },
          },
          yWpm: {
            type: "linear",
            position: "right",
            beginAtZero: true,
            grid: { drawOnChartArea: false },
            title: { display: true, text: "WPM" },
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
  }, [hasRealTimeline, timeline]);

  if (loading) return <p className="loading glass">Loading report...</p>;

  if (!report) {
    return (
      <div className="report-root" style={{ backgroundImage: `url(${bg})` }}>
        <header className="topbar glass">
          <div className="brand">Aura Coach</div>
        </header>
        <div className="report-container glass">
          <h2>No report found 😕</h2>
          {fetchError && <p>{fetchError}</p>}
          <button className="btn start" onClick={onBack}>
            ← Back to Home
          </button>
        </div>
      </div>
    );
  }

  // Prefer backend report.metrics for the selected sessionId (never overwrite with stale App props)
  const m = report.metrics || data || {};
  const formatNum = (val, decimals = 0, suffix = "") => {
    if (typeof val !== "number" || !Number.isFinite(val)) return "Unavailable";
    return `${decimals > 0 ? val.toFixed(decimals) : Math.round(val)}${suffix}`;
  };

  const insights = report.insights && typeof report.insights === "object" ? report.insights : {};
  const recommendations = Array.isArray(report.recommendations) ? report.recommendations : [];

  return (
    <div className="report-root" style={{ backgroundImage: `url(${bg})` }}>
      <header className="topbar glass">
        <div className="brand">Aura Coach</div>
      </header>

      <div className="report-container">
        {/* Summary */}
        <div className="glass overview-box">
          <h2>Session Summary</h2>
          <p>{report.summary || "Session summary unavailable."}</p>
          {typeof report.ml_performance_score === "number" && Number.isFinite(report.ml_performance_score) ? (
            <div className="ml-score-badge">
              <span className="ml-score-label">🧠 Estimated Performance Score (heuristic model)</span>
              <span className="ml-score-value">{report.ml_performance_score.toFixed(1)}/100</span>
            </div>
          ) : (
            <p className="ml-score-note">
              Estimated score withheld (requires measured WPM and posture).
            </p>
          )}
        </div>

        {/* Stats */}
        <div className="stats-row">
          <div className="glass box">
            <h3>WPM</h3>
            <p className="big">{formatNum(m.wpm, 0)}</p>
            <span>Words per Minute</span>
          </div>
          <div className="glass box">
            <h3>Posture</h3>
            <p className="big">{formatNum(m.posture, 0, "%")}</p>
            <span>Shoulder Alignment</span>
          </div>
          <div className="glass box">
            <h3>Camera Alignment</h3>
            <p className="big">{m.eyeContact || "Unavailable"}</p>
            <span>Face Framing (approx.)</span>
          </div>
          <div className="glass box">
            <h3>Volume</h3>
            <p className="big">{formatNum(m.volume, 1)}</p>
            <span>RMS × 100</span>
          </div>
          <div className="glass box">
            <h3>Expression</h3>
            <p className="big">{m.emotion || "Unavailable"}</p>
            <span>Facial Estimate</span>
          </div>
        </div>

        {/* Chart + Insights */}
        <div className="middle-row">
          <div className="glass chart-box" style={{ height: "320px" }}>
            <h2>Pacing &amp; Posture Timeline</h2>
            {hasRealTimeline ? (
              <canvas ref={chartRef}></canvas>
            ) : (
              <p className="empty-timeline">No session timeline recorded</p>
            )}
          </div>

          <div className="glass feedback-box">
            <h2>Insights</h2>
            <ul>
              {insights.posture && <li>{insights.posture}</li>}
              {insights.eye_contact && <li>{insights.eye_contact}</li>}
              {insights.emotion && <li>{insights.emotion}</li>}
              {insights.tone && <li>{insights.tone}</li>}
            </ul>
          </div>
        </div>

        {/* Recommendations */}
        <div className="glass feedback-box">
          <h2>Recommendations</h2>
          <ul>
            {recommendations.map((r, i) => (
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
