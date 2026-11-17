import React, { useState, useRef, useEffect } from "react";
import "./App.css";
import HomePage from "./pages/HomePage/HomePage";
import TrainingPage from "./pages/TrainingPage/TrainingPage";
import ReportPage from "./pages/ReportPage/ReportPage";
import SessionHistoryPage from "./pages/SessionHistoryPage/SessionHistoryPage";

// ✅ Load backend URL from environment variable or auto-resolve to current origin in cloud deployment
const BACKEND_URL =
  process.env.REACT_APP_BACKEND_URL ||
  (typeof window !== "undefined" && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1"
    ? window.location.origin
    : "http://localhost:8010");

console.log("🔗 Backend URL:", BACKEND_URL);

export default function App() {
  const [view, setView] = useState("home");
  const [sessionData, setSessionData] = useState(null);
  const [sessionId, setSessionId] = useState(null);

  const streamRef = useRef(null);

  // Stop any residual stream if leaving the training view
  useEffect(() => {
    if (view !== "training" && streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }, [view]);

  // 🧩 Handle training session end — TrainingPage owns POST /api/session/log
  // and passes the finalized session summary + session_id here for navigation.
  const handleEndSession = (savedSession) => {
    if (savedSession) {
      setSessionData(savedSession);
      if (savedSession.session_id) {
        setSessionId(savedSession.session_id);
      }
    }
    setView("report");
  };

  // 🔄 Navigation Handlers
  const handleStartSession = () => setView("training");
  const handleBackToHome = () => setView("home");
  const handleViewHistory = () => setView("history");

  return (
    <div className="App">
      {/* 🏠 Home Page */}
      {view === "home" && (
        <HomePage onStart={handleStartSession} onHistory={handleViewHistory} />
      )}

      {/* 🧘 Training Page */}
      {view === "training" && (
        <TrainingPage
          onEndSession={handleEndSession}
          onBack={handleBackToHome}
          streamRef={streamRef}
          backendUrl={BACKEND_URL}
        />
      )}

      {/* 📊 Report Page */}
      {view === "report" && (
        <ReportPage
          data={sessionData}
          sessionId={sessionId}
          onBack={handleBackToHome}
          backendUrl={BACKEND_URL}
        />
      )}

      {/* 🗂 Session History Page */}
      {view === "history" && (
        <SessionHistoryPage
          onBack={handleBackToHome}
          onOpenReport={(id) => {
            setSessionId(id);
            setView("report");
          }}
          backendUrl={BACKEND_URL}
        />
      )}
    </div>
  );
}
