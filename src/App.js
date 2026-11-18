import React, { useState } from "react";
import "./App.css";
import HomePage from "./pages/HomePage/HomePage";
import TrainingPage from "./pages/TrainingPage/TrainingPage";
import ReportPage from "./pages/ReportPage/ReportPage";
import SessionHistoryPage from "./pages/SessionHistoryPage/SessionHistoryPage";
import { resolveBackendUrl } from "./lib/backend";

// ✅ Load backend URL from environment variable or auto-resolve to current origin in cloud deployment
const BACKEND_URL = resolveBackendUrl();

export default function App() {
  const [view, setView] = useState("home");
  const [sessionId, setSessionId] = useState(null);

  // TrainingPage owns the camera/microphone and saves the session exactly
  // once; App only navigates to that session's report.
  const handleEndSession = (savedSessionId) => {
    setSessionId(savedSessionId);
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
          backendUrl={BACKEND_URL}
        />
      )}

      {/* 📊 Report Page */}
      {view === "report" && (
        <ReportPage
          key={sessionId || "none"}
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
