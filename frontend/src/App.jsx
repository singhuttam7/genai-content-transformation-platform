import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { useEffect, useState } from "react";

import AppShell from "./components/layout/AppShell";

import { getHealth } from "./services/api/health";

import TransformationPage from "./features/transformation/TransformationPage";
import ExecutionsPage from "./features/executions/ExecutionsPage";
import ArtifactsPage from "./features/artifacts/ArtifactsPage";

import { WorkspaceProvider } from "./context/WorkspaceContext";

function DashboardPage() {
  const [health, setHealth] = useState(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState(null);

  useEffect(() => {
    let isMounted = true;

    async function loadHealth() {
      try {
        const data = await getHealth();

        if (isMounted) {
          setHealth(data);
          setHealthError(null);
        }
      } catch (error) {
        if (isMounted) {
          setHealthError(error.message);
        }
      } finally {
        if (isMounted) {
          setHealthLoading(false);
        }
      }
    }

    loadHealth();

    return () => {
      isMounted = false;
    };
  }, []);

  const isHealthy =
    health?.status === "healthy" && health?.database === "healthy";

  let statusText = "Checking platform status...";

  if (!healthLoading) {
    if (isHealthy) {
      statusText = "Platform operational";
    } else if (healthError) {
      statusText = "Platform unavailable";
    } else {
      statusText = "Platform requires attention";
    }
  }

  return (
    <section className="hero">
      <span className="eyebrow">GENAI CONTENT PLATFORM</span>

      <h1>
        Transform information into
        <span> intelligent content.</span>
      </h1>

      <p>
        A multimodal AI platform for transforming reports, advisories, articles,
        documents, images, videos, and prompts into audience-ready communication
        artifacts.
      </p>

      <div className="status">
        <span
          className={`status-dot ${isHealthy ? "status-dot-healthy" : ""}`}
        />

        {statusText}
      </div>
    </section>
  );
}

function PlaceholderPage({ title, description }) {
  return (
    <section className="hero">
      <span className="eyebrow">GENAI CONTENT PLATFORM</span>

      <h1>{title}</h1>

      <p>{description}</p>
    </section>
  );
}

function App() {
  return (
    <BrowserRouter>
      <WorkspaceProvider>
        <AppShell>
          <Routes>
            <Route path="/" element={<DashboardPage />} />

            <Route path="/transform" element={<TransformationPage />} />

            <Route path="/executions" element={<ExecutionsPage />} />

            <Route path="/artifacts" element={<ArtifactsPage />} />

            <Route
              path="/knowledge"
              element={
                <PlaceholderPage
                  title="Manage knowledge."
                  description="Manage contextual knowledge sources for transformation workflows."
                />
              }
            />

            <Route
              path="/workflows"
              element={
                <PlaceholderPage
                  title="Build workflows."
                  description="Design and manage automated transformation workflows."
                />
              }
            />

            <Route
              path="/agents"
              element={
                <PlaceholderPage
                  title="Manage agents."
                  description="Configure specialized AI transformation agents."
                />
              }
            />

            <Route
              path="/integrations"
              element={
                <PlaceholderPage
                  title="Configure integrations."
                  description="Connect external services to the content transformation platform."
                />
              }
            />

            <Route
              path="/settings"
              element={
                <PlaceholderPage
                  title="Platform settings."
                  description="Configure workspace and platform preferences."
                />
              }
            />

            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AppShell>
      </WorkspaceProvider>
    </BrowserRouter>
  );
}

export default App;
