import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import AppShell from "./components/layout/AppShell";

import TransformationPage from "./features/transformation/TransformationPage";
import ExecutionsPage from "./features/executions/ExecutionsPage";
import ArtifactsPage from "./features/artifacts/ArtifactsPage";
import DashboardPage from "./features/dashboard/DashboardPage";

import { WorkspaceProvider } from "./context/WorkspaceContext";
import KnowledgePage from "./features/knowledge/KnowledgePage";

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

            <Route path="/knowledge" element={<KnowledgePage />} />

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
