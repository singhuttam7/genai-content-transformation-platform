import { useEffect, useState } from "react";

import { getArtifact, getArtifacts } from "../../services/api/artifacts";

function formatDate(value) {
  if (!value) {
    return "—";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

function formatJson(value) {
  if (value === null || value === undefined) {
    return "—";
  }

  if (typeof value === "string") {
    return value;
  }

  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function getStatusClass(status) {
  if (!status) {
    return "artifact-status";
  }

  return `artifact-status artifact-status-${status.toLowerCase()}`;
}

function getResponseData(response) {
  return response?.data ?? response;
}

function getErrorMessage(requestError) {
  return (
    requestError?.response?.data?.error?.message ||
    requestError?.response?.data?.detail ||
    requestError?.message ||
    "An unexpected error occurred."
  );
}

function ArtifactsPage() {
  const [artifacts, setArtifacts] = useState([]);
  const [selectedArtifact, setSelectedArtifact] = useState(null);

  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);

  const [error, setError] = useState(null);

  async function loadArtifacts() {
    try {
      setLoading(true);
      setError(null);

      const response = await getArtifacts();
      const data = getResponseData(response);

      const items = Array.isArray(data) ? data : data?.items || [];

      setArtifacts(items);

      /*
       * Clear the selected artifact if it no longer
       * exists after refreshing the list.
       */
      setSelectedArtifact((current) => {
        if (!current) {
          return null;
        }

        return items.some((artifact) => artifact.id === current.id)
          ? current
          : null;
      });
    } catch (requestError) {
      setError(getErrorMessage(requestError));
    } finally {
      setLoading(false);
    }
  }

  async function handleSelectArtifact(artifactId) {
    if (!artifactId) {
      return;
    }

    try {
      setDetailLoading(true);
      setError(null);

      const response = await getArtifact(artifactId);

      const artifact = getResponseData(response);

      setSelectedArtifact(artifact);
    } catch (requestError) {
      setError(getErrorMessage(requestError));
    } finally {
      setDetailLoading(false);
    }
  }

  useEffect(() => {
    loadArtifacts();
  }, []);

  return (
    <section className="artifacts-page">
      <div className="artifacts-page-header">
        <div>
          <span className="eyebrow">ARTIFACT CENTER</span>

          <h1>Artifacts</h1>

          <p>
            Review generated transformation artifacts and inspect their
            execution context, content, and metadata.
          </p>
        </div>

        <button
          type="button"
          className="secondary-action-button"
          onClick={loadArtifacts}
          disabled={loading}
        >
          {loading ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="artifacts-error">
          <strong>Unable to load artifacts</strong>

          <span>{error}</span>
        </div>
      )}

      <div className="artifacts-layout">
        <section className="artifacts-card">
          <div className="artifacts-card-header">
            <div>
              <span className="transformation-card-kicker">OUTPUTS</span>

              <h2>Generated artifacts</h2>
            </div>

            <span className="artifacts-count">
              {artifacts.length}{" "}
              {artifacts.length === 1 ? "artifact" : "artifacts"}
            </span>
          </div>

          {loading ? (
            <div className="artifacts-empty">
              <strong>Loading artifacts...</strong>
            </div>
          ) : artifacts.length === 0 ? (
            <div className="artifacts-empty">
              <strong>No artifacts yet</strong>

              <p>
                Generated outputs will appear here after a transformation
                execution produces artifacts.
              </p>
            </div>
          ) : (
            <div className="artifact-list">
              {artifacts.map((artifact) => (
                <button
                  key={artifact.id}
                  type="button"
                  className={`artifact-list-item ${
                    selectedArtifact?.id === artifact.id ? "selected" : ""
                  }`}
                  onClick={() => handleSelectArtifact(artifact.id)}
                >
                  <div className="artifact-list-main">
                    <strong>
                      {artifact.title ||
                        artifact.artifact_type ||
                        "Unnamed artifact"}
                    </strong>

                    <span>
                      {artifact.artifact_type || "Unknown artifact type"}
                    </span>

                    <span>Created {formatDate(artifact.created_at)}</span>
                  </div>

                  <div className="artifact-list-meta">
                    <span className={getStatusClass(artifact.status)}>
                      {artifact.status || "UNKNOWN"}
                    </span>

                    <span>
                      {artifact.execution_id
                        ? `Execution ${artifact.execution_id}`
                        : "Execution unavailable"}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>

        <aside className="artifact-detail-card">
          <div className="artifacts-card-header">
            <div>
              <span className="transformation-card-kicker">DETAILS</span>

              <h2>Artifact details</h2>
            </div>
          </div>

          {detailLoading ? (
            <div className="artifacts-empty">
              <strong>Loading artifact...</strong>
            </div>
          ) : !selectedArtifact ? (
            <div className="artifacts-empty">
              <strong>Select an artifact</strong>

              <p>
                Choose a generated artifact from the list to inspect its content
                and metadata.
              </p>
            </div>
          ) : (
            <div className="artifact-detail-content">
              <div className="artifact-detail-status">
                <span>Status</span>

                <strong className={getStatusClass(selectedArtifact.status)}>
                  {selectedArtifact.status || "UNKNOWN"}
                </strong>
              </div>

              <div className="artifact-detail-row">
                <span>Artifact ID</span>

                <strong>{selectedArtifact.id || "—"}</strong>
              </div>

              <div className="artifact-detail-row">
                <span>Type</span>

                <strong>{selectedArtifact.artifact_type || "—"}</strong>
              </div>

              <div className="artifact-detail-row">
                <span>Execution ID</span>

                <strong>{selectedArtifact.execution_id || "—"}</strong>
              </div>

              <div className="artifact-detail-row">
                <span>Transformation ID</span>

                <strong>{selectedArtifact.transformation_id || "—"}</strong>
              </div>

              <div className="artifact-detail-row">
                <span>Created</span>

                <strong>{formatDate(selectedArtifact.created_at)}</strong>
              </div>

              {selectedArtifact.storage_uri && (
                <div className="artifact-detail-row">
                  <span>Storage URI</span>

                  <strong>{selectedArtifact.storage_uri}</strong>
                </div>
              )}

              <div className="artifact-detail-section">
                <span className="artifact-detail-section-title">Content</span>

                <pre className="artifact-detail-json">
                  {formatJson(selectedArtifact.content)}
                </pre>
              </div>

              <div className="artifact-detail-section">
                <span className="artifact-detail-section-title">Metadata</span>

                <pre className="artifact-detail-json">
                  {formatJson(selectedArtifact.metadata)}
                </pre>
              </div>
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}

export default ArtifactsPage;
