import { useEffect, useState } from "react";

import { getExecution, getExecutions } from "../../services/api/executions";

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
    return "execution-status";
  }

  return `execution-status execution-status-${status.toLowerCase()}`;
}

function ExecutionsPage() {
  const [executions, setExecutions] = useState([]);
  const [selectedExecution, setSelectedExecution] = useState(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState(null);

  async function loadExecutions() {
    try {
      setLoading(true);
      setError(null);

      const response = await getExecutions();

      const items = Array.isArray(response)
        ? response
        : response?.items || response?.data || [];

      setExecutions(items);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleSelectExecution(executionId) {
    if (!executionId) {
      return;
    }

    try {
      setDetailLoading(true);
      setError(null);

      const execution = await getExecution(executionId);

      setSelectedExecution(execution);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setDetailLoading(false);
    }
  }

  useEffect(() => {
    loadExecutions();
  }, []);

  return (
    <section className="executions-page">
      <div className="executions-page-header">
        <div>
          <span className="eyebrow">EXECUTION CENTER</span>

          <h1>Executions</h1>

          <p>
            Monitor transformation runs, inspect workflow execution details, and
            review the current execution state.
          </p>
        </div>

        <button
          type="button"
          className="secondary-action-button"
          onClick={loadExecutions}
          disabled={loading}
        >
          {loading ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="executions-error">
          <strong>Unable to load executions</strong>
          <span>{error}</span>
        </div>
      )}

      <div className="executions-layout">
        <section className="executions-card">
          <div className="executions-card-header">
            <div>
              <span className="transformation-card-kicker">RUNS</span>
              <h2>Recent executions</h2>
            </div>

            <span className="executions-count">
              {executions.length}{" "}
              {executions.length === 1 ? "execution" : "executions"}
            </span>
          </div>

          {loading ? (
            <div className="executions-empty">
              <strong>Loading executions...</strong>
            </div>
          ) : executions.length === 0 ? (
            <div className="executions-empty">
              <strong>No executions yet</strong>
              <p>
                Start a transformation from the Transform workspace and its
                execution will appear here.
              </p>
            </div>
          ) : (
            <div className="execution-list">
              {executions.map((execution) => (
                <button
                  key={execution.id}
                  type="button"
                  className={`execution-list-item ${
                    selectedExecution?.id === execution.id ? "selected" : ""
                  }`}
                  onClick={() => handleSelectExecution(execution.id)}
                >
                  <div className="execution-list-main">
                    <strong>{execution.id || "Unnamed execution"}</strong>

                    <span>Created {formatDate(execution.created_at)}</span>
                  </div>

                  <div className="execution-list-meta">
                    <span className={getStatusClass(execution.status)}>
                      {execution.status || "UNKNOWN"}
                    </span>

                    <span>
                      {execution.workflow_id
                        ? `Workflow ${execution.workflow_id}`
                        : "Workflow unavailable"}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>

        <aside className="execution-detail-card">
          <div className="executions-card-header">
            <div>
              <span className="transformation-card-kicker">DETAILS</span>
              <h2>Execution details</h2>
            </div>
          </div>

          {detailLoading ? (
            <div className="executions-empty">
              <strong>Loading execution...</strong>
            </div>
          ) : !selectedExecution ? (
            <div className="executions-empty">
              <strong>Select an execution</strong>
              <p>
                Choose a run from the list to inspect its workflow and execution
                metadata.
              </p>
            </div>
          ) : (
            <div className="execution-detail-content">
              <div className="execution-detail-status">
                <span>Status</span>

                <strong className={getStatusClass(selectedExecution.status)}>
                  {selectedExecution.status || "UNKNOWN"}
                </strong>
              </div>

              <div className="execution-detail-row">
                <span>Execution ID</span>
                <strong>{selectedExecution.id || "—"}</strong>
              </div>

              <div className="execution-detail-row">
                <span>Transformation ID</span>
                <strong>{selectedExecution.transformation_id || "—"}</strong>
              </div>

              <div className="execution-detail-row">
                <span>Workflow ID</span>
                <strong>{selectedExecution.workflow_id || "—"}</strong>
              </div>

              <div className="execution-detail-row">
                <span>Workflow version</span>
                <strong>{selectedExecution.workflow_version ?? "—"}</strong>
              </div>

              <div className="execution-detail-row">
                <span>Created</span>
                <strong>{formatDate(selectedExecution.created_at)}</strong>
              </div>

              <div className="execution-detail-row">
                <span>Started</span>
                <strong>{formatDate(selectedExecution.started_at)}</strong>
              </div>

              <div className="execution-detail-row">
                <span>Completed</span>
                <strong>{formatDate(selectedExecution.completed_at)}</strong>
              </div>

              {selectedExecution.error && (
                <div className="execution-detail-error">
                  <span>Error</span>
                  <strong>{selectedExecution.error}</strong>
                </div>
              )}

              <div className="execution-detail-section">
                <span className="execution-detail-section-title">
                  Execution result
                </span>

                <pre className="execution-detail-json">
                  {formatJson(selectedExecution.output)}
                </pre>
              </div>

              <div className="execution-detail-section">
                <span className="execution-detail-section-title">
                  Execution context
                </span>

                <pre className="execution-detail-json">
                  {formatJson(selectedExecution.execution_context)}
                </pre>
              </div>

              <div className="execution-detail-section">
                <span className="execution-detail-section-title">Metrics</span>

                <pre className="execution-detail-json">
                  {formatJson(selectedExecution.metrics)}
                </pre>
              </div>
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}

export default ExecutionsPage;
