import PropTypes from "prop-types";

import ExecutionEmptyState from "./ExecutionEmptyState";
import ExecutionMetadata from "./ExecutionMetadata";
import ExecutionResultViewer from "./ExecutionResultViewer";

function formatStatus(status) {
  if (!status) {
    return "Unknown";
  }

  return String(status)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function getStatusClass(status) {
  if (!status) {
    return "execution-status";
  }

  return `execution-status execution-status-${String(status).toLowerCase()}`;
}

function getExecutionIcon(status) {
  const normalizedStatus = String(status ?? "").toLowerCase();

  if (
    normalizedStatus === "completed" ||
    normalizedStatus === "success" ||
    normalizedStatus === "succeeded"
  ) {
    return "✓";
  }

  if (normalizedStatus === "failed" || normalizedStatus === "error") {
    return "!";
  }

  if (normalizedStatus === "running" || normalizedStatus === "in_progress") {
    return "◌";
  }

  return "◆";
}

function getTransformationType(execution) {
  return (
    execution?.execution_context?.transformation_type ??
    execution?.transformation_type ??
    "Transformation"
  );
}

function formatTransformationType(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function ExecutionInspector({ execution, loading = false }) {
  return (
    <aside className="execution-detail-card">
      <div className="executions-card-header">
        <div>
          <span className="transformation-card-kicker">INSPECTOR</span>

          <h2>Execution details</h2>

          <p>
            Inspect the selected run, workflow metadata, result, context, and
            metrics.
          </p>
        </div>
      </div>

      {loading ? (
        <div className="executions-loading-state">
          <div className="executions-loading-spinner" aria-hidden="true" />

          <div>
            <strong>Loading execution</strong>

            <span>Retrieving execution details...</span>
          </div>
        </div>
      ) : !execution ? (
        <ExecutionEmptyState mode="detail" />
      ) : (
        <div className="execution-detail-content">
          <div className="execution-detail-identity">
            <div className="execution-detail-identity-icon" aria-hidden="true">
              {getExecutionIcon(execution.status)}
            </div>

            <div className="execution-detail-identity-copy">
              <span>
                {formatTransformationType(getTransformationType(execution))}
              </span>

              <h3>
                {execution.id
                  ? `Execution ${String(execution.id).slice(0, 8)}…`
                  : "Unnamed execution"}
              </h3>
            </div>

            <span className={getStatusClass(execution.status)}>
              {formatStatus(execution.status)}
            </span>
          </div>

          <ExecutionMetadata execution={execution} />

          <ExecutionResultViewer
            output={execution.output}
            executionContext={execution.execution_context}
            metrics={execution.metrics}
            error={execution.error}
          />
        </div>
      )}
    </aside>
  );
}

ExecutionInspector.propTypes = {
  execution: PropTypes.shape({
    id: PropTypes.string,
    status: PropTypes.string,
    transformation_type: PropTypes.string,
    transformation_id: PropTypes.string,
    workflow_id: PropTypes.string,
    workflow_version: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    created_at: PropTypes.string,
    started_at: PropTypes.string,
    completed_at: PropTypes.string,
    output: PropTypes.any,
    execution_context: PropTypes.object,
    metrics: PropTypes.any,
    error: PropTypes.string,
  }),
  loading: PropTypes.bool,
};

export default ExecutionInspector;
