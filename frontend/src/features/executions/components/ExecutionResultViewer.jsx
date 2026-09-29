import PropTypes from "prop-types";

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

function ExecutionResultViewer({ output, executionContext, metrics, error }) {
  return (
    <div className="execution-result-viewer">
      {error && (
        <section className="execution-detail-section">
          <div className="execution-detail-section-heading">
            <span className="execution-detail-section-title">
              Execution error
            </span>
          </div>

          <div className="execution-detail-error">
            <span>Error</span>
            <strong>{error}</strong>
          </div>
        </section>
      )}

      <section className="execution-detail-section">
        <div className="execution-detail-section-heading">
          <span className="execution-detail-section-title">
            Execution result
          </span>
        </div>

        <pre className="execution-detail-json execution-result-block">
          {formatJson(output)}
        </pre>
      </section>

      <section className="execution-detail-section">
        <div className="execution-detail-section-heading">
          <span className="execution-detail-section-title">
            Execution context
          </span>
        </div>

        <pre className="execution-detail-json">
          {formatJson(executionContext)}
        </pre>
      </section>

      <section className="execution-detail-section">
        <div className="execution-detail-section-heading">
          <span className="execution-detail-section-title">Metrics</span>
        </div>

        <pre className="execution-detail-json">{formatJson(metrics)}</pre>
      </section>
    </div>
  );
}

ExecutionResultViewer.propTypes = {
  output: PropTypes.any,
  executionContext: PropTypes.any,
  metrics: PropTypes.any,
  error: PropTypes.string,
};

export default ExecutionResultViewer;
