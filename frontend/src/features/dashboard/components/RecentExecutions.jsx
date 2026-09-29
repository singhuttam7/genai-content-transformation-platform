import { useNavigate } from "react-router-dom";

function formatDate(value) {
  if (!value) {
    return "Unknown time";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return "Unknown time";
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function formatTransformationType(value) {
  if (!value) {
    return "Transformation";
  }

  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function getStatusClass(status) {
  const normalized = String(status ?? "UNKNOWN").toLowerCase();

  return `execution-status-${normalized}`;
}

function getExecutionTitle(execution) {
  const transformationType =
    execution?.execution_context?.transformation_type ??
    execution?.transformation_type;

  return formatTransformationType(transformationType);
}

function getExecutionSource(execution) {
  const sourceId = execution?.execution_context?.source_id;

  if (!sourceId) {
    return "Workflow execution";
  }

  return `Source ${String(sourceId).slice(0, 8)}…`;
}

function RecentExecutions({ executions = [] }) {
  const navigate = useNavigate();

  function openExecutions() {
    navigate("/executions");
  }

  return (
    <section className="dashboard-recent-card">
      <div className="dashboard-section-heading dashboard-section-heading-with-action">
        <div>
          <span className="dashboard-section-kicker">Activity</span>

          <h2>Recent executions</h2>
        </div>

        <button
          type="button"
          className="dashboard-section-link"
          onClick={openExecutions}
        >
          View all
          <span aria-hidden="true">→</span>
        </button>
      </div>

      {executions.length === 0 ? (
        <div className="dashboard-empty-state">
          <div className="dashboard-empty-icon" aria-hidden="true">
            ▶
          </div>

          <strong>No executions yet</strong>

          <p>Start a transformation to see execution activity here.</p>

          <button
            type="button"
            className="secondary-action-button"
            onClick={() => navigate("/transform")}
          >
            Create transformation
          </button>
        </div>
      ) : (
        <div className="dashboard-execution-list">
          {executions.map((execution) => {
            const status = String(execution?.status ?? "UNKNOWN").toUpperCase();

            return (
              <button
                key={execution.id}
                type="button"
                className="dashboard-execution-item"
                onClick={openExecutions}
              >
                <span className="dashboard-execution-icon" aria-hidden="true">
                  {status === "COMPLETED" ? "✓" : "▶"}
                </span>

                <span className="dashboard-execution-main">
                  <strong>{getExecutionTitle(execution)}</strong>

                  <span>{getExecutionSource(execution)}</span>
                </span>

                <span className="dashboard-execution-time">
                  {formatDate(execution?.created_at)}
                </span>

                <span
                  className={[
                    "execution-status",
                    getStatusClass(execution?.status),
                  ].join(" ")}
                >
                  {status}
                </span>

                <span className="dashboard-execution-arrow" aria-hidden="true">
                  →
                </span>
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}

export default RecentExecutions;
