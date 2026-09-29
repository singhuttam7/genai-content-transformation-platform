import PropTypes from "prop-types";

function formatDate(value) {
  if (!value) {
    return "Unknown date";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function formatStatus(value) {
  if (!value) {
    return "Unknown";
  }

  return String(value)
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

function ExecutionListItem({ execution, selected = false, onSelect }) {
  const status = execution?.status;
  const transformationType = getTransformationType(execution);

  return (
    <button
      type="button"
      className={["execution-list-item", selected ? "selected" : ""].join(" ")}
      onClick={() => onSelect(execution?.id)}
      aria-pressed={selected}
    >
      <span className="execution-list-icon" aria-hidden="true">
        {getExecutionIcon(status)}
      </span>

      <span className="execution-list-main">
        <strong>{formatTransformationType(transformationType)}</strong>

        <span>
          {execution?.id
            ? `Execution ${String(execution.id).slice(0, 8)}…`
            : "Unnamed execution"}
        </span>

        <span>Created {formatDate(execution?.created_at)}</span>
      </span>

      <span className="execution-list-meta">
        <span className={getStatusClass(status)}>{formatStatus(status)}</span>

        <span className="execution-list-workflow">
          {execution?.workflow_id
            ? `Workflow ${String(execution.workflow_id).slice(0, 8)}…`
            : "Workflow unavailable"}
        </span>
      </span>

      <span className="execution-list-arrow" aria-hidden="true">
        →
      </span>
    </button>
  );
}

ExecutionListItem.propTypes = {
  execution: PropTypes.shape({
    id: PropTypes.string,
    status: PropTypes.string,
    created_at: PropTypes.string,
    workflow_id: PropTypes.string,
    transformation_type: PropTypes.string,
    execution_context: PropTypes.shape({
      transformation_type: PropTypes.string,
    }),
  }).isRequired,
  selected: PropTypes.bool,
  onSelect: PropTypes.func.isRequired,
};

export default ExecutionListItem;
