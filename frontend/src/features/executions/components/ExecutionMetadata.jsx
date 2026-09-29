import PropTypes from "prop-types";

function formatDate(value) {
  if (!value) {
    return "—";
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

function MetadataRow({ label, value }) {
  return (
    <div className="execution-detail-row">
      <span>{label}</span>

      <strong title={value || undefined}>{value || "—"}</strong>
    </div>
  );
}

MetadataRow.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.string,
};

function ExecutionMetadata({ execution }) {
  if (!execution) {
    return null;
  }

  return (
    <section className="execution-detail-section">
      <div className="execution-detail-section-heading">
        <span className="execution-detail-section-title">
          Execution metadata
        </span>
      </div>

      <div className="execution-detail-grid">
        <MetadataRow label="Execution ID" value={execution.id} />

        <MetadataRow
          label="Transformation ID"
          value={execution.transformation_id}
        />

        <MetadataRow label="Workflow ID" value={execution.workflow_id} />

        <MetadataRow
          label="Workflow version"
          value={
            execution.workflow_version !== null &&
            execution.workflow_version !== undefined
              ? String(execution.workflow_version)
              : undefined
          }
        />

        <MetadataRow label="Created" value={formatDate(execution.created_at)} />

        <MetadataRow label="Started" value={formatDate(execution.started_at)} />

        <MetadataRow
          label="Completed"
          value={formatDate(execution.completed_at)}
        />
      </div>
    </section>
  );
}

ExecutionMetadata.propTypes = {
  execution: PropTypes.shape({
    id: PropTypes.string,
    transformation_id: PropTypes.string,
    workflow_id: PropTypes.string,
    workflow_version: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    created_at: PropTypes.string,
    started_at: PropTypes.string,
    completed_at: PropTypes.string,
  }),
};

export default ExecutionMetadata;
