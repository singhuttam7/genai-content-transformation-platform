import PropTypes from "prop-types";

function formatLabel(value) {
  if (!value) {
    return "—";
  }

  return value
    .toString()
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatDate(value) {
  if (!value) {
    return "—";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function KnowledgeInspector({ source, onClose }) {
  if (!source) {
    return (
      <aside className="knowledge-inspector-card knowledge-inspector-empty">
        <div className="knowledge-inspector-placeholder">
          <div className="knowledge-empty-icon">◇</div>

          <h3>Select a knowledge source</h3>

          <p>
            Choose a source from the library to inspect its metadata and
            processing information.
          </p>
        </div>
      </aside>
    );
  }

  const title =
    source?.title ||
    source?.filename ||
    source?.original_filename ||
    "Untitled source";

  const sourceType = source?.input_type || source?.inputType || "unknown";

  const status =
    source?.status ||
    source?.processing_status ||
    source?.processingStatus ||
    "unknown";

  const metadata =
    source?.metadata && typeof source.metadata === "object"
      ? source.metadata
      : {};

  return (
    <aside className="knowledge-inspector-card">
      <div className="knowledge-inspector-header">
        <div>
          <span className="eyebrow">SOURCE DETAILS</span>

          <h2>{title}</h2>
        </div>

        <button
          type="button"
          className="knowledge-inspector-close"
          onClick={onClose}
          aria-label="Close source details"
        >
          ×
        </button>
      </div>

      <div className="knowledge-inspector-status-row">
        <span
          className={`knowledge-source-status status-${status
            .toString()
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "-")}`}
        >
          {formatLabel(status)}
        </span>

        <span className="knowledge-inspector-type">
          {formatLabel(sourceType)}
        </span>
      </div>

      <div className="knowledge-metadata-section">
        <h3>Source metadata</h3>

        <dl className="knowledge-metadata-list">
          <div>
            <dt>Source ID</dt>
            <dd>{source?.id || "—"}</dd>
          </div>

          <div>
            <dt>Project ID</dt>
            <dd>{source?.project_id || "—"}</dd>
          </div>

          <div>
            <dt>Filename</dt>
            <dd>{source?.filename || source?.original_filename || "—"}</dd>
          </div>

          <div>
            <dt>MIME type</dt>
            <dd>{source?.mime_type || "—"}</dd>
          </div>

          <div>
            <dt>Created</dt>
            <dd>{formatDate(source?.created_at)}</dd>
          </div>

          <div>
            <dt>Updated</dt>
            <dd>{formatDate(source?.updated_at)}</dd>
          </div>
        </dl>
      </div>

      {Object.keys(metadata).length > 0 && (
        <div className="knowledge-metadata-section">
          <h3>Processing metadata</h3>

          <dl className="knowledge-metadata-list">
            {Object.entries(metadata).map(([key, value]) => (
              <div key={key}>
                <dt>{formatLabel(key)}</dt>

                <dd>
                  {typeof value === "object"
                    ? JSON.stringify(value)
                    : String(value)}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </aside>
  );
}

KnowledgeInspector.propTypes = {
  source: PropTypes.object,
  onClose: PropTypes.func.isRequired,
};

export default KnowledgeInspector;
