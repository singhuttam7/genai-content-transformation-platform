import PropTypes from "prop-types";

function formatSourceType(source) {
  const type = source?.input_type || source?.inputType || "source";

  return type
    .toString()
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatStatus(status) {
  if (!status) {
    return "Unknown";
  }

  return status
    .toString()
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function KnowledgeListItem({ source, selected = false, onSelect }) {
  const title =
    source?.title ||
    source?.filename ||
    source?.original_filename ||
    "Untitled source";

  const sourceType = formatSourceType(source);

  const status =
    source?.status ||
    source?.processing_status ||
    source?.processingStatus ||
    "unknown";

  const sourceId = source?.id;

  return (
    <button
      type="button"
      className={`knowledge-source-row${selected ? " is-selected" : ""}`}
      onClick={() => onSelect(sourceId)}
      aria-pressed={selected}
    >
      <div className="knowledge-source-row-main">
        <div className="knowledge-source-type">
          {sourceType.charAt(0).toUpperCase()}
        </div>

        <div className="knowledge-source-info">
          <strong>{title}</strong>

          <span>{sourceType}</span>
        </div>
      </div>

      <div className="knowledge-source-row-meta">
        <span
          className={`knowledge-source-status status-${status
            .toString()
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "-")}`}
        >
          {formatStatus(status)}
        </span>

        <span className="knowledge-source-chevron" aria-hidden="true">
          →
        </span>
      </div>
    </button>
  );
}

KnowledgeListItem.propTypes = {
  source: PropTypes.object.isRequired,
  selected: PropTypes.bool,
  onSelect: PropTypes.func.isRequired,
};

export default KnowledgeListItem;
