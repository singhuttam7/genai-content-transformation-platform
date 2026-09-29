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

function formatArtifactType(value) {
  if (!value) {
    return "Unknown type";
  }

  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function getArtifactIcon(type) {
  const icons = {
    advisory: "A",
    executive_summary: "E",
    social_media: "S",
    infographic: "I",
    presentation: "P",
    video: "V",
  };

  return icons[type] || "◆";
}

function getStatusClass(status) {
  if (!status) {
    return "artifact-status";
  }

  return `artifact-status artifact-status-${String(status).toLowerCase()}`;
}

function getStatusLabel(status) {
  if (!status) {
    return "Unknown";
  }

  return String(status)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function ArtifactListItem({ artifact, selected = false, onSelect }) {
  const artifactType = artifact?.artifact_type;

  const artifactTitle =
    artifact?.title || formatArtifactType(artifactType) || "Unnamed artifact";

  return (
    <button
      type="button"
      className={["artifact-list-item", selected ? "selected" : ""].join(" ")}
      onClick={() => onSelect(artifact?.id)}
      aria-pressed={selected}
    >
      <span className="artifact-list-icon" aria-hidden="true">
        {getArtifactIcon(artifactType)}
      </span>

      <span className="artifact-list-main">
        <strong>{artifactTitle}</strong>

        <span className="artifact-list-type">
          {formatArtifactType(artifactType)}
        </span>

        <span className="artifact-list-date">
          Created {formatDate(artifact?.created_at)}
        </span>
      </span>

      <span className="artifact-list-meta">
        <span className={getStatusClass(artifact?.status)}>
          {getStatusLabel(artifact?.status)}
        </span>

        <span className="artifact-list-execution">
          {artifact?.execution_id
            ? `Execution ${String(artifact.execution_id).slice(0, 8)}…`
            : "Execution unavailable"}
        </span>
      </span>

      <span className="artifact-list-arrow" aria-hidden="true">
        →
      </span>
    </button>
  );
}

ArtifactListItem.propTypes = {
  artifact: PropTypes.shape({
    id: PropTypes.string,
    title: PropTypes.string,
    artifact_type: PropTypes.string,
    status: PropTypes.string,
    execution_id: PropTypes.string,
    created_at: PropTypes.string,
  }).isRequired,
  selected: PropTypes.bool,
  onSelect: PropTypes.func.isRequired,
};

export default ArtifactListItem;
