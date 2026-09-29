import PropTypes from "prop-types";

function formatArtifactType(value) {
  if (!value) {
    return "—";
  }

  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

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
    <div className="artifact-detail-row">
      <span>{label}</span>

      <strong title={value || undefined}>{value || "—"}</strong>
    </div>
  );
}

MetadataRow.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.string,
};

function ArtifactMetadata({ artifact }) {
  if (!artifact) {
    return null;
  }

  return (
    <section className="artifact-detail-section">
      <div className="artifact-detail-section-heading">
        <span className="artifact-detail-section-title">Execution context</span>
      </div>

      <div className="artifact-detail-grid">
        <MetadataRow label="Artifact ID" value={artifact.id} />

        <MetadataRow
          label="Artifact type"
          value={formatArtifactType(artifact.artifact_type)}
        />

        <MetadataRow label="Execution ID" value={artifact.execution_id} />

        <MetadataRow
          label="Transformation ID"
          value={artifact.transformation_id}
        />

        <MetadataRow label="Created" value={formatDate(artifact.created_at)} />

        {artifact.storage_uri && (
          <MetadataRow label="Storage URI" value={artifact.storage_uri} />
        )}
      </div>
    </section>
  );
}

ArtifactMetadata.propTypes = {
  artifact: PropTypes.shape({
    id: PropTypes.string,
    artifact_type: PropTypes.string,
    execution_id: PropTypes.string,
    transformation_id: PropTypes.string,
    created_at: PropTypes.string,
    storage_uri: PropTypes.string,
  }),
};

export default ArtifactMetadata;
