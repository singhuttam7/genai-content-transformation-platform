import PropTypes from "prop-types";

import ArtifactContentViewer from "./ArtifactContentViewer";
import ArtifactEmptyState from "./ArtifactEmptyState";
import ArtifactMetadata from "./ArtifactMetadata";

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

function ArtifactInspector({ artifact, loading = false }) {
  return (
    <aside className="artifact-detail-card">
      <div className="artifacts-card-header">
        <div>
          <span className="transformation-card-kicker">INSPECTOR</span>

          <h2>Artifact details</h2>

          <p>Inspect the selected artifact and its execution context.</p>
        </div>
      </div>

      {loading ? (
        <div className="artifacts-loading-state">
          <div className="artifacts-loading-spinner" aria-hidden="true" />

          <div>
            <strong>Loading artifact</strong>

            <span>Retrieving artifact details...</span>
          </div>
        </div>
      ) : !artifact ? (
        <ArtifactEmptyState mode="detail" />
      ) : (
        <div className="artifact-detail-content">
          <div className="artifact-detail-identity">
            <div className="artifact-detail-identity-icon" aria-hidden="true">
              {getArtifactIcon(artifact.artifact_type)}
            </div>

            <div className="artifact-detail-identity-copy">
              <span>{formatArtifactType(artifact.artifact_type)}</span>

              <h3>
                {artifact.title || formatArtifactType(artifact.artifact_type)}
              </h3>
            </div>

            <span className={getStatusClass(artifact.status)}>
              {getStatusLabel(artifact.status)}
            </span>
          </div>

          <ArtifactMetadata artifact={artifact} />

          <ArtifactContentViewer
            content={artifact.content}
            metadata={artifact.metadata}
          />
        </div>
      )}
    </aside>
  );
}

ArtifactInspector.propTypes = {
  artifact: PropTypes.shape({
    id: PropTypes.string,
    title: PropTypes.string,
    artifact_type: PropTypes.string,
    status: PropTypes.string,
    execution_id: PropTypes.string,
    transformation_id: PropTypes.string,
    created_at: PropTypes.string,
    storage_uri: PropTypes.string,
    content: PropTypes.any,
    metadata: PropTypes.any,
  }),
  loading: PropTypes.bool,
};

export default ArtifactInspector;
