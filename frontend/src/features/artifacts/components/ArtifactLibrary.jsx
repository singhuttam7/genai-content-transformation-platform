import PropTypes from "prop-types";

import ArtifactEmptyState from "./ArtifactEmptyState";
import ArtifactListItem from "./ArtifactListItem";

function ArtifactLibrary({
  artifacts = [],
  selectedArtifactId = null,
  loading = false,
  onSelect,
}) {
  return (
    <section className="artifacts-card artifacts-library-card">
      <div className="artifacts-card-header">
        <div>
          <span className="transformation-card-kicker">OUTPUT LIBRARY</span>

          <h2>Generated artifacts</h2>

          <p>Select an artifact to inspect its output and execution context.</p>
        </div>

        <span className="artifacts-count">
          {artifacts.length} {artifacts.length === 1 ? "artifact" : "artifacts"}
        </span>
      </div>

      {loading ? (
        <div className="artifacts-loading-state">
          <div className="artifacts-loading-spinner" aria-hidden="true" />

          <div>
            <strong>Loading artifacts</strong>

            <span>Retrieving generated content...</span>
          </div>
        </div>
      ) : artifacts.length === 0 ? (
        <ArtifactEmptyState mode="library" />
      ) : (
        <div className="artifact-list">
          {artifacts.map((artifact) => (
            <ArtifactListItem
              key={artifact.id}
              artifact={artifact}
              selected={selectedArtifactId === artifact.id}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </section>
  );
}

ArtifactLibrary.propTypes = {
  artifacts: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string,
      title: PropTypes.string,
      artifact_type: PropTypes.string,
      status: PropTypes.string,
      execution_id: PropTypes.string,
      created_at: PropTypes.string,
    }),
  ),
  selectedArtifactId: PropTypes.string,
  loading: PropTypes.bool,
  onSelect: PropTypes.func.isRequired,
};

export default ArtifactLibrary;
