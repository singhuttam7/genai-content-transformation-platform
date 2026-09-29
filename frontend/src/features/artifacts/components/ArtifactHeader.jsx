import PropTypes from "prop-types";

function ArtifactHeader({ onRefresh, refreshing = false }) {
  return (
    <header className="artifacts-page-header">
      <div className="artifacts-page-heading">
        <span className="eyebrow">ARTIFACT CENTER</span>

        <h1>Generated artifacts</h1>

        <p>
          Review, inspect, and understand the content produced by your
          transformation workflows.
        </p>
      </div>

      <div className="artifacts-page-actions">
        <button
          type="button"
          className="secondary-action-button"
          onClick={onRefresh}
          disabled={refreshing}
        >
          <span aria-hidden="true">{refreshing ? "↻" : "⟳"}</span>

          {refreshing ? "Refreshing..." : "Refresh artifacts"}
        </button>
      </div>
    </header>
  );
}

ArtifactHeader.propTypes = {
  onRefresh: PropTypes.func.isRequired,
  refreshing: PropTypes.bool,
};

export default ArtifactHeader;
