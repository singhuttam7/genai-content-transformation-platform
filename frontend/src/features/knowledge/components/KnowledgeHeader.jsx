import PropTypes from "prop-types";

function KnowledgeHeader({ onRefresh, refreshing = false }) {
  return (
    <header className="knowledge-page-header">
      <div className="knowledge-page-heading">
        <span className="eyebrow">KNOWLEDGE BASE</span>

        <h1>Knowledge library</h1>

        <p>
          Manage the source material and indexed knowledge that supports
          retrieval-powered transformation workflows.
        </p>
      </div>

      <div className="knowledge-page-actions">
        <button
          type="button"
          className="secondary-action-button"
          onClick={onRefresh}
          disabled={refreshing}
        >
          <span aria-hidden="true">{refreshing ? "↻" : "⟳"}</span>

          {refreshing ? "Refreshing..." : "Refresh knowledge"}
        </button>
      </div>
    </header>
  );
}

KnowledgeHeader.propTypes = {
  onRefresh: PropTypes.func.isRequired,
  refreshing: PropTypes.bool,
};

export default KnowledgeHeader;
