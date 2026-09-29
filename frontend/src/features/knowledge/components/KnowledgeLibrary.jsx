import PropTypes from "prop-types";

import KnowledgeListItem from "./KnowledgeListItem";

function KnowledgeLibrary({
  sources = [],
  selectedSourceId = null,
  loading = false,
  onSelect,
}) {
  if (loading) {
    return (
      <section className="knowledge-library-card">
        <div className="knowledge-section-header">
          <div>
            <span className="eyebrow">SOURCE LIBRARY</span>

            <h2>Knowledge sources</h2>

            <p>Loading the source material available to the knowledge layer.</p>
          </div>
        </div>

        <div className="knowledge-library-loading">
          <span className="knowledge-loading-spinner" />
          <span>Loading knowledge sources...</span>
        </div>
      </section>
    );
  }

  return (
    <section className="knowledge-library-card">
      <div className="knowledge-section-header">
        <div>
          <span className="eyebrow">SOURCE LIBRARY</span>

          <h2>Knowledge sources</h2>

          <p>
            Browse source material currently available for retrieval and
            transformation workflows.
          </p>
        </div>

        <span className="knowledge-source-count">
          {sources.length} {sources.length === 1 ? "source" : "sources"}
        </span>
      </div>

      {sources.length === 0 ? (
        <div className="knowledge-library-empty">
          <div className="knowledge-empty-icon">◇</div>

          <h3>No knowledge sources yet</h3>

          <p>
            Sources added through the transformation workflow will appear here.
          </p>
        </div>
      ) : (
        <div className="knowledge-source-list">
          {sources.map((source) => (
            <KnowledgeListItem
              key={source.id}
              source={source}
              selected={source.id === selectedSourceId}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </section>
  );
}

KnowledgeLibrary.propTypes = {
  sources: PropTypes.arrayOf(PropTypes.object),
  selectedSourceId: PropTypes.string,
  loading: PropTypes.bool,
  onSelect: PropTypes.func.isRequired,
};

export default KnowledgeLibrary;
