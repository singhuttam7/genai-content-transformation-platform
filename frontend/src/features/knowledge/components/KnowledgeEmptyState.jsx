import PropTypes from "prop-types";

function KnowledgeEmptyState({ onCreateSource }) {
  return (
    <section className="knowledge-empty-state">
      <div className="knowledge-empty-state-icon" aria-hidden="true">
        ◇
      </div>

      <div className="knowledge-empty-state-content">
        <span className="eyebrow">KNOWLEDGE BASE</span>

        <h2>Your knowledge library is empty</h2>

        <p>
          Add source material through the transformation workflow to make it
          available for retrieval and downstream content generation.
        </p>

        {onCreateSource && (
          <button
            type="button"
            className="primary-action-button"
            onClick={onCreateSource}
          >
            Add knowledge source
          </button>
        )}
      </div>
    </section>
  );
}

KnowledgeEmptyState.propTypes = {
  onCreateSource: PropTypes.func,
};

export default KnowledgeEmptyState;
