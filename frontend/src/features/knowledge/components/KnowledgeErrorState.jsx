import PropTypes from "prop-types";

function KnowledgeErrorState({ message, onRetry }) {
  return (
    <section className="knowledge-error-state">
      <div className="knowledge-error-icon" aria-hidden="true">
        !
      </div>

      <div className="knowledge-error-content">
        <span className="eyebrow">KNOWLEDGE BASE</span>

        <h2>Unable to load knowledge</h2>

        <p>
          {message ||
            "The knowledge sources could not be loaded. Please try again."}
        </p>

        <button
          type="button"
          className="secondary-action-button"
          onClick={onRetry}
        >
          Try again
        </button>
      </div>
    </section>
  );
}

KnowledgeErrorState.propTypes = {
  message: PropTypes.string,
  onRetry: PropTypes.func.isRequired,
};

export default KnowledgeErrorState;
