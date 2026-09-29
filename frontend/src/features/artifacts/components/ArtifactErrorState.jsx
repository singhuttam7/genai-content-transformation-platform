import PropTypes from "prop-types";

function ArtifactErrorState({ message, onRetry, retrying = false }) {
  return (
    <div className="artifacts-error">
      <div className="artifacts-error-icon" aria-hidden="true">
        !
      </div>

      <div className="artifacts-error-content">
        <strong>Unable to load artifact data</strong>

        <span>
          {message || "An unexpected error occurred while loading artifacts."}
        </span>

        {onRetry && (
          <button
            type="button"
            className="secondary-action-button artifacts-error-retry"
            onClick={onRetry}
            disabled={retrying}
          >
            {retrying ? "Retrying..." : "Try again"}
          </button>
        )}
      </div>
    </div>
  );
}

ArtifactErrorState.propTypes = {
  message: PropTypes.string,
  onRetry: PropTypes.func,
  retrying: PropTypes.bool,
};

export default ArtifactErrorState;
