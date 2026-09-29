import PropTypes from "prop-types";

function ExecutionErrorState({ message, onRetry, retrying = false }) {
  return (
    <div className="executions-error">
      <div className="executions-error-icon" aria-hidden="true">
        !
      </div>

      <div className="executions-error-content">
        <strong>Unable to load execution data</strong>

        <span>
          {message || "An unexpected error occurred while loading executions."}
        </span>

        {onRetry && (
          <button
            type="button"
            className="secondary-action-button executions-error-retry"
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

ExecutionErrorState.propTypes = {
  message: PropTypes.string,
  onRetry: PropTypes.func,
  retrying: PropTypes.bool,
};

export default ExecutionErrorState;
