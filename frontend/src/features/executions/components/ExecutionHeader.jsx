import PropTypes from "prop-types";

function ExecutionHeader({ onRefresh, refreshing = false }) {
  return (
    <header className="executions-page-header">
      <div className="executions-page-heading">
        <span className="eyebrow">EXECUTION CENTER</span>

        <h1>Transformation executions</h1>

        <p>
          Monitor transformation runs, inspect workflow execution details, and
          review the current execution state.
        </p>
      </div>

      <div className="executions-page-actions">
        <button
          type="button"
          className="secondary-action-button"
          onClick={onRefresh}
          disabled={refreshing}
        >
          <span aria-hidden="true">{refreshing ? "↻" : "⟳"}</span>

          {refreshing ? "Refreshing..." : "Refresh executions"}
        </button>
      </div>
    </header>
  );
}

ExecutionHeader.propTypes = {
  onRefresh: PropTypes.func.isRequired,
  refreshing: PropTypes.bool,
};

export default ExecutionHeader;
