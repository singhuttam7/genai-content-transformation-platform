import PropTypes from "prop-types";

import ExecutionEmptyState from "./ExecutionEmptyState";
import ExecutionListItem from "./ExecutionListItem";

function ExecutionLibrary({
  executions = [],
  selectedExecutionId = null,
  loading = false,
  onSelect,
}) {
  return (
    <section className="executions-card executions-library-card">
      <div className="executions-card-header">
        <div>
          <span className="transformation-card-kicker">EXECUTION LIBRARY</span>

          <h2>Recent executions</h2>

          <p>
            Select a run to inspect its workflow, status, result, and execution
            context.
          </p>
        </div>

        <span className="executions-count">
          {executions.length}{" "}
          {executions.length === 1 ? "execution" : "executions"}
        </span>
      </div>

      {loading ? (
        <div className="executions-loading-state">
          <div className="executions-loading-spinner" aria-hidden="true" />

          <div>
            <strong>Loading executions</strong>

            <span>Retrieving transformation runs...</span>
          </div>
        </div>
      ) : executions.length === 0 ? (
        <ExecutionEmptyState mode="library" />
      ) : (
        <div className="execution-list">
          {executions.map((execution) => (
            <ExecutionListItem
              key={execution.id}
              execution={execution}
              selected={selectedExecutionId === execution.id}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </section>
  );
}

ExecutionLibrary.propTypes = {
  executions: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string,
      status: PropTypes.string,
      created_at: PropTypes.string,
      workflow_id: PropTypes.string,
      transformation_type: PropTypes.string,
      execution_context: PropTypes.shape({
        transformation_type: PropTypes.string,
      }),
    }),
  ),
  selectedExecutionId: PropTypes.string,
  loading: PropTypes.bool,
  onSelect: PropTypes.func.isRequired,
};

export default ExecutionLibrary;
