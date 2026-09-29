import PropTypes from "prop-types";

function SummaryCard({ icon, label, value, variant = "default" }) {
  return (
    <article
      className={[
        "executions-summary-card",
        `executions-summary-card-${variant}`,
      ].join(" ")}
    >
      <div className="executions-summary-icon">{icon}</div>

      <div className="executions-summary-content">
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </article>
  );
}

function ExecutionSummary({
  total = 0,
  completed = 0,
  running = 0,
  failed = 0,
}) {
  return (
    <section className="executions-summary" aria-label="Execution overview">
      <SummaryCard
        icon="◆"
        label="Total executions"
        value={total}
        variant="primary"
      />

      <SummaryCard
        icon="✓"
        label="Completed"
        value={completed}
        variant="success"
      />

      <SummaryCard
        icon="◌"
        label="In progress"
        value={running}
        variant="active"
      />

      <SummaryCard icon="!" label="Failed" value={failed} variant="warning" />
    </section>
  );
}

SummaryCard.propTypes = {
  icon: PropTypes.string.isRequired,
  label: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  variant: PropTypes.string,
};

ExecutionSummary.propTypes = {
  total: PropTypes.number,
  completed: PropTypes.number,
  running: PropTypes.number,
  failed: PropTypes.number,
};

export default ExecutionSummary;
