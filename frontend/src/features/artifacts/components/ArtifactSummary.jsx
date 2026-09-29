import PropTypes from "prop-types";

function SummaryCard({ icon, label, value, variant = "default" }) {
  return (
    <article
      className={[
        "artifacts-summary-card",
        `artifacts-summary-card-${variant}`,
      ].join(" ")}
    >
      <div className="artifacts-summary-icon">{icon}</div>

      <div className="artifacts-summary-content">
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </article>
  );
}

function ArtifactSummary({
  total = 0,
  completed = 0,
  failed = 0,
  selected = false,
}) {
  return (
    <section className="artifacts-summary" aria-label="Artifact overview">
      <SummaryCard
        icon="◆"
        label="Total artifacts"
        value={total}
        variant="primary"
      />

      <SummaryCard
        icon="✓"
        label="Generated successfully"
        value={completed}
        variant="success"
      />

      <SummaryCard
        icon="!"
        label="Requires attention"
        value={failed}
        variant="warning"
      />

      <SummaryCard
        icon="◇"
        label="Selected artifact"
        value={selected ? 1 : "—"}
        variant="neutral"
      />
    </section>
  );
}

SummaryCard.propTypes = {
  icon: PropTypes.string.isRequired,
  label: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  variant: PropTypes.string,
};

ArtifactSummary.propTypes = {
  total: PropTypes.number,
  completed: PropTypes.number,
  failed: PropTypes.number,
  selected: PropTypes.bool,
};

export default ArtifactSummary;
