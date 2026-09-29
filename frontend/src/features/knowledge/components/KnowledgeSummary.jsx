import PropTypes from "prop-types";

function getSourceTypeLabel(source) {
  const inputType = source?.input_type || source?.inputType || "unknown";

  return inputType
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function KnowledgeSummary({ sources = [] }) {
  const totalSources = sources.length;

  const indexedSources = sources.filter((source) => {
    const status =
      source?.status || source?.processing_status || source?.processingStatus;

    return (
      typeof status === "string" &&
      ["indexed", "completed", "ready"].includes(status.toLowerCase())
    );
  }).length;

  const sourceTypes = new Set(
    sources.map((source) => getSourceTypeLabel(source)),
  );

  const summaryItems = [
    {
      label: "Knowledge sources",
      value: totalSources,
      description: "Source records available to the knowledge layer.",
      icon: "▣",
    },
    {
      label: "Indexed sources",
      value: indexedSources,
      description: "Sources currently reported as ready or indexed.",
      icon: "✓",
    },
    {
      label: "Source types",
      value: sourceTypes.size,
      description: "Different source formats currently represented.",
      icon: "◇",
    },
  ];

  return (
    <section className="knowledge-summary-grid" aria-label="Knowledge summary">
      {summaryItems.map((item) => (
        <article className="knowledge-summary-card" key={item.label}>
          <div className="knowledge-summary-card-top">
            <span className="knowledge-summary-icon">{item.icon}</span>

            <span className="knowledge-summary-label">{item.label}</span>
          </div>

          <strong className="knowledge-summary-value">{item.value}</strong>

          <p className="knowledge-summary-description">{item.description}</p>
        </article>
      ))}
    </section>
  );
}

KnowledgeSummary.propTypes = {
  sources: PropTypes.arrayOf(PropTypes.object),
};

export default KnowledgeSummary;
