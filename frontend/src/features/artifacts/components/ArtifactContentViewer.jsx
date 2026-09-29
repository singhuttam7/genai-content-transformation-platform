import PropTypes from "prop-types";

function formatJson(value) {
  if (value === null || value === undefined) {
    return "—";
  }

  if (typeof value === "string") {
    return value;
  }

  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function ArtifactContentViewer({ content, metadata }) {
  return (
    <>
      <section className="artifact-detail-section">
        <div className="artifact-detail-section-heading">
          <span className="artifact-detail-section-title">
            Generated content
          </span>
        </div>

        <pre className="artifact-detail-json artifact-detail-content-block">
          {formatJson(content)}
        </pre>
      </section>

      <section className="artifact-detail-section">
        <div className="artifact-detail-section-heading">
          <span className="artifact-detail-section-title">Metadata</span>
        </div>

        <pre className="artifact-detail-json">{formatJson(metadata)}</pre>
      </section>
    </>
  );
}

ArtifactContentViewer.propTypes = {
  content: PropTypes.any,
  metadata: PropTypes.any,
};

export default ArtifactContentViewer;
