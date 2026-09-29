import PropTypes from "prop-types";

function KnowledgeChunkViewer({ source }) {
  if (!source) {
    return (
      <section className="knowledge-chunk-card knowledge-chunk-empty">
        <div className="knowledge-content-placeholder">
          <span className="knowledge-content-icon" aria-hidden="true">
            ▤
          </span>

          <h3>Knowledge content</h3>

          <p>Select a source to inspect its knowledge content.</p>
        </div>
      </section>
    );
  }

  const canonicalText =
    source?.canonical_text || source?.canonicalText || source?.content;

  return (
    <section className="knowledge-chunk-card">
      <div className="knowledge-subsection-header">
        <div>
          <span className="eyebrow">KNOWLEDGE CONTENT</span>

          <h3>Source content</h3>

          <p>Content available directly from the source record.</p>
        </div>
      </div>

      {canonicalText ? (
        <div className="knowledge-content-viewer">
          <pre>{canonicalText}</pre>
        </div>
      ) : (
        <div className="knowledge-content-placeholder">
          <span className="knowledge-content-icon" aria-hidden="true">
            ▤
          </span>

          <h4>Chunk-level content is not available here yet</h4>

          <p>
            The current frontend source API does not expose persisted knowledge
            chunks. The knowledge data model supports chunked content, but this
            page will only display chunk details once a dedicated read endpoint
            is available.
          </p>
        </div>
      )}
    </section>
  );
}

KnowledgeChunkViewer.propTypes = {
  source: PropTypes.object,
};

export default KnowledgeChunkViewer;
