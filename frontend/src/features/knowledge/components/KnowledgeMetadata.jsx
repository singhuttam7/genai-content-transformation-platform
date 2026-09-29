import PropTypes from "prop-types";

function formatLabel(value) {
  if (!value) {
    return "—";
  }

  return value
    .toString()
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  if (typeof value === "object") {
    return JSON.stringify(value);
  }

  return String(value);
}

function KnowledgeMetadata({ source }) {
  if (!source) {
    return null;
  }

  const metadata =
    source?.metadata && typeof source.metadata === "object"
      ? source.metadata
      : {};

  const fields = [
    {
      label: "Source ID",
      value: source?.id,
    },
    {
      label: "Project ID",
      value: source?.project_id,
    },
    {
      label: "Source type",
      value: source?.input_type || source?.inputType,
    },
    {
      label: "Filename",
      value: source?.filename || source?.original_filename,
    },
    {
      label: "MIME type",
      value: source?.mime_type,
    },
    {
      label: "Status",
      value: source?.status,
    },
    {
      label: "Storage key",
      value: source?.storage_key,
    },
    {
      label: "Content hash",
      value: source?.content_hash,
    },
  ];

  return (
    <section className="knowledge-metadata-card">
      <div className="knowledge-subsection-header">
        <div>
          <span className="eyebrow">METADATA</span>

          <h3>Source information</h3>
        </div>
      </div>

      <dl className="knowledge-metadata-list">
        {fields.map((field) => (
          <div key={field.label}>
            <dt>{field.label}</dt>
            <dd>{formatValue(field.value)}</dd>
          </div>
        ))}
      </dl>

      {Object.keys(metadata).length > 0 && (
        <div className="knowledge-custom-metadata">
          <h4>Additional metadata</h4>

          <dl className="knowledge-metadata-list">
            {Object.entries(metadata).map(([key, value]) => (
              <div key={key}>
                <dt>{formatLabel(key)}</dt>
                <dd>{formatValue(value)}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </section>
  );
}

KnowledgeMetadata.propTypes = {
  source: PropTypes.object,
};

export default KnowledgeMetadata;
