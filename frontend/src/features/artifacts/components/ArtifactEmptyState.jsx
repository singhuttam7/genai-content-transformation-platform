import PropTypes from "prop-types";

function ArtifactEmptyState({ mode = "library" }) {
  const isLibrary = mode === "library";

  const content = isLibrary
    ? {
        icon: "◆",
        title: "No artifacts yet",
        description:
          "Generated outputs will appear here after a transformation execution produces artifacts.",
      }
    : {
        icon: "◇",
        title: "Select an artifact",
        description:
          "Choose an artifact from the library to inspect its content, metadata, and execution context.",
      };

  return (
    <div className={["artifacts-empty", `artifacts-empty-${mode}`].join(" ")}>
      <div className="artifacts-empty-icon" aria-hidden="true">
        {content.icon}
      </div>

      <strong>{content.title}</strong>

      <p>{content.description}</p>
    </div>
  );
}

ArtifactEmptyState.propTypes = {
  mode: PropTypes.oneOf(["library", "detail"]),
};

export default ArtifactEmptyState;
