import PropTypes from "prop-types";

function ExecutionEmptyState({ mode = "library" }) {
  const isLibrary = mode === "library";

  const content = isLibrary
    ? {
        icon: "◆",
        title: "No executions yet",
        description:
          "Start a transformation from the Transform workspace and its execution will appear here.",
      }
    : {
        icon: "◇",
        title: "Select an execution",
        description:
          "Choose a run from the list to inspect its workflow, status, result, and execution metadata.",
      };

  return (
    <div className={["executions-empty", `executions-empty-${mode}`].join(" ")}>
      <div className="executions-empty-icon" aria-hidden="true">
        {content.icon}
      </div>

      <strong>{content.title}</strong>

      <p>{content.description}</p>
    </div>
  );
}

ExecutionEmptyState.propTypes = {
  mode: PropTypes.oneOf(["library", "detail"]),
};

export default ExecutionEmptyState;
