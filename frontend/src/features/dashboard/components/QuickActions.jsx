import { useNavigate } from "react-router-dom";

const QUICK_ACTIONS = [
  {
    id: "text",
    icon: "T",
    title: "Start with text",
    description: "Paste source content and transform it.",
    sourceType: "text",
  },
  {
    id: "url",
    icon: "↗",
    title: "Transform a URL",
    description: "Fetch and transform web content.",
    sourceType: "url",
  },
  {
    id: "pdf",
    icon: "▤",
    title: "Upload a PDF",
    description: "Process a PDF document into outputs.",
    sourceType: "pdf",
  },
  {
    id: "prompt",
    icon: "✦",
    title: "Start with a prompt",
    description: "Use an instruction as your source.",
    sourceType: "prompt",
  },
];

function QuickActions() {
  const navigate = useNavigate();

  function handleAction(sourceType) {
    navigate("/transform", {
      state: {
        sourceType,
      },
    });
  }

  return (
    <section className="dashboard-quick-actions">
      <div className="dashboard-section-heading">
        <div>
          <span className="dashboard-section-kicker">Get started</span>

          <h2>Quick actions</h2>
        </div>
      </div>

      <div className="dashboard-quick-action-list">
        {QUICK_ACTIONS.map((action) => (
          <button
            key={action.id}
            type="button"
            className="dashboard-quick-action"
            onClick={() => handleAction(action.sourceType)}
          >
            <span className="dashboard-quick-action-icon" aria-hidden="true">
              {action.icon}
            </span>

            <span className="dashboard-quick-action-content">
              <strong>{action.title}</strong>

              <span>{action.description}</span>
            </span>

            <span className="dashboard-quick-action-arrow" aria-hidden="true">
              →
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}

export default QuickActions;
