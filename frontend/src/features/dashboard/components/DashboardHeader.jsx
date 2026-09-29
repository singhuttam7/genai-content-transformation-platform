import { useNavigate } from "react-router-dom";

function DashboardHeader({ onRefresh, refreshing = false }) {
  const navigate = useNavigate();

  return (
    <header className="dashboard-header">
      <div className="dashboard-header-content">
        <div>
          <span className="eyebrow">Workspace overview</span>

          <h1>Content Transformation Workspace</h1>

          <p>
            Monitor your transformation activity, generated artifacts, and
            platform status from one place.
          </p>
        </div>

        <div className="dashboard-header-actions">
          <button
            type="button"
            className="secondary-action-button"
            onClick={onRefresh}
            disabled={refreshing}
          >
            <span aria-hidden="true">{refreshing ? "↻" : "⟳"}</span>

            {refreshing ? "Refreshing..." : "Refresh"}
          </button>

          <button
            type="button"
            className="primary-action-button dashboard-create-button"
            onClick={() => navigate("/transform")}
          >
            <span aria-hidden="true">✦</span>
            New Transformation
          </button>
        </div>
      </div>
    </header>
  );
}

export default DashboardHeader;
