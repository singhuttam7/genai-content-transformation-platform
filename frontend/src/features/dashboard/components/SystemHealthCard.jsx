function SystemHealthCard({ health, healthState }) {
  const backendHealthy = healthState?.backendHealthy ?? false;
  const databaseHealthy = healthState?.databaseHealthy ?? false;
  const overallHealthy = healthState?.healthy ?? false;

  const overallLabel = overallHealthy ? "Healthy" : "Attention required";

  return (
    <article className="dashboard-health-card">
      <div className="dashboard-section-heading">
        <div>
          <span className="dashboard-section-kicker">Platform status</span>

          <h2>System health</h2>
        </div>

        <span
          className={[
            "dashboard-health-badge",
            overallHealthy
              ? "dashboard-health-badge-healthy"
              : "dashboard-health-badge-warning",
          ].join(" ")}
        >
          <span className="dashboard-health-badge-dot" aria-hidden="true" />
          {overallLabel}
        </span>
      </div>

      <div className="dashboard-health-services">
        <div className="dashboard-health-service">
          <div className="dashboard-health-service-icon" aria-hidden="true">
            ◈
          </div>

          <div className="dashboard-health-service-content">
            <strong>API service</strong>

            <span>
              {health?.service || "GenAI Content Transformation Platform"}
            </span>
          </div>

          <span
            className={[
              "dashboard-service-status",
              backendHealthy
                ? "dashboard-service-status-healthy"
                : "dashboard-service-status-warning",
            ].join(" ")}
          >
            {backendHealthy ? "Operational" : "Unavailable"}
          </span>
        </div>

        <div className="dashboard-health-service">
          <div className="dashboard-health-service-icon" aria-hidden="true">
            ◇
          </div>

          <div className="dashboard-health-service-content">
            <strong>Database</strong>

            <span>PostgreSQL persistence layer</span>
          </div>

          <span
            className={[
              "dashboard-service-status",
              databaseHealthy
                ? "dashboard-service-status-healthy"
                : "dashboard-service-status-warning",
            ].join(" ")}
          >
            {databaseHealthy ? "Healthy" : "Unavailable"}
          </span>
        </div>
      </div>
    </article>
  );
}

export default SystemHealthCard;
