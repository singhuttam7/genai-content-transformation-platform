function ActivityOverview({ statusCounts }) {
  const total = statusCounts?.total ?? 0;
  const completed = statusCounts?.completed ?? 0;
  const running = statusCounts?.running ?? 0;
  const pending = statusCounts?.pending ?? 0;
  const failed = statusCounts?.failed ?? 0;

  const completedPercentage =
    total > 0 ? Math.round((completed / total) * 100) : 0;

  const runningPercentage = total > 0 ? Math.round((running / total) * 100) : 0;

  const pendingPercentage = total > 0 ? Math.round((pending / total) * 100) : 0;

  const failedPercentage = total > 0 ? Math.round((failed / total) * 100) : 0;

  return (
    <section className="dashboard-activity-card">
      <div className="dashboard-section-heading">
        <div>
          <span className="dashboard-section-kicker">Execution overview</span>

          <h2>Transformation activity</h2>
        </div>

        <span className="dashboard-activity-total">{total} total</span>
      </div>

      {total === 0 ? (
        <div className="dashboard-activity-empty">
          <span aria-hidden="true">◌</span>

          <div>
            <strong>No execution activity yet</strong>

            <p>
              Execution statistics will appear here after your first
              transformation.
            </p>
          </div>
        </div>
      ) : (
        <>
          <div className="dashboard-activity-summary">
            <div className="dashboard-activity-summary-value">
              {completedPercentage}%
            </div>

            <div>
              <strong>Completion rate</strong>

              <span>
                {completed} of {total} executions completed
              </span>
            </div>
          </div>

          <div className="dashboard-activity-bar">
            <span
              className="dashboard-activity-bar-completed"
              style={{
                width: `${completedPercentage}%`,
              }}
              title={`Completed: ${completed}`}
            />

            <span
              className="dashboard-activity-bar-running"
              style={{
                width: `${runningPercentage}%`,
              }}
              title={`Running: ${running}`}
            />

            <span
              className="dashboard-activity-bar-pending"
              style={{
                width: `${pendingPercentage}%`,
              }}
              title={`Pending: ${pending}`}
            />

            <span
              className="dashboard-activity-bar-failed"
              style={{
                width: `${failedPercentage}%`,
              }}
              title={`Failed: ${failed}`}
            />
          </div>

          <div className="dashboard-activity-legend">
            <ActivityLegendItem
              label="Completed"
              value={completed}
              className="completed"
            />

            <ActivityLegendItem
              label="Running"
              value={running}
              className="running"
            />

            <ActivityLegendItem
              label="Pending"
              value={pending}
              className="pending"
            />

            <ActivityLegendItem
              label="Failed"
              value={failed}
              className="failed"
            />
          </div>
        </>
      )}
    </section>
  );
}

function ActivityLegendItem({ label, value, className }) {
  return (
    <div className="dashboard-activity-legend-item">
      <span
        className={[
          "dashboard-activity-legend-dot",
          `dashboard-activity-legend-dot-${className}`,
        ].join(" ")}
        aria-hidden="true"
      />

      <span className="dashboard-activity-legend-label">{label}</span>

      <strong>{value}</strong>
    </div>
  );
}

export default ActivityOverview;
