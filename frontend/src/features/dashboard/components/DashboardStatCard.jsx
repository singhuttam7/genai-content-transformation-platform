function DashboardStatCard({
  label,
  value,
  description,
  icon,
  accent = "default",
}) {
  return (
    <article
      className={["dashboard-stat-card", `dashboard-stat-card-${accent}`].join(
        " ",
      )}
    >
      <div className="dashboard-stat-card-top">
        <div className="dashboard-stat-card-icon" aria-hidden="true">
          {icon}
        </div>

        <span className="dashboard-stat-card-label">{label}</span>
      </div>

      <div className="dashboard-stat-card-value">{value}</div>

      {description && (
        <div className="dashboard-stat-card-description">{description}</div>
      )}
    </article>
  );
}

export default DashboardStatCard;
