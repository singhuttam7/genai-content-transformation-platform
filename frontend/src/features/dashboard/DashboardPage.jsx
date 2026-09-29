import DashboardHeader from "./components/DashboardHeader";
import DashboardStatCard from "./components/DashboardStatCard";
import SystemHealthCard from "./components/SystemHealthCard";
import QuickActions from "./components/QuickActions";
import RecentExecutions from "./components/RecentExecutions";
import RecentArtifacts from "./components/RecentArtifacts";
import ActivityOverview from "./components/ActivityOverview";

import { useDashboardData } from "./hooks/useDashboardData";

function DashboardPage() {
  const { data, loading, refreshing, error, refresh } = useDashboardData();

  if (loading && !data) {
    return (
      <div className="dashboard-page">
        <div className="dashboard-loading">
          <div className="dashboard-loading-spinner" />

          <div>
            <strong>Loading workspace</strong>

            <p>Gathering transformation activity and platform status...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="dashboard-page">
        <DashboardHeader onRefresh={refresh} refreshing={refreshing} />

        <div className="dashboard-error-state">
          <div className="dashboard-error-icon" aria-hidden="true">
            !
          </div>

          <div>
            <strong>Dashboard data could not be loaded</strong>

            <p>{error}</p>

            <button
              type="button"
              className="primary-action-button dashboard-error-action"
              onClick={refresh}
              disabled={refreshing}
            >
              {refreshing ? "Retrying..." : "Try again"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  const health = data?.health;
  const healthState = data?.healthState;

  const executionStats = data?.executions?.statusCounts ?? {};

  const artifactTotal = data?.artifacts?.total ?? 0;

  const executionTotal = data?.executions?.total ?? 0;

  return (
    <div className="dashboard-page">
      <DashboardHeader onRefresh={refresh} refreshing={refreshing} />

      {data?.partialFailure && (
        <div className="dashboard-partial-warning">
          <span className="dashboard-partial-warning-icon" aria-hidden="true">
            !
          </span>

          <div>
            <strong>Some dashboard data could not be loaded.</strong>

            <span>
              Available sections are still shown with the information that was
              retrieved successfully.
            </span>
          </div>
        </div>
      )}

      {/* ==========================================================
          Primary Statistics
          ========================================================== */}

      <section className="dashboard-stat-grid">
        <DashboardStatCard
          label="Executions"
          value={executionTotal}
          description="Total workflow executions"
          icon="▶"
          accent="primary"
        />

        <DashboardStatCard
          label="Completed"
          value={executionStats.completed ?? 0}
          description="Successfully completed"
          icon="✓"
          accent="success"
        />

        <DashboardStatCard
          label="Artifacts"
          value={artifactTotal}
          description="Generated content artifacts"
          icon="□"
          accent="purple"
        />

        <DashboardStatCard
          label="Platform"
          value={healthState?.healthy ? "Healthy" : "Attention"}
          description={
            healthState?.healthy
              ? "Core services operational"
              : "Check platform status"
          }
          icon="◈"
          accent={healthState?.healthy ? "success" : "warning"}
        />
      </section>

      {/* ==========================================================
          Main Dashboard Grid
          ========================================================== */}

      <section className="dashboard-primary-grid">
        <SystemHealthCard health={health} healthState={healthState} />

        <QuickActions />
      </section>

      <section className="dashboard-secondary-grid">
        <ActivityOverview statusCounts={executionStats} />

        <RecentArtifacts artifacts={data?.recentArtifacts ?? []} />
      </section>

      <section className="dashboard-full-width">
        <RecentExecutions executions={data?.recentExecutions ?? []} />
      </section>
    </div>
  );
}

export default DashboardPage;
