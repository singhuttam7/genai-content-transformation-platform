import { getHealth } from "../../../services/api/health";
import { getExecutions } from "../../../services/api/executions";
import { getArtifacts } from "../../../services/api/artifacts";

function getResponseData(response) {
  return response?.data ?? response;
}

function normalizeListResponse(response) {
  const data = getResponseData(response);

  if (Array.isArray(data)) {
    return {
      items: data,
      total: data.length,
    };
  }

  return {
    items: Array.isArray(data?.items) ? data.items : [],
    total:
      typeof data?.total === "number"
        ? data.total
        : Array.isArray(data?.items)
          ? data.items.length
          : 0,
  };
}

function normalizeHealthResponse(response) {
  const data = getResponseData(response);

  return {
    status: data?.status ?? null,
    database: data?.database ?? null,
    service: data?.service ?? null,
  };
}

function getExecutionStatusCounts(executions) {
  return executions.reduce(
    (counts, execution) => {
      const status = String(execution?.status ?? "UNKNOWN").toUpperCase();

      counts.total += 1;

      if (
        status === "COMPLETED" ||
        status === "SUCCESS" ||
        status === "SUCCEEDED"
      ) {
        counts.completed += 1;
      } else if (status === "FAILED" || status === "ERROR") {
        counts.failed += 1;
      } else if (status === "RUNNING" || status === "IN_PROGRESS") {
        counts.running += 1;
      } else if (status === "QUEUED" || status === "PENDING") {
        counts.pending += 1;
      }

      return counts;
    },
    {
      total: 0,
      completed: 0,
      failed: 0,
      running: 0,
      pending: 0,
    },
  );
}

function sortByDateDescending(items, field = "created_at") {
  return [...items].sort((first, second) => {
    const firstTime = new Date(first?.[field] ?? 0).getTime();
    const secondTime = new Date(second?.[field] ?? 0).getTime();

    return secondTime - firstTime;
  });
}

function buildActivity(executions, artifacts) {
  const executionActivity = executions.map((execution) => ({
    id: `execution-${execution.id}`,
    type: "execution",
    title: "Transformation execution",
    subtitle: execution.status
      ? `Execution ${String(execution.status).toLowerCase()}`
      : "Transformation execution",
    status: execution.status ?? "UNKNOWN",
    timestamp: execution.created_at,
    resourceId: execution.id,
  }));

  const artifactActivity = artifacts.map((artifact) => ({
    id: `artifact-${artifact.id}`,
    type: "artifact",
    title: artifact.title || artifact.artifact_type || "Generated artifact",
    subtitle: artifact.artifact_type
      ? `Generated ${artifact.artifact_type.replaceAll("_", " ")}`
      : "Generated artifact",
    status: artifact.status ?? "GENERATED",
    timestamp: artifact.created_at,
    resourceId: artifact.id,
  }));

  return [...executionActivity, ...artifactActivity]
    .sort((first, second) => {
      const firstTime = new Date(first.timestamp ?? 0).getTime();
      const secondTime = new Date(second.timestamp ?? 0).getTime();

      return secondTime - firstTime;
    })
    .slice(0, 10);
}

function buildHealthState(health) {
  const backendHealthy = health.status === "healthy";
  const databaseHealthy = health.database === "healthy";

  return {
    backendHealthy,
    databaseHealthy,
    healthy: backendHealthy && databaseHealthy,
  };
}

export async function getDashboardData() {
  const [healthResult, executionsResult, artifactsResult] =
    await Promise.allSettled([getHealth(), getExecutions(), getArtifacts()]);

  const health =
    healthResult.status === "fulfilled"
      ? normalizeHealthResponse(healthResult.value)
      : {
          status: null,
          database: null,
          service: null,
        };

  const executionData =
    executionsResult.status === "fulfilled"
      ? normalizeListResponse(executionsResult.value)
      : {
          items: [],
          total: 0,
        };

  const artifactData =
    artifactsResult.status === "fulfilled"
      ? normalizeListResponse(artifactsResult.value)
      : {
          items: [],
          total: 0,
        };

  const executionStatusCounts = getExecutionStatusCounts(executionData.items);

  const healthState = buildHealthState(health);

  const errors = [];

  if (healthResult.status === "rejected") {
    errors.push({
      resource: "health",
      message:
        healthResult.reason?.message || "Unable to load platform health.",
    });
  }

  if (executionsResult.status === "rejected") {
    errors.push({
      resource: "executions",
      message: executionsResult.reason?.message || "Unable to load executions.",
    });
  }

  if (artifactsResult.status === "rejected") {
    errors.push({
      resource: "artifacts",
      message: artifactsResult.reason?.message || "Unable to load artifacts.",
    });
  }

  return {
    health,
    healthState,

    executions: {
      items: executionData.items,
      total: executionData.total,
      statusCounts: executionStatusCounts,
    },

    artifacts: {
      items: artifactData.items,
      total: artifactData.total,
    },

    recentExecutions: sortByDateDescending(executionData.items).slice(0, 5),

    recentArtifacts: sortByDateDescending(artifactData.items).slice(0, 5),

    activity: buildActivity(executionData.items, artifactData.items),

    errors,
    partialFailure: errors.length > 0,
  };
}
