import { apiClient } from "./client";

export function getDevelopmentWorkflow({
  projectId,
  transformationType = "executive_summary",
}) {
  const params = new URLSearchParams({
    project_id: projectId,
    transformation_type: transformationType,
  });

  return apiClient.post(`/development/workflow?${params.toString()}`);
}
