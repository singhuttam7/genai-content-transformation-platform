import { apiClient } from "./client";

export function getDevelopmentWorkspace() {
  return apiClient.get("/development/workspace");
}
