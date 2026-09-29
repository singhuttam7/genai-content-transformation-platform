import { apiClient } from "./client";

/**
 * Fetch the current backend and database health status.
 */
export function getHealth() {
  return apiClient.get("/health");
}
