import { apiClient } from "./client";

/**
 * Create and start an execution for a transformation.
 *
 * @param {Object} execution
 * @param {string} execution.transformation_id
 * @param {string} execution.workflow_id
 * @param {Object} [execution.execution_context]
 * @returns {Promise<Object>}
 */
export function createExecution({
  transformationId,
  workflowId,
  executionContext = {},
}) {
  return apiClient.post("/executions", {
    transformation_id: transformationId,
    workflow_id: workflowId,
    execution_context: executionContext,
  });
}

/**
 * Get a single execution by ID.
 *
 * @param {string} executionId
 * @returns {Promise<Object>}
 */
export function getExecution(executionId) {
  return apiClient.get(`/executions/${executionId}`);
}

/**
 * List executions.
 *
 * @param {Object} [params]
 * @returns {Promise<Object|Array>}
 */
export function getExecutions(params = {}) {
  const searchParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      searchParams.set(key, String(value));
    }
  });

  const query = searchParams.toString();

  return apiClient.get(query ? `/executions?${query}` : "/executions");
}
